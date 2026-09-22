"""compaction_c.compact with a fake summarize_fn and a whitespace tokenizer."""

from __future__ import annotations

import pytest

from benchmark.bineval.run_reader import build_prompt
from benchmark.mcbuild_bench import compaction_c
from benchmark.mcbuild_bench.compaction_c import (
    SUMMARIZE_INSTRUCTION,
    compact,
    scaffold_reserve,
    summary_cap_tokens,
)
from benchmark.mcbuild_bench.windows import (
    assemble_context,
    build_window,
    count_tokens,
    render_round_trip,
    summary_block,
)


class WsTok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


TOK = WsTok()


def _rt(idx: int, n_words: int) -> dict:
    words = " ".join("w%d_%d" % (idx, k) for k in range(n_words))
    return {"idx": idx, "human": "ask%d" % idx, "events": [{"kind": "text", "text": words}]}


class FakeSummarizer:
    """Returns a summary of exactly ``out_words`` words; records every call."""

    def __init__(self, out_words: int = 4) -> None:
        self.out_words = out_words
        self.calls: list[tuple[str, int]] = []

    def __call__(self, text: str, *, cap_tokens: int) -> dict:
        self.calls.append((text, cap_tokens))
        n = len(self.calls)
        return {
            "text": " ".join("sum%d_%d" % (n, k) for k in range(self.out_words)),
            "prompt_tokens": count_tokens(TOK, text),
            "completion_tokens": self.out_words,
            "wall_ms": 12.5,
        }


def test_no_call_when_everything_fits() -> None:
    rts = [_rt(i, 3) for i in range(4)]
    fake = FakeSummarizer()
    res = compact(rts, 1000, TOK, fake, render_round_trip, question="q?")
    assert res["n_calls"] == 0 and res["calls"] == []
    assert res["summary"] == ""
    assert res["recent_idx"] == [0, 1, 2, 3]


def test_number_of_calls_and_final_window_fits() -> None:
    # each rendered RT = "### Human" (2) + "askN" (1) + "### text" (2) + 10 words = 15 tokens
    rts = [_rt(i, 10) for i in range(10)]
    # reserve = run_reader scaffold + "### Summary" header + question allowance,
    # measured with the same tokenizer (what run_arms must do on the pod).
    scaffold = count_tokens(TOK, build_prompt(assemble_context(None, ["### Summary"]), "q?"))
    W = scaffold + 60              # 60 tokens for bodies -> 4 RTs fit (60), the 5th overflows
    fake = FakeSummarizer(out_words=4)
    res = compact(rts, W, TOK, fake, render_round_trip, question="q?")
    # pass 1: RTs 0-3 (60) -> rt4 overflows -> call 1 (summary 4 tokens)
    # then summary(4)+rt4..rt6 = 4+45 = 49; rt7 -> 64 > 60 -> call 2; rt7..rt9 = 4+45 = 49
    assert res["n_calls"] == 2
    assert [c["older_round_trips"] for c in res["calls"]] == [[0, 1, 2, 3], [4, 5, 6]]
    assert res["recent_idx"] == [7, 8, 9]
    assert all(cap == summary_cap_tokens(W) == W // 4 for _, cap in fake.calls)
    for c in res["calls"]:
        assert set(c) >= {"prompt_tokens", "completion_tokens", "wall_ms"}
        assert c["completion_tokens"] == 4 and c["wall_ms"] == 12.5
    # the second call saw the first summary plus RTs 4-6
    assert fake.calls[1][0].startswith("sum1_0 sum1_1 sum1_2 sum1_3")
    # final window composed through build_window (arm C) with the same tokenizer fits W
    recent = [rt for rt in rts if rt["idx"] in res["recent_idx"]]
    win = build_window("C", W, None, [summary_block(res["summary"])] + recent, TOK, "q?")
    assert win["window_tokens"] <= W
    assert res["summary"] in win["prompt_context"]
    assert win["n_recent_rts"] == 3


def test_oversized_single_round_trip_is_folded_not_dropped() -> None:
    rts = [_rt(0, 5), _rt(1, 500), _rt(2, 5)]
    fake = FakeSummarizer(out_words=3)
    res = compact(rts, 100, TOK, fake, render_round_trip, question="q?")
    # rt1 cannot fit: call 1 folds rt0, call 2 folds rt1 itself; rt2 stays recent
    assert res["n_calls"] == 2
    assert [c["older_round_trips"] for c in res["calls"]] == [[0], [1]]
    assert res["recent_idx"] == [2]
    assert "w1_499" in fake.calls[1][0]


def test_summarizer_cap_violation_stops() -> None:
    rts = [_rt(0, 5), _rt(1, 500)]
    with pytest.raises(RuntimeError, match="cap"):
        compact(rts, 100, TOK, FakeSummarizer(out_words=400), render_round_trip, question="q?")


def test_bad_budget_rejected() -> None:
    with pytest.raises(ValueError):
        compact([], 0, TOK, FakeSummarizer(), render_round_trip, question="q?")
    with pytest.raises(ValueError):
        compact([], 10, TOK, FakeSummarizer(), render_round_trip, question=None)  # type: ignore[arg-type]


def test_instruction_is_design_section_8() -> None:
    text = SUMMARIZE_INSTRUCTION.format(cap=2000, older_log="LOG")
    assert text.startswith(
        "Summarize the following conversation log for later reference. Keep every "
        "concrete value, name, path, decision and instruction. Plain text, at most "
        "2000 tokens.\n\nLOG"
    )


# -- F6: no re-summarization of the summary alone -----------------------------------


def test_two_consecutive_oversized_round_trips_make_exactly_two_calls() -> None:
    rts = [_rt(0, 500), _rt(1, 500)]
    fake = FakeSummarizer(out_words=3)
    res = compact(rts, 100, TOK, fake, render_round_trip, question="q?")
    # rt0 folded (call 1), rt1 folded (call 2); never a call over the summary alone
    assert res["n_calls"] == 2
    assert [c["older_round_trips"] for c in res["calls"]] == [[0], [1]]
    assert all(c["older_round_trips"] for c in res["calls"])
    assert res["recent_idx"] == []
    assert "w1_499" in fake.calls[1][0]


# -- F5: reserve = the full scaffold build_window counts ----------------------------


def test_scaffold_reserve_matches_build_window_with_an_empty_summary() -> None:
    q = "what is the port?"
    expected = count_tokens(TOK, build_prompt(assemble_context(None, ["### Summary\n"]), q))
    assert scaffold_reserve(TOK, q) == expected
    # build_window on an empty summary and no round trips is exactly the reserve
    win = build_window("C", 1000, None, [summary_block("")], TOK, q)
    assert win["window_tokens"] == scaffold_reserve(TOK, q)


class NonAdditiveTok:
    """Token count = non-space chars // 3: concatenation changes the count, so an
    additive budget (sum of parts) and the exact count of the final string differ."""

    def __call__(self, text, add_special_tokens=False):
        n = len(text.replace(" ", "")) // 3
        return {"input_ids": list(range(n))}


def test_non_additive_tokenizer_declared_recent_equals_build_window(monkeypatch) -> None:
    """Item 5: compaction's fit test counts the EXACT final string build_window
    counts, so recent_idx always equals build_window(...)["n_recent_rts"]."""
    tok = NonAdditiveTok()
    q = "what is the port?"
    mismatches = 0
    for n_words, W in [(n, W) for n in range(3, 14) for W in range(120, 301, 7)]:
        rts = [_rt(i, n_words + (i % 3)) for i in range(12)]
        res = compact(rts, W, tok, FakeSummarizer(out_words=5), render_round_trip, question=q)
        recent = [rt for rt in rts if rt["idx"] in res["recent_idx"]]
        win = build_window("C", W, None, [summary_block(res["summary"])] + recent, tok, q)
        assert win["n_recent_rts"] == len(res["recent_idx"]), (n_words, W, res["recent_idx"])
        assert win["window_tokens"] <= W
        assert res["summary"] in win["prompt_context"]
        # the additive approximation would have declared a different recent set here
        body = "\n\n".join(render_round_trip(rt) for rt in recent)
        additive = count_tokens(tok, res["summary"]) + count_tokens(tok, body)
        exact = count_tokens(tok, win["prompt"])
        mismatches += int(additive + scaffold_reserve(tok, q) != exact)
    assert mismatches > 0  # the fake tokenizer really is non-additive


# -- item 12: CLI validation before any model import ---------------------------------


def test_cli_invalid_values_exit_before_model_import(monkeypatch, tmp_path) -> None:
    import sys

    from benchmark.bineval import run_reader

    def boom(*a, **k):
        raise AssertionError("load_reader must not be called")

    monkeypatch.setattr(run_reader, "load_reader", boom)
    monkeypatch.setitem(sys.modules, "torch", None)  # importing torch would raise ImportError
    q = tmp_path / "q.json"
    s = tmp_path / "s.json"
    q.write_text("[]", encoding="utf-8")
    s.write_text('{"round_trips": []}', encoding="utf-8")
    base = ["--model-id", "m", "--out", str(tmp_path / "c.json"), "--questions", str(q), "--session", str(s),
            "--gpu-csv", str(tmp_path / "gpu.csv")]
    with pytest.raises(SystemExit, match="9000"):
        compaction_c.main(["--W", "9000"] + base)
    with pytest.raises(SystemExit, match="file not found"):
        compaction_c.main(["--W", "8000", "--model-id", "m", "--out", str(tmp_path / "c.json"),
                           "--questions", str(tmp_path / "absent.json"), "--session", str(s),
                           "--gpu-csv", str(tmp_path / "gpu.csv")])
    # item J: the GPU sampler CSV is required
    with pytest.raises(SystemExit):
        compaction_c.main(["--W", "8000"] + base[:-2])
    # H22 (c): a malformed --exclude-rt is refused before torch is imported
    with pytest.raises(SystemExit, match="exclude-rt"):
        compaction_c.main(["--W", "8000", "--exclude-rt", "abc"] + base)
    assert compaction_c.build_parser().parse_args(["--W", "8000"] + base).exclude_rt == "36"


# -- item J (2026-09-18): GPU sampler + real completion tokens per call ------------------


def _write_power_csv(path, *, seconds: float = 60.0) -> None:
    import time as _time
    from datetime import datetime

    lines = ["timestamp, utilization.gpu [%], utilization.memory [%], memory.used [MiB], power.draw [W]"]
    now = _time.time()
    t = now - 3.0
    while t < now + seconds:
        dt = datetime.fromtimestamp(t)
        lines.append("%s.%03d, 50 %%, 10 %%, 100 MiB, 200.00 W"
                     % (dt.strftime("%Y/%m/%d %H:%M:%S"), dt.microsecond // 1000))
        t += 0.5
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _FakeReader:
    def __init__(self) -> None:
        self.tokenizer = TOK
        self.last_generated_tokens = None

    def generate(self, prompt: str) -> str:
        import time as _time

        _time.sleep(0.02)
        self.last_generated_tokens = 7
        return "a short summary of seven tokens here"


def test_reader_summarize_fn_records_tokens_wall_and_energy(tmp_path) -> None:
    csv = tmp_path / "gpu.csv"
    _write_power_csv(csv)
    llm = _FakeReader()
    fn = compaction_c.make_reader_summarize_fn(llm, gpu_csv=csv)
    res = fn("older log text", cap_tokens=100)
    prompt = SUMMARIZE_INSTRUCTION.format(cap=100, older_log="older log text")
    assert res["prompt_tokens"] == count_tokens(TOK, prompt)
    assert res["completion_tokens"] == 7  # llm.last_generated_tokens, not a re-tokenization
    assert res["wall_ms"] > 0
    assert res["energy_joules"] is not None and res["energy_joules"] > 0
    assert res["t_end"] > res["t_start"]


def test_compact_carries_energy_of_each_call() -> None:
    class Energetic(FakeSummarizer):
        def __call__(self, text, *, cap_tokens):
            out = super().__call__(text, cap_tokens=cap_tokens)
            out["energy_joules"] = 1.5
            out["t_start"] = 10.0
            out["t_end"] = 11.0
            return out

    rts = [_rt(0, 5), _rt(1, 500), _rt(2, 5)]
    res = compact(rts, 100, TOK, Energetic(out_words=3), render_round_trip, question="q?")
    assert res["n_calls"] == 2
    assert all(c["energy_joules"] == 1.5 and c["t_start"] == 10.0 and c["t_end"] == 11.0 for c in res["calls"])
    assert compaction_c.energy_total(res["calls"]) == 3.0
    assert compaction_c.energy_total([{"energy_joules": None}, {"energy_joules": 2.0}]) == 2.0
