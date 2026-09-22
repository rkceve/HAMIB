"""Item 6b (Astra round 3): build_cd.main() with the REAL JevClient and SummarizerClient
over injected fake transports that answer with DESIGN.md 3 / 4 shaped JSON.

Accounting files, manifest totals (item 4), the budget guard (item 1) and the
client validation (item 5) are exercised together on two synthetic round trips.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.mcbuild_bench import build_cd, gpu_sampler, jev_client, summarizer_client
from benchmark.mcbuild_bench.jev_client import JEV_ENDPOINT, JevClient
from benchmark.mcbuild_bench.summarizer_client import INSTRUCTION, SummarizerClient
from tests.mcbuild._fakes import TAG_DROP, TAG_PLANET, TAG_SUN, FakeJev
from tests.mcbuild.test_build_cd import SESSION, _NoGpuSampler


class JevTransport:
    """HTTP stand-in: decodes the wire body, answers with FakeJev's rules in the
    DESIGN.md 3 envelope ({"model", "answers", "usage"})."""

    def __init__(self, mutate=None) -> None:
        self.calls: list[dict] = []
        self.usage: list[dict[str, int]] = []
        self.rules = FakeJev()
        self.mutate = mutate

    def __call__(self, url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
        assert url == JEV_ENDPOINT and headers["Authorization"] == "Bearer sk-test"
        request = json.loads(body.decode("utf-8"))
        assert list(request) == ["state", "model", "questions"] and request["model"] == "jev-latest"
        self.calls.append(request)
        reply = self.rules.ask(request["state"], request["questions"])
        usage = {"input_tokens": 300 + len(request["state"]) // 3, "output_tokens": 7 * len(request["questions"])}
        self.usage.append(usage)
        envelope = {"model": "jev-latest", "answers": reply["answers"], "usage": usage}
        if self.mutate is not None:
            self.mutate(envelope, request)
        return 200, json.dumps(envelope).encode("utf-8")


class SummarizerTransport:
    """OpenAI-compatible stand-in for the vLLM summarizer (DESIGN.md 4)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.usage: list[dict[str, int]] = []

    def __call__(self, url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
        assert url.endswith("/v1/chat/completions")
        request = json.loads(body.decode("utf-8"))
        self.calls.append(request)
        content = request["messages"][0]["content"]
        assert content.startswith(INSTRUCTION + "\n\n")
        excerpt = content[len(INSTRUCTION) + 2:]
        for tag in (TAG_SUN, TAG_PLANET, TAG_DROP):
            excerpt = excerpt.replace(tag, "")
        text = " ".join(excerpt.split())[:120]
        usage = {"prompt_tokens": 40 + len(content) // 4, "completion_tokens": max(1, len(text) // 4)}
        self.usage.append(usage)
        envelope = {
            "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": usage,
        }
        return 200, json.dumps(envelope).encode("utf-8")


def _wire(monkeypatch, tmp_path: Path, jev_transport, summ_transport) -> list[str]:
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")
    monkeypatch.setattr(
        jev_client, "JevClient",
        lambda **kw: JevClient(transport=jev_transport, sleep_fn=lambda s: None, **kw),
    )
    monkeypatch.setattr(
        summarizer_client, "SummarizerClient",
        lambda **kw: SummarizerClient(transport=summ_transport, **kw),
    )
    monkeypatch.setattr(gpu_sampler, "GpuSampler", _NoGpuSampler)
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps(SESSION), encoding="utf-8")
    return [
        "--session", str(session_path),
        "--exclude-rt", "none",  # H22 (c): the two-round-trip fixture has no round trip 36
        "--out", str(tmp_path / "cd.json"),
        "--jev-accounting", str(tmp_path / "jev_calls.jsonl"),
        "--summarizer-accounting", str(tmp_path / "summarizer_calls.jsonl"),
        "--gpu-csv", str(tmp_path / "gpu.csv"),
        "--max-jev-requests", "1000",
        "--max-jev-input-tokens", "10000000",
    ]


def _lines(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_two_round_trips_with_real_clients_over_fake_transports(monkeypatch, tmp_path: Path) -> None:
    jev_t, summ_t = JevTransport(), SummarizerTransport()
    argv = _wire(monkeypatch, tmp_path, jev_t, summ_t)
    assert build_cd.main(argv + ["--run-id", "real1"]) == 0
    out = json.loads((tmp_path / "cd.json").read_text(encoding="utf-8"))
    assert "stopped" not in out
    s, m = out["summary"], out["manifest"]
    assert s["turns"] == 2 and s["dropped_chunks"] == 1 and s["total"] >= 3
    assert s["harness_quality"]["unparsed"] == 0 and s["harness_quality"]["node_fallback"] == 0

    # accounting files: one line per HTTP attempt, all with integer usage
    jev_lines = _lines(tmp_path / "jev_calls.real1.jsonl")
    summ_lines = _lines(tmp_path / "summarizer_calls.real1.jsonl")
    assert len(jev_lines) == len(jev_t.calls) == sum(s["harness_calls"].values())
    assert all(isinstance(ln["input_tokens"], int) and ln["http_status"] == 200 for ln in jev_lines)
    assert len(summ_lines) == len(summ_t.calls) == s["total"]  # kept chunks only, no retries
    assert all(ln["retried"] is False and ln["chars"] <= 120 for ln in summ_lines)

    # manifest totals are the sums over THIS run's files (item 4), not the counters
    assert m["jev_requests"] == len(jev_lines)
    assert m["jev_input_tokens_total"] == sum(u["input_tokens"] for u in jev_t.usage)
    assert m["jev_output_tokens_total"] == sum(u["output_tokens"] for u in jev_t.usage)
    assert m["jev_cost_usd"] == pytest.approx(m["jev_input_tokens_total"] * 0.042 / 1e6)
    assert m["summarizer_calls"] == len(summ_lines)
    assert m["summarizer_prompt_tokens"] == sum(u["prompt_tokens"] for u in summ_t.usage)
    assert m["summarizer_completion_tokens"] == sum(u["completion_tokens"] for u in summ_t.usage)
    # the budget guard saw the same requests and tokens
    assert m["jev_budget"] == {
        "max_requests": 1000, "max_input_tokens": 10000000,
        "requests_sent": len(jev_lines), "input_tokens": m["jev_input_tokens_total"],
    }
    assert m["capacities"]["max_satellite_nodes_per_planet"] == 1000
    assert m["jev_model"] == "jev-latest" and m["published_price_per_mtok"] == 0.042
    # the API key never reaches the accounting file
    text = (tmp_path / "jev_calls.real1.jsonl").read_text(encoding="utf-8")
    assert "sk-test" not in text


def test_invalid_probability_from_the_wire_stops_the_run_with_the_partial(monkeypatch, tmp_path: Path) -> None:
    def poison(envelope: dict, request: dict) -> None:
        # item 5: a negative probability in the first classification of round trip 1
        if "scale" in request["state"] and "detail" in envelope["answers"]:
            envelope["answers"]["detail"]["probabilities"]["4"] = -0.1

    jev_t, summ_t = JevTransport(mutate=poison), SummarizerTransport()
    argv = _wire(monkeypatch, tmp_path, jev_t, summ_t)
    assert build_cd.main(argv + ["--run-id", "real2"]) == 1
    out = json.loads((tmp_path / "cd.json").read_text(encoding="utf-8"))
    assert out["stopped"]["type"] == "JevStop" and "probabilit" in out["stopped"]["reason"]
    assert out["summary"]["turns"] == 1 and {r["created_turn"] for r in out["nodes"]} == {0}
    assert out["summary"]["harness_quality"]["unparsed"] == 1
    m = out["manifest"]
    jev_lines = _lines(tmp_path / "jev_calls.real2.jsonl")
    assert m["jev_requests"] == len(jev_lines) == len(jev_t.calls)
    # the poisoned request's tokens are still counted (usage parsed first, H10)
    assert m["jev_input_tokens_total"] == sum(u["input_tokens"] for u in jev_t.usage)


def test_request_budget_with_real_clients(monkeypatch, tmp_path: Path) -> None:
    jev_t, summ_t = JevTransport(), SummarizerTransport()
    argv = _wire(monkeypatch, tmp_path, jev_t, summ_t)
    argv = argv[:-4] + ["--max-jev-requests", "4", "--max-jev-input-tokens", "10000000"]
    assert build_cd.main(argv + ["--run-id", "real3"]) == 1
    out = json.loads((tmp_path / "cd.json").read_text(encoding="utf-8"))
    assert "budget exceeded: requests 5 > max 4" in out["stopped"]["reason"]
    assert len(jev_t.calls) == 4 and out["manifest"]["jev_requests"] == 4
    # round trip 0 costs exactly 4 requests: it is kept, round trip 1 never starts
    assert out["summary"]["turns"] == 1 and {r["created_turn"] for r in out["nodes"]} == {0}
    assert out["manifest"]["jev_budget"]["requests_sent"] == 4
