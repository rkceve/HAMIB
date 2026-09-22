"""Unit tests for benchmark/mcbuild_bench/jev_client.py against the DESIGN.md §3 JSON verbatim.

The HTTP layer is replaced by an injected transport; no network access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmark.mcbuild_bench.errors import JevStop
from benchmark.mcbuild_bench.jev_client import (
    JEV_ENDPOINT,
    JevClient,
    argmax_level,
    choice_of,
    cost_usd,
    noul_of,
    read_price_per_mtok,
)

FIXTURE = Path(__file__).parent / "fixtures" / "jev_examples.json"
API_KEY = "sk-test-SECRET-KEY-0123456789"


@pytest.fixture()
def examples() -> dict[str, Any]:
    with FIXTURE.open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


class FakeTransport:
    """Scripted transport: returns the queued (status, body) pairs in order and records calls."""

    def __init__(self, responses: list[tuple[int, bytes]]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, str], bytes, float]] = []

    def __call__(
        self, url: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> tuple[int, bytes]:
        self.calls.append((url, headers, body, timeout))
        if not self._responses:
            raise AssertionError("transport called more times than scripted")
        return self._responses.pop(0)


def _client(
    tmp_path: Path, transport: FakeTransport, sleeps: list[float], **kwargs: Any
) -> tuple[JevClient, Path]:
    accounting = tmp_path / "jev_calls.jsonl"
    client = JevClient(
        API_KEY,
        accounting_path=accounting,
        transport=transport,
        sleep_fn=sleeps.append,
        **kwargs,
    )
    return client, accounting


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


ACCOUNTING_KEYS = [
    "ts", "question_ids", "question_types", "state_chars", "input_tokens",
    "output_tokens", "latency_ms", "http_status", "retry_index", "answers",
]


# ------------------------------------------------------------------ success path
def test_ask_parses_all_three_answer_types(tmp_path: Path, examples: dict[str, Any]) -> None:
    request, response = examples["request"], examples["response"]
    transport = FakeTransport([(200, json.dumps(response).encode("utf-8"))])
    sleeps: list[float] = []
    client, accounting = _client(tmp_path, transport, sleeps)

    result = client.ask(request["state"], request["questions"])

    assert result["answers"] == response["answers"]
    assert result["usage"] == {"input_tokens": 312, "output_tokens": 48}
    assert result["http_status"] == 200
    assert result["retries"] == 0
    assert isinstance(result["latency_ms"], float) and result["latency_ms"] >= 0.0
    assert sleeps == []

    # the three answer types parse with the pure helpers
    assert noul_of(result["answers"]["is_urgent"]) == 0.92
    assert choice_of(result["answers"]["department"]) == "technical"
    assert argmax_level(result["answers"]["frustration"]) == 2

    # request wire format
    url, headers, body, timeout = transport.calls[0]
    assert url == JEV_ENDPOINT
    assert headers == {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    assert timeout == 60.0

    # one accounting line, all keys, tokens and answers filled
    lines = _lines(accounting)
    assert len(lines) == 1
    assert list(lines[0]) == ACCOUNTING_KEYS
    assert lines[0]["question_ids"] == ["department", "frustration", "is_urgent"]
    assert lines[0]["question_types"] == {
        "is_urgent": "noul", "department": "choice", "frustration": "score",
    }
    assert lines[0]["state_chars"] == len(request["state"])
    assert lines[0]["input_tokens"] == 312
    assert lines[0]["output_tokens"] == 48
    assert lines[0]["http_status"] == 200
    assert lines[0]["retry_index"] == 0
    assert lines[0]["answers"] == response["answers"]
    assert lines[0]["ts"].endswith("+00:00")


def test_request_body_is_byte_exact_json(tmp_path: Path, examples: dict[str, Any]) -> None:
    request, response = examples["request"], examples["response"]
    transport = FakeTransport([(200, json.dumps(response).encode("utf-8"))])
    client, _ = _client(tmp_path, transport, [])

    client.ask(request["state"], request["questions"])

    body = transport.calls[0][2]
    expected = json.dumps(
        {"state": request["state"], "model": "jev-latest", "questions": request["questions"]}
    ).encode("utf-8")
    assert body == expected
    assert list(json.loads(body)) == ["state", "model", "questions"]


def test_api_key_never_written_to_accounting(tmp_path: Path, examples: dict[str, Any]) -> None:
    request, response = examples["request"], examples["response"]
    transport = FakeTransport(
        [(429, b"slow down"), (200, json.dumps(response).encode("utf-8"))]
    )
    client, accounting = _client(tmp_path, transport, [])
    client.ask(request["state"], request["questions"])
    text = accounting.read_text(encoding="utf-8")
    assert API_KEY not in text
    assert "SECRET" not in text
    assert "Bearer" not in text


# ------------------------------------------------------------------ retry / stop
def test_429_three_times_raises_after_backoff(tmp_path: Path, examples: dict[str, Any]) -> None:
    request = examples["request"]
    transport = FakeTransport([(429, b"rate"), (429, b"rate"), (429, b"rate")])
    sleeps: list[float] = []
    client, accounting = _client(tmp_path, transport, sleeps)

    with pytest.raises(JevStop, match="429"):
        client.ask(request["state"], request["questions"])

    assert len(transport.calls) == 3
    assert sleeps == [1.0, 4.0]  # [1, 4, 16] truncated: sleeps happen BETWEEN 3 attempts
    lines = _lines(accounting)
    assert len(lines) == 3
    assert [ln["http_status"] for ln in lines] == [429, 429, 429]
    assert [ln["retry_index"] for ln in lines] == [0, 1, 2]
    assert all(ln["input_tokens"] is None and ln["output_tokens"] is None for ln in lines)
    assert all(ln["answers"] is None for ln in lines)


def test_529_then_success_records_retries(tmp_path: Path, examples: dict[str, Any]) -> None:
    request, response = examples["request"], examples["response"]
    transport = FakeTransport(
        [(529, b"overloaded"), (529, b"overloaded"), (200, json.dumps(response).encode())]
    )
    sleeps: list[float] = []
    client, accounting = _client(tmp_path, transport, sleeps)

    result = client.ask(request["state"], request["questions"])

    assert result["retries"] == 2
    assert sleeps == [1.0, 4.0]
    lines = _lines(accounting)
    assert [ln["http_status"] for ln in lines] == [529, 529, 200]
    assert lines[2]["input_tokens"] == 312


def test_422_stops_immediately_with_body_excerpt(
    tmp_path: Path, examples: dict[str, Any]
) -> None:
    request = examples["request"]
    body = b'{"error": "validation failed: criteria"}' + b"x" * 1000
    transport = FakeTransport([(422, body)])
    sleeps: list[float] = []
    client, accounting = _client(tmp_path, transport, sleeps)

    with pytest.raises(JevStop) as info:
        client.ask(request["state"], request["questions"])

    message = str(info.value)
    assert "422" in message
    assert "validation failed" in message
    assert len(message) < 600  # only the first 500 chars of the body are quoted
    assert sleeps == []
    assert len(transport.calls) == 1
    lines = _lines(accounting)
    assert len(lines) == 1
    assert lines[0]["http_status"] == 422
    assert lines[0]["answers"] is None


@pytest.mark.parametrize("status", [401, 500, 503])
def test_other_non_2xx_stop_immediately(
    tmp_path: Path, examples: dict[str, Any], status: int
) -> None:
    request = examples["request"]
    transport = FakeTransport([(status, b"nope")])
    sleeps: list[float] = []
    client, accounting = _client(tmp_path, transport, sleeps)
    with pytest.raises(JevStop, match=str(status)):
        client.ask(request["state"], request["questions"])
    assert sleeps == []
    assert len(_lines(accounting)) == 1


def test_missing_question_id_in_answers_raises(
    tmp_path: Path, examples: dict[str, Any]
) -> None:
    request, response = examples["request"], examples["response"]
    partial = json.loads(json.dumps(response))
    del partial["answers"]["frustration"]
    transport = FakeTransport([(200, json.dumps(partial).encode("utf-8"))])
    client, accounting = _client(tmp_path, transport, [])

    with pytest.raises(JevStop, match="frustration"):
        client.ask(request["state"], request["questions"])
    lines = _lines(accounting)
    assert len(lines) == 1
    assert lines[0]["http_status"] == 200
    assert lines[0]["answers"] is None
    # H10: usage is parsed FIRST and recorded even though the answers failed.
    assert lines[0]["input_tokens"] == 312 and lines[0]["output_tokens"] == 48


@pytest.mark.parametrize(
    "mutate",
    [
        lambda env: env.pop("answers"),
    ],
)
def test_malformed_envelope_raises(
    tmp_path: Path, examples: dict[str, Any], mutate: Any
) -> None:
    request, response = examples["request"], examples["response"]
    env = json.loads(json.dumps(response))
    mutate(env)
    transport = FakeTransport([(200, json.dumps(env).encode("utf-8"))])
    client, _ = _client(tmp_path, transport, [])
    with pytest.raises(JevStop):
        client.ask(request["state"], request["questions"])


def test_usage_integral_float_is_accepted_as_int(
    tmp_path: Path, examples: dict[str, Any]
) -> None:
    """F4: ``usage.input_tokens: 312.0`` is an integer in disguise -> 312."""
    request, response = examples["request"], examples["response"]
    env = json.loads(json.dumps(response))
    env["usage"]["input_tokens"] = 312.0
    env["usage"]["output_tokens"] = 48.0
    transport = FakeTransport([(200, json.dumps(env).encode("utf-8"))])
    client, accounting = _client(tmp_path, transport, [])
    result = client.ask(request["state"], request["questions"])
    assert result["answers"] == response["answers"]
    assert result["usage"] == {"input_tokens": 312, "output_tokens": 48}
    assert isinstance(result["usage"]["input_tokens"], int)
    line = _lines(accounting)[0]
    assert line["input_tokens"] == 312 and line["output_tokens"] == 48


@pytest.mark.parametrize(
    "mutate",
    [
        lambda env: env.pop("usage"),
        lambda env: env["usage"].pop("input_tokens"),
        lambda env: env["usage"].__setitem__("input_tokens", "312"),
        lambda env: env["usage"].__setitem__("input_tokens", 312.5),
        lambda env: env["usage"].__setitem__("input_tokens", True),
        lambda env: env["usage"].__setitem__("input_tokens", None),
    ],
)
def test_non_integer_usage_stops_the_run_h10(
    tmp_path: Path, examples: dict[str, Any], mutate: Any
) -> None:
    """H10 / D5: a 2xx without integer usage is an accounting hole -> JevStop."""
    request, response = examples["request"], examples["response"]
    env = json.loads(json.dumps(response))
    mutate(env)
    transport = FakeTransport([(200, json.dumps(env).encode("utf-8"))])
    client, accounting = _client(tmp_path, transport, [])
    with pytest.raises(JevStop, match="usage"):
        client.ask(request["state"], request["questions"])
    line = _lines(accounting)[0]
    assert line["input_tokens"] is None and line["output_tokens"] is None
    assert line["answers"] is None
    assert line["http_status"] == 200


def test_answers_validation_failure_still_records_usage_h10(
    tmp_path: Path, examples: dict[str, Any]
) -> None:
    """H10: usage is parsed before the answers, so the accounting line carries
    the tokens even when the answers block is missing."""
    request, response = examples["request"], examples["response"]
    env = json.loads(json.dumps(response))
    env.pop("answers")
    transport = FakeTransport([(200, json.dumps(env).encode("utf-8"))])
    client, accounting = _client(tmp_path, transport, [])
    with pytest.raises(JevStop, match="answers"):
        client.ask(request["state"], request["questions"])
    line = _lines(accounting)[0]
    assert line["input_tokens"] == 312 and line["output_tokens"] == 48
    assert line["answers"] is None and line["http_status"] == 200


def test_accounting_append_is_thread_safe(tmp_path: Path) -> None:
    """Item 14: 50 threads x 10 appends -> exactly 500 well-formed JSON lines."""
    import threading

    client, accounting = _client(tmp_path, FakeTransport([]), [])

    def work() -> None:
        for i in range(10):
            client._record(["q"], {"q": "noul"}, 10, i, 1, 0.5, 200, 0, {"q": {"noul": 0.5}})

    threads = [threading.Thread(target=work) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    raw = accounting.read_text(encoding="utf-8").splitlines()
    assert len(raw) == 500
    parsed = [json.loads(line) for line in raw]
    assert all(set(p) == set(ACCOUNTING_KEYS) for p in parsed)


def test_non_json_2xx_body_raises(tmp_path: Path, examples: dict[str, Any]) -> None:
    request = examples["request"]
    transport = FakeTransport([(200, b"<html>gateway</html>")])
    client, _ = _client(tmp_path, transport, [])
    with pytest.raises(JevStop, match="not JSON"):
        client.ask(request["state"], request["questions"])


def test_transport_exception_raises_jevstop_without_retry(
    tmp_path: Path, examples: dict[str, Any]
) -> None:
    request = examples["request"]

    def boom(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
        raise TimeoutError("timed out")

    sleeps: list[float] = []
    accounting = tmp_path / "acc.jsonl"
    client = JevClient(API_KEY, accounting_path=accounting, transport=boom, sleep_fn=sleeps.append)
    with pytest.raises(JevStop, match="transport error"):
        client.ask(request["state"], request["questions"])
    assert sleeps == []
    lines = _lines(accounting)
    assert len(lines) == 1 and lines[0]["http_status"] is None


def test_constructor_rejects_bad_arguments(tmp_path: Path) -> None:
    acc = tmp_path / "a.jsonl"
    with pytest.raises(ValueError):
        JevClient("", accounting_path=acc)
    with pytest.raises(ValueError):
        JevClient(API_KEY, max_attempts=0, accounting_path=acc)
    with pytest.raises(ValueError):
        JevClient(API_KEY, max_attempts=5, accounting_path=acc)  # beyond the 1/4/16 schedule
    with pytest.raises(ValueError):
        JevClient(API_KEY, timeout_s=0, accounting_path=acc)


def test_ask_rejects_empty_inputs(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, FakeTransport([]), [])
    with pytest.raises(ValueError):
        client.ask("", {"q": {"type": "noul", "instructions": "x"}})
    with pytest.raises(ValueError):
        client.ask("state", {})
    with pytest.raises(ValueError):
        client.ask("state", {"q": {"instructions": "no type"}})


# ------------------------------------------------------------------ pure helpers
def test_argmax_level_on_design_example() -> None:
    answer = {"type": "score", "probabilities": {"0": 0.05, "1": 0.3, "2": 0.65}}
    assert argmax_level(answer) == 2


def test_argmax_level_missing_levels_are_zero_and_bound_by_n_levels() -> None:
    """Unified rule (F3): argmax over the keys PRESENT; a missing level is
    probability 0; at least one key; index < n_levels when given."""
    assert argmax_level({"type": "score", "probabilities": {"3": 0.4, "1": 0.3}}) == 3
    assert argmax_level({"type": "score", "probabilities": {"4": 1.0}}, n_levels=5) == 4
    with pytest.raises(JevStop):
        argmax_level({"type": "score", "probabilities": {"5": 1.0}}, n_levels=5)
    with pytest.raises(JevStop):
        argmax_level({"type": "score", "probabilities": {"2": 0.65}}, n_levels=2)
    with pytest.raises(JevStop):
        argmax_level({"type": "score", "probabilities": {"0": None}})


def test_argmax_level_tie_goes_to_lower_index() -> None:
    assert argmax_level({"type": "score", "probabilities": {"0": 0.5, "1": 0.5}}) == 0
    # key order in the dict must not matter
    assert argmax_level({"type": "score", "probabilities": {"1": 0.5, "0": 0.5}}) == 0
    assert argmax_level({"type": "score", "probabilities": {"2": 0.4, "0": 0.2, "1": 0.4}}) == 1


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "choice", "probabilities": {"0": 1.0}},
        {"type": "score"},
        {"type": "score", "probabilities": {}},
        {"type": "score", "probabilities": {"a": 1.0}},
        {"type": "score", "probabilities": {"0": "high"}},
        {"probabilities": {"0": 1.0}},
    ],
)
def test_argmax_level_rejects_malformed(answer: dict[str, Any]) -> None:
    with pytest.raises(JevStop):
        argmax_level(answer)


def test_choice_of(examples: dict[str, Any]) -> None:
    assert choice_of(examples["response"]["answers"]["department"]) == "technical"
    with pytest.raises(JevStop):
        choice_of({"type": "choice"})
    with pytest.raises(JevStop):
        choice_of({"type": "choice", "choice": ""})
    with pytest.raises(JevStop):
        choice_of({"type": "noul", "noul": 0.9})


def test_noul_of(examples: dict[str, Any]) -> None:
    assert noul_of(examples["response"]["answers"]["is_urgent"]) == 0.92
    assert noul_of({"type": "noul", "noul": 0}) == 0.0
    assert noul_of({"type": "noul", "noul": 1}) == 1.0
    with pytest.raises(JevStop):
        noul_of({"type": "noul"})
    with pytest.raises(JevStop):
        noul_of({"type": "noul", "noul": 1.5})
    with pytest.raises(JevStop):
        noul_of({"type": "noul", "noul": True})
    with pytest.raises(JevStop):
        noul_of({"type": "choice", "choice": "x"})


def test_price_and_cost() -> None:
    assert read_price_per_mtok() == 0.042
    assert cost_usd(0) == 0.0
    assert cost_usd(1_000_000) == pytest.approx(0.042)
    assert cost_usd(312) == pytest.approx(312 * 0.042 / 1_000_000)
    with pytest.raises(ValueError):
        cost_usd(-1)


# ------------------------------------------------------------------ item 5: probability validation
@pytest.mark.parametrize(
    "probabilities",
    [
        {"0": -1, "4": -0.1},  # negative
        {"0": float("nan"), "1": 0.5},  # NaN
        {"0": float("inf")},  # inf
        {"0": 1.5},  # above 1
        {"0": 0.0, "1": 0.0},  # all zero
        {"0": 0.2, "7": 0.8},  # key outside the level table (n_levels=5)
    ],
)
def test_argmax_level_rejects_invalid_probabilities(probabilities: dict[str, Any]) -> None:
    with pytest.raises(JevStop):
        argmax_level({"type": "score", "probabilities": probabilities}, n_levels=5)


def test_argmax_level_out_of_range_key_stops_even_when_it_is_not_the_winner() -> None:
    # "7" is not the argmax, but it is not a level of a 5-level table either.
    with pytest.raises(JevStop, match="7"):
        argmax_level({"type": "score", "probabilities": {"0": 0.9, "7": 0.1}}, n_levels=5)
    # without n_levels the key set is unconstrained (the caller has no table)
    assert argmax_level({"type": "score", "probabilities": {"0": 0.9, "7": 0.1}}) == 0


def test_noul_of_rejects_nan_and_inf() -> None:
    with pytest.raises(JevStop):
        noul_of({"type": "noul", "noul": float("nan")})
    with pytest.raises(JevStop):
        noul_of({"type": "noul", "noul": float("inf")})
    with pytest.raises(JevStop):
        noul_of({"type": "noul", "noul": -0.0001})


def test_choice_of_validates_probabilities_against_the_offered_options() -> None:
    options = ["billing", "technical", "sales"]
    ok = {
        "type": "choice",
        "choice": "technical",
        "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07},
    }
    assert choice_of(ok, options=options) == "technical"
    assert choice_of(ok) == "technical"  # no options given: keys unconstrained
    with pytest.raises(JevStop, match="legal"):
        choice_of({**ok, "probabilities": {"billing": 0.1, "legal": 0.9}}, options=options)
    with pytest.raises(JevStop):
        choice_of({**ok, "probabilities": {"billing": float("nan")}}, options=options)
    with pytest.raises(JevStop):
        choice_of({**ok, "probabilities": {"billing": -0.5}}, options=options)
    with pytest.raises(JevStop):
        choice_of({**ok, "probabilities": {"billing": 0.0, "sales": 0.0}}, options=options)
    with pytest.raises(JevStop, match="not an offered option"):
        choice_of({**ok, "choice": "legal"}, options=options)
    # probabilities is optional for a choice answer (DESIGN 3: "if present")
    assert choice_of({"type": "choice", "choice": "sales"}, options=options) == "sales"
