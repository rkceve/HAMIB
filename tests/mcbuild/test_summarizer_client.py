"""Unit tests for benchmark/mcbuild_bench/summarizer_client.py (DESIGN.md §4, §8; D2).

The HTTP layer is replaced by an injected transport; no network access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmark.mcbuild_bench.errors import SummarizerStop
from benchmark.mcbuild_bench.summarizer_client import (
    INSTRUCTION,
    RETRY_SUFFIX,
    SummarizerClient,
)

EXCERPT = "The RCON port was changed to 25575 and the server runs Paper 1.21.8 on demo-server."
OK_TEXT = "RCON port is 25575; the server runs Paper 1.21.8 on demo-server."
LONG_TEXT = "x" * 121


def _completion(
    content: str, *, usage: dict[str, int] | None, finish_reason: str | None = "stop"
) -> bytes:
    choice: dict[str, Any] = {"index": 0, "message": {"role": "assistant", "content": content}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    envelope: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "summarizer",
        "choices": [choice],
    }
    if usage is not None:
        envelope["usage"] = usage
    return json.dumps(envelope).encode("utf-8")


USAGE = {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}


class FakeTransport:
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


def _client(tmp_path: Path, transport: FakeTransport) -> tuple[SummarizerClient, Path]:
    accounting = tmp_path / "summarizer_calls.jsonl"
    client = SummarizerClient(accounting_path=accounting, transport=transport)
    return client, accounting


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


ACCOUNTING_KEYS = ["ts", "prompt_tokens", "completion_tokens", "latency_ms", "retried", "chars"]


def test_design_strings_verbatim() -> None:
    assert INSTRUCTION == (
        "Rewrite the following excerpt as one self-contained statement of at most 120 "
        "characters, in the excerpt's language, keeping concrete values. Output the "
        "statement only."
    )
    assert RETRY_SUFFIX == "\n\nYour previous answer was too long. At most 120 characters."


def test_ok_reply(tmp_path: Path) -> None:
    transport = FakeTransport([(200, _completion("  " + OK_TEXT + "\n", usage=USAGE))])
    client, accounting = _client(tmp_path, transport)

    text = client.summarize(EXCERPT)

    assert text == OK_TEXT  # stripped
    assert len(transport.calls) == 1
    url, headers, body, timeout = transport.calls[0]
    assert url == "http://127.0.0.1:8001/v1/chat/completions"
    assert headers == {"Content-Type": "application/json"}
    assert timeout == 120.0
    assert json.loads(body) == {
        "model": "summarizer",
        "messages": [{"role": "user", "content": INSTRUCTION + "\n\n" + EXCERPT}],
        "temperature": 0,
        "max_tokens": 96,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    lines = _lines(accounting)
    assert len(lines) == 1
    assert list(lines[0]) == ACCOUNTING_KEYS
    assert lines[0]["prompt_tokens"] == 80
    assert lines[0]["completion_tokens"] == 20
    assert lines[0]["retried"] is False
    assert lines[0]["chars"] == len(OK_TEXT)
    assert isinstance(lines[0]["latency_ms"], float)
    assert lines[0]["ts"].endswith("+00:00")


def test_exactly_120_chars_is_accepted(tmp_path: Path) -> None:
    text120 = "y" * 120
    transport = FakeTransport([(200, _completion(text120, usage=USAGE))])
    client, _ = _client(tmp_path, transport)
    assert client.summarize(EXCERPT) == text120
    assert len(transport.calls) == 1


def test_over_120_then_ok_records_retry(tmp_path: Path) -> None:
    transport = FakeTransport(
        [(200, _completion(LONG_TEXT, usage=USAGE)), (200, _completion(OK_TEXT, usage=USAGE))]
    )
    client, accounting = _client(tmp_path, transport)

    text = client.summarize(EXCERPT)

    assert text == OK_TEXT
    assert len(transport.calls) == 2
    first = json.loads(transport.calls[0][2])["messages"][0]["content"]
    second = json.loads(transport.calls[1][2])["messages"][0]["content"]
    assert first == INSTRUCTION + "\n\n" + EXCERPT
    assert second == INSTRUCTION + "\n\n" + EXCERPT + RETRY_SUFFIX

    lines = _lines(accounting)
    assert len(lines) == 2
    assert lines[0]["retried"] is False and lines[0]["chars"] == 121
    assert lines[1]["retried"] is True and lines[1]["chars"] == len(OK_TEXT)


def test_over_120_twice_raises(tmp_path: Path) -> None:
    other_long = "z" * 200
    transport = FakeTransport(
        [(200, _completion(LONG_TEXT, usage=USAGE)), (200, _completion(other_long, usage=USAGE))]
    )
    client, accounting = _client(tmp_path, transport)

    with pytest.raises(SummarizerStop) as info:
        client.summarize(EXCERPT)

    message = str(info.value)
    assert LONG_TEXT in message and other_long in message  # both replies reported
    assert len(transport.calls) == 2
    lines = _lines(accounting)
    assert [ln["retried"] for ln in lines] == [False, True]
    assert [ln["chars"] for ln in lines] == [121, 200]


def test_empty_reply_retries_then_raises(tmp_path: Path) -> None:
    transport = FakeTransport(
        [(200, _completion("   \n", usage=USAGE)), (200, _completion("", usage=USAGE))]
    )
    client, accounting = _client(tmp_path, transport)
    with pytest.raises(SummarizerStop):
        client.summarize(EXCERPT)
    assert len(transport.calls) == 2
    assert [ln["chars"] for ln in _lines(accounting)] == [0, 0]


def test_missing_usage_raises(tmp_path: Path) -> None:
    transport = FakeTransport([(200, _completion(OK_TEXT, usage=None))])
    client, accounting = _client(tmp_path, transport)
    with pytest.raises(SummarizerStop, match="usage"):
        client.summarize(EXCERPT)
    lines = _lines(accounting)
    assert len(lines) == 1
    assert lines[0]["prompt_tokens"] is None and lines[0]["completion_tokens"] is None
    assert lines[0]["chars"] is None


def test_partial_usage_raises(tmp_path: Path) -> None:
    transport = FakeTransport([(200, _completion(OK_TEXT, usage={"prompt_tokens": 80}))])
    client, _ = _client(tmp_path, transport)
    with pytest.raises(SummarizerStop, match="usage"):
        client.summarize(EXCERPT)


def test_missing_content_raises(tmp_path: Path) -> None:
    envelope = {"choices": [{"index": 0, "message": {"role": "assistant"}}], "usage": USAGE}
    transport = FakeTransport([(200, json.dumps(envelope).encode())])
    client, _ = _client(tmp_path, transport)
    with pytest.raises(SummarizerStop, match="content"):
        client.summarize(EXCERPT)


def test_non_2xx_raises(tmp_path: Path) -> None:
    transport = FakeTransport([(503, b"model loading")])
    client, accounting = _client(tmp_path, transport)
    with pytest.raises(SummarizerStop, match="503"):
        client.summarize(EXCERPT)
    assert len(transport.calls) == 1
    lines = _lines(accounting)
    assert len(lines) == 1 and lines[0]["prompt_tokens"] is None


def test_base_url_and_model_are_used(tmp_path: Path) -> None:
    transport = FakeTransport([(200, _completion(OK_TEXT, usage=USAGE))])
    client = SummarizerClient(
        base_url="http://10.0.0.5:9000/v1/",
        model="qwen-node",
        accounting_path=tmp_path / "s.jsonl",
        timeout_s=30,
        transport=transport,
    )
    client.summarize(EXCERPT)
    url, _, body, timeout = transport.calls[0]
    assert url == "http://10.0.0.5:9000/v1/chat/completions"
    assert json.loads(body)["model"] == "qwen-node"
    assert timeout == 30.0


def test_blank_excerpt_rejected(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, FakeTransport([]))
    with pytest.raises(ValueError):
        client.summarize("   ")


def test_missing_content_still_records_usage(tmp_path: Path) -> None:
    """Item 8: a 2xx whose content fails validation still books its usage tokens."""
    envelope = {"choices": [{"index": 0, "message": {"role": "assistant"}}], "usage": USAGE}
    transport = FakeTransport([(200, json.dumps(envelope).encode())])
    client, accounting = _client(tmp_path, transport)
    with pytest.raises(SummarizerStop, match="content"):
        client.summarize(EXCERPT)
    lines = _lines(accounting)
    assert len(lines) == 1
    assert lines[0]["prompt_tokens"] == 80 and lines[0]["completion_tokens"] == 20
    assert lines[0]["chars"] is None


def test_accounting_append_is_thread_safe(tmp_path: Path) -> None:
    """Item 14: 50 threads x 10 appends -> exactly 500 well-formed JSON lines."""
    import threading

    client, accounting = _client(tmp_path, FakeTransport([]))

    def work() -> None:
        for i in range(10):
            client._record(80, i, 0.5, False, 40)

    threads = [threading.Thread(target=work) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    raw = accounting.read_text(encoding="utf-8").splitlines()
    assert len(raw) == 500
    parsed = [json.loads(line) for line in raw]
    assert all(set(p) == {"ts", "prompt_tokens", "completion_tokens", "latency_ms", "retried", "chars"} for p in parsed)


# -- item I (2026-09-18): node text validation ----------------------------------------


@pytest.mark.parametrize("bad", [
    "RCON port is 25575;\nthe server runs Paper 1.21.8.",   # two lines
    "[SN] RCON port is 25575",                                # marker syntax
    "The port [PN2.0] is 25575",
    "[RN] a satellite line",
    "<CONTEXT> port 25575",
    "port 25575 </CONTEXT>",
])
def test_invalid_node_text_is_retried_once_then_stops(tmp_path: Path, bad: str) -> None:
    transport = FakeTransport(
        [(200, _completion(bad, usage=USAGE)), (200, _completion(OK_TEXT, usage=USAGE))]
    )
    client, accounting = _client(tmp_path, transport)
    assert client.summarize(EXCERPT) == OK_TEXT
    assert len(transport.calls) == 2
    assert [ln["retried"] for ln in _lines(accounting)] == [False, True]
    transport = FakeTransport(
        [(200, _completion(bad, usage=USAGE)), (200, _completion(bad, usage=USAGE))]
    )
    client, _ = _client(tmp_path, transport)
    with pytest.raises(SummarizerStop):
        client.summarize(EXCERPT)


@pytest.mark.parametrize("reason", ["length", None])
def test_finish_reason_other_than_stop_is_retried_then_stops(tmp_path: Path, reason) -> None:
    transport = FakeTransport(
        [(200, _completion(OK_TEXT, usage=USAGE, finish_reason=reason)),
         (200, _completion(OK_TEXT, usage=USAGE))]
    )
    client, _ = _client(tmp_path, transport)
    assert client.summarize(EXCERPT) == OK_TEXT
    assert len(transport.calls) == 2
    transport = FakeTransport(
        [(200, _completion(OK_TEXT, usage=USAGE, finish_reason=reason)),
         (200, _completion(OK_TEXT, usage=USAGE, finish_reason="length"))]
    )
    client, _ = _client(tmp_path, transport)
    with pytest.raises(SummarizerStop, match="finish_reason"):
        client.summarize(EXCERPT)
