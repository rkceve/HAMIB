"""B7: backend transports.  No network call and no API key are used."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from management.harness.backends import (
    AnthropicJudge,
    FakeJudge,
    OpenAICompatJudge,
    call_with_retries,
    make_driver_fake_judge,
)
from management.harness.prompts import Q_EXTRACT, Q_SAME, Q_SUPPORTED


# -- FakeJudge ---------------------------------------------------------------


def test_fake_judge_rules_in_order_and_default() -> None:
    judge = FakeJudge({"同じ事柄": "yes", "話題": "no"}, default="???")
    assert judge.complete("AとBは同じ事柄ですか", max_tokens=8) == "yes"
    assert judge.complete("話題が変わりますか", max_tokens=8) == "no"
    assert judge.complete("something else", max_tokens=8) == "???"
    assert len(judge.prompts) == 3


def test_fake_judge_policy_wins_over_rules() -> None:
    judge = FakeJudge({"x": "rule"}, policy=lambda p: "policy" if "x" in p else None)
    assert judge.complete("x", max_tokens=8) == "policy"
    assert judge.complete("y", max_tokens=8) == "no"


def test_driver_fake_judge_echoes_the_chunk_as_one_statement() -> None:
    judge = make_driver_fake_judge(max_chars=120)
    prompt = Q_EXTRACT.format(text="この店はホシノ亭です。", max_chars=120)
    assert json.loads(judge.complete(prompt, max_tokens=64)) == ["この店はホシノ亭です。"]
    assert judge.complete("AとBは同じ事柄ですか", max_tokens=8) == "no"


def test_driver_fake_judge_truncates() -> None:
    judge = make_driver_fake_judge(max_chars=5)
    prompt = Q_EXTRACT.format(text="あ" * 50, max_chars=5)
    assert json.loads(judge.complete(prompt, max_tokens=64)) == ["あ" * 5]


def test_driver_fake_judge_says_yes_to_faithfulness_only() -> None:
    """H10: the statement IS the chunk, so it is trivially supported; every
    other yes/no stays "no" so the smoke run still lands on satellites."""
    judge = make_driver_fake_judge()
    supported = Q_SUPPORTED.format(text="a chunk", statement="a chunk")
    assert judge.complete(supported, max_tokens=8) == "yes"
    assert judge.complete(Q_SAME.format(a="x", b="y"), max_tokens=8) == "no"


# -- transport retry (H4) ----------------------------------------------------


class _Flaky:
    """Fails `n_failures` times with `exc`, then returns "yes"."""

    def __init__(self, n_failures: int, exc: BaseException) -> None:
        self.left = n_failures
        self.exc = exc
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.left > 0:
            self.left -= 1
            raise self.exc
        return "yes"


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://h", code, "boom", {}, None)  # type: ignore[arg-type]


def test_retry_backoff_delays_and_success() -> None:
    slept: list[float] = []
    flaky = _Flaky(2, RuntimeError("transient"))
    assert (
        call_with_retries(flaky, sleep_fn=slept.append) == "yes"
    )
    assert flaky.calls == 3
    assert slept == [0.5, 1.0]


def test_retry_gives_up_after_three_attempts() -> None:
    slept: list[float] = []
    flaky = _Flaky(5, RuntimeError("down"))
    with pytest.raises(RuntimeError):
        call_with_retries(flaky, sleep_fn=slept.append)
    assert flaky.calls == 3
    assert slept == [0.5, 1.0]


@pytest.mark.parametrize("code", [429, 500, 502, 503])
def test_http_429_and_5xx_are_retried(code: int, monkeypatch) -> None:
    slept: list[float] = []
    attempts = {"n": 0}
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _http_error(code)
        captured["ok"] = True
        body = {"choices": [{"message": {"content": "yes"}}]}
        return _FakeResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    judge = OpenAICompatJudge("http://h", "m", sleep_fn=slept.append)
    assert judge.complete("P", max_tokens=8) == "yes"
    assert attempts["n"] == 2
    assert slept == [0.5]
    assert captured["ok"]


def test_http_400_is_not_retried(monkeypatch) -> None:
    slept: list[float] = []
    attempts = {"n": 0}

    def fake_urlopen(req, timeout=None):
        attempts["n"] += 1
        raise _http_error(400)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    judge = OpenAICompatJudge("http://h", "m", sleep_fn=slept.append)
    with pytest.raises(urllib.error.HTTPError):
        judge.complete("P", max_tokens=8)
    assert attempts["n"] == 1  # a malformed request will never succeed
    assert slept == []


def test_anthropic_judge_retries_transport_errors() -> None:
    slept: list[float] = []

    class _FlakyMessages:
        def __init__(self) -> None:
            self.n = 0

        def create(self, **kwargs: object):
            self.n += 1
            if self.n == 1:
                raise ConnectionError("reset")

            class _Block:
                text = "yes"

            class _Resp:
                content = [_Block()]

            return _Resp()

    class _Client:
        def __init__(self) -> None:
            self.messages = _FlakyMessages()

    client = _Client()
    judge = AnthropicJudge(model="m", client=client, sleep_fn=slept.append)
    assert judge.complete("P", max_tokens=8) == "yes"
    assert client.messages.n == 2
    assert slept == [0.5]


# -- AnthropicJudge ----------------------------------------------------------


class _StubMessages:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def create(self, **kwargs: object):
        self.kwargs = kwargs

        class _Block:
            text = "yes"

        class _Resp:
            content = [_Block()]

        return _Resp()


class _StubClient:
    def __init__(self) -> None:
        self.messages = _StubMessages()


def test_anthropic_judge_call_shape() -> None:
    client = _StubClient()
    judge = AnthropicJudge(model="claude-test", client=client)
    assert judge.complete("PROMPT", max_tokens=123) == "yes"
    kw = client.messages.kwargs
    assert kw["model"] == "claude-test"
    assert kw["max_tokens"] == 123
    assert kw["temperature"] == 0
    assert kw["messages"] == [{"role": "user", "content": "PROMPT"}]


# -- OpenAICompatJudge -------------------------------------------------------


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _patch_urlopen(monkeypatch, captured: dict, content: str = "no"):
    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = dict(req.header_items())
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        body = {"choices": [{"message": {"content": content}}]}
        return _FakeResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_openai_compat_judge_payload_and_parsing(monkeypatch) -> None:
    captured: dict = {}
    _patch_urlopen(monkeypatch, captured, content="yes")
    judge = OpenAICompatJudge("https://example.modal.run/", "qwen3-27b", timeout=7.5)
    assert judge.complete("PROMPT", max_tokens=64) == "yes"
    assert captured["url"] == "https://example.modal.run/v1/chat/completions"
    assert captured["method"] == "POST"
    assert captured["timeout"] == 7.5
    assert captured["payload"] == {
        "model": "qwen3-27b",
        "messages": [{"role": "user", "content": "PROMPT"}],
        "temperature": 0,
        "max_tokens": 64,
    }
    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers["content-type"] == "application/json"
    assert "authorization" not in headers


def test_openai_compat_judge_adds_auth_header_only_with_a_key(monkeypatch) -> None:
    captured: dict = {}
    _patch_urlopen(monkeypatch, captured)
    OpenAICompatJudge("http://h:8000", "m", api_key="sk-x").complete("P", max_tokens=8)
    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers["authorization"] == "Bearer sk-x"


def test_openai_compat_judge_raises_on_malformed_reply(monkeypatch) -> None:
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(json.dumps({"choices": []}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert captured == {}
    with pytest.raises(IndexError):
        OpenAICompatJudge("http://h", "m").complete("P", max_tokens=8)


def test_openai_compat_judge_extra_body_and_think_stripping(monkeypatch) -> None:
    captured: dict = {}
    _patch_urlopen(monkeypatch, captured, content="<think>reasoning</think>\nyes")
    judge = OpenAICompatJudge(
        "http://h:8000", "qwen3.8-27b",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    assert judge.complete("P", max_tokens=8) == "yes"
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["payload"]["model"] == "qwen3.8-27b"
