"""Judge backends: send a prompt, return the raw reply text.

Parsing, answer-format retries, caching and call counting live in ``judge.py``
and the managers, so every backend behaves the same.  Only network retries
(rate limits, server errors, dropped connections) live here, because only the
transport can tell those apart.

  FakeJudge          deterministic rule-based judge for tests and smoke runs.
  AnthropicJudge     anthropic SDK ``messages.create``, temperature 0.
  OpenAICompatJudge  urllib POST to an OpenAI-compatible ``/v1/chat/completions``
                     (e.g. vLLM on Modal); no extra dependency.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping, Sequence

from management.harness.prompts import FENCE_CLOSE, FENCE_OPEN

# -- network retry -----------------------------------------------------------

# Up to 3 attempts.  The sleep after failed attempt i is BACKOFF_DELAYS[i]
# (0.5 s, then 1.0 s); the last entry is reused when a caller asks for more.
MAX_ATTEMPTS = 3
BACKOFF_DELAYS: tuple[float, ...] = (0.5, 1.0, 2.0)

SleepFn = Callable[[float], None]


def is_retryable_http(exc: BaseException) -> bool:
    """Retry HTTP 429 (rate limit), 5xx (server error) and any non-HTTP error."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or 500 <= exc.code < 600
    return True


def call_with_retries(
    fn: Callable[[], str],
    *,
    attempts: int = MAX_ATTEMPTS,
    sleep_fn: SleepFn = time.sleep,
    retryable: Callable[[BaseException], bool] = lambda _e: True,
) -> str:
    """Call ``fn``, retrying with backoff while ``retryable`` allows; re-raises
    the last error."""
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 -- classified by `retryable`
            last = exc
            if not retryable(exc) or attempt == attempts - 1:
                raise
            sleep_fn(BACKOFF_DELAYS[min(attempt, len(BACKOFF_DELAYS) - 1)])
    raise AssertionError("unreachable") from last


# -- FakeJudge ---------------------------------------------------------------


class FakeJudge:
    """Deterministic judge for tests.

    ``policy(prompt)`` is tried first; if it returns None, the first regex in
    ``rules`` (a mapping or a list of ``(regex, answer)`` pairs) that matches
    the prompt gives the answer; otherwise ``default``.  Every prompt is kept
    in ``self.prompts``.
    """

    def __init__(
        self,
        rules: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
        *,
        policy: Callable[[str], str | None] | None = None,
        default: str = "no",
    ) -> None:
        if rules is None:
            pairs: list[tuple[str, str]] = []
        elif isinstance(rules, Mapping):
            pairs = list(rules.items())
        else:
            pairs = list(rules)
        self._rules: list[tuple[re.Pattern[str], str]] = [
            (re.compile(pat, re.DOTALL), ans) for pat, ans in pairs
        ]
        self._policy = policy
        self._default = default
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        self.prompts.append(prompt)
        if self._policy is not None:
            answer = self._policy(prompt)
            if answer is not None:
                return answer
        for pattern, answer in self._rules:
            if pattern.search(prompt):
                return answer
        return self._default


_FENCE_RE = re.compile(
    re.escape(FENCE_OPEN) + r"\n(.*?)\n" + re.escape(FENCE_CLOSE), re.DOTALL
)


def first_fenced_span(prompt: str) -> str:
    """Return the first ``<<< ... >>>`` span of a harness prompt (or "")."""
    m = _FENCE_RE.search(prompt)
    return m.group(1) if m else ""


def make_driver_fake_judge(max_chars: int = 120) -> FakeJudge:
    """Fake judge for HarnessManager ``--judge fake`` runs.

    Extraction returns the whole chunk (read back from the prompt) as one
    statement, the "is it supported?" question answers yes (the statement is
    the chunk itself), and every other question answers no.  All-no axes make
    satellites, which the merge promotes, so a smoke run still builds a
    non-empty diagram and exercises normalize and serialization end to end.
    """

    def policy(prompt: str) -> str | None:
        if "JSON array" in prompt:
            text = first_fenced_span(prompt).strip()[:max_chars]
            return json.dumps([text] if text else [], ensure_ascii=False)
        if "supported by the source" in prompt:
            return "yes"
        return "no"

    return FakeJudge(policy=policy)


# -- AnthropicJudge ----------------------------------------------------------


class AnthropicJudge:
    """Anthropic SDK backend.  Pass ``client`` to inject a stub in tests."""

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        client: Any | None = None,
        *,
        attempts: int = MAX_ATTEMPTS,
        sleep_fn: SleepFn = time.sleep,
    ) -> None:
        self._client: Any = client
        if self._client is None:
            import anthropic  # imported lazily: no key needed to import the module

            self._client = anthropic.Anthropic()
        self._model = model
        self._attempts = attempts
        self._sleep_fn = sleep_fn

    def _once(self, prompt: str, max_tokens: int) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return str(resp.content[0].text)

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        return call_with_retries(
            lambda: self._once(prompt, max_tokens),
            attempts=self._attempts,
            sleep_fn=self._sleep_fn,
        )


# -- OpenAICompatJudge -------------------------------------------------------


class OpenAICompatJudge:
    """POST ``{base_url}/v1/chat/completions`` with urllib only."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        extra_body: Mapping[str, Any] | None = None,
        attempts: int = MAX_ATTEMPTS,
        sleep_fn: SleepFn = time.sleep,
    ) -> None:
        self._url = base_url.rstrip("/") + "/v1/chat/completions"
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        # Server-specific request fields, e.g. vLLM + Qwen3.x:
        # {"chat_template_kwargs": {"enable_thinking": False}}.
        self._extra_body: dict[str, Any] = dict(extra_body) if extra_body else {}
        self._attempts = attempts
        self._sleep_fn = sleep_fn
        # Replies that carried only a reasoning trace and no answer text.
        self.reasoning_only_replies = 0

    def _once(self, prompt: str, max_tokens: int) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        payload.update(self._extra_body)
        req = urllib.request.Request(
            self._url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self._api_key:
            req.add_header("Authorization", "Bearer " + self._api_key)
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        message = data["choices"][0]["message"]
        content = message.get("content")
        # A null content (reasoning-only reply, refusal, length stop) or a
        # structured content list becomes "" -- not str(None) == "None", which
        # the caller would take for a real but unparsable answer.
        if not isinstance(content, str):
            # Newer vLLM calls the field ``reasoning`` (older:
            # ``reasoning_content``).  The answer is never mined out of the
            # reasoning, which would invent judgements; the reply is only
            # counted, so a run where every reply is reasoning-only stands out.
            if message.get("reasoning") or message.get("reasoning_content"):
                self.reasoning_only_replies += 1
            return ""
        return strip_think_blocks(content)

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        return call_with_retries(
            lambda: self._once(prompt, max_tokens),
            attempts=self._attempts,
            sleep_fn=self._sleep_fn,
            retryable=is_retryable_http,
        )


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks (Qwen3 family).

    Some servers drop the opening tag (the chat template consumes it), so
    everything up to the last ``</think>`` is removed.  An unclosed
    ``<think>`` means the model never reached its answer, so "" is returned
    rather than guessing one from the reasoning.
    """
    cleaned = _THINK_RE.sub("", text)
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[1]
    if "<think>" in cleaned:
        return ""
    return cleaned.strip()
