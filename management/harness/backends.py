"""Judge backends.  Design: HARNESS_DESIGN.md Stream B, B1.

Every backend is a thin transport: prompt in, raw text out.  All parsing,
retrying (of the ANSWER FORMAT), caching and counting lives in ``judge.py`` /
``manager.py``; the transport-level retry of H4 lives here because only the
transport knows what a 429 or a socket error is.

  FakeJudge          deterministic rule-based, for tests and the driver smoke.
  AnthropicJudge     anthropic SDK ``messages.create``, temperature 0.
  OpenAICompatJudge  urllib POST to an OpenAI-compatible ``/v1/chat/completions``
                     (e.g. vLLM on Modal).  No new dependency.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping, Sequence

from management.harness.prompts import FENCE_CLOSE, FENCE_OPEN

# -- transport retry (H4) ----------------------------------------------------

# Up to 3 attempts; the delay AFTER attempt i is BACKOFF_DELAYS[i].  With the
# default 3 attempts the sleeps are 0.5s and 1.0s (the third entry is used when
# a caller raises `attempts`).
MAX_ATTEMPTS = 3
BACKOFF_DELAYS: tuple[float, ...] = (0.5, 1.0, 2.0)

SleepFn = Callable[[float], None]


def is_retryable_http(exc: BaseException) -> bool:
    """HTTP 429 (rate limit) and 5xx (server) are worth another attempt."""
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
    """Run ``fn`` with exponential backoff.  Re-raises the last exception."""
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 -- classified by `retryable`
            last = exc
            if not retryable(exc) or attempt == attempts - 1:
                raise
            delay = BACKOFF_DELAYS[min(attempt, len(BACKOFF_DELAYS) - 1)]
            sleep_fn(delay)
    raise AssertionError("unreachable") from last


# -- FakeJudge ---------------------------------------------------------------


class FakeJudge:
    """Deterministic judge driven by regex rules or a callable policy.

    ``rules``  : mapping/sequence of (regex, answer).  First match wins; the
                 regexes are tried in insertion order.
    ``policy`` : callable(prompt) -> str, checked BEFORE the rules; returning
                 None falls through to the rules.
    ``default``: answer when nothing matches.

    Every prompt is recorded in ``self.prompts`` for assertions.
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
    """The ``--judge fake`` policy of B6 / H10.

    Extraction returns the whole chunk as ONE statement (recovered verbatim from
    the prompt's fenced span); the FAITHFULNESS question answers "yes" (the
    statement IS the chunk, so it is trivially supported) and every other yes/no
    question answers "no".  All-no on the three axes means satellite, and the
    satellites are then promoted by 0061, so the smoke run produces a NON-EMPTY
    correlation diagram and exercises normalize + serialization end to end.
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
        # A5: how many replies carried only a reasoning trace and no content.
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
        # F9: a reply whose content is null (a reasoning-only turn, a refusal, a
        # length stop) or a structured content LIST must come back as the empty
        # string. ``str(None)`` produced the literal "None", which is a
        # non-empty, unparseable answer: the caller counted it as a PARSE
        # failure with a plausible-looking body instead of as an empty reply.
        if not isinstance(content, str):
            # A5: vLLM 0.28 renamed ``reasoning_content`` to ``reasoning``. The
            # answer is deliberately NOT read out of it -- a reasoning trace is
            # not an answer, and mining one would fabricate judgements. Only the
            # fact is recorded, so a 100%-reasoning-only run is visible.
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
    """Remove ``<think>...</think>`` reasoning blocks (Qwen3.x family).

    H5: some servers emit the closing tag WITHOUT the opening one (the opener is
    consumed by the chat template), so everything up to and including the LAST
    ``</think>`` is dropped.  An unterminated ``<think>`` means the answer itself
    was never emitted -- the whole reply is discarded (returns "") rather than
    guessing an answer out of the reasoning trace.
    """
    cleaned = _THINK_RE.sub("", text)
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[1]
    if "<think>" in cleaned:
        return ""
    return cleaned.strip()
