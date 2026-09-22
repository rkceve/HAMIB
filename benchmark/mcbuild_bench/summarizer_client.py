"""Local summarizer client (Qwen3.5-4B via vLLM) — DESIGN.md §4 / §6 / §8, DECISIONS D2.

OpenAI-compatible ``POST {base_url}/chat/completions``. Node text rule (D2):
<=120 characters; one retry with the explicit length instruction, then
``SummarizerNodeTextUnusable`` (a ``SummarizerStop``), which ``JevNodeFn`` maps to
the spec manager's node_fallback (H28). Transport / HTTP / body failures raise
the plain ``SummarizerStop`` and stop the run.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeGuard

from benchmark.mcbuild_bench.errors import SummarizerNodeTextUnusable, SummarizerStop

# DESIGN.md §8, verbatim.
INSTRUCTION = (
    "Rewrite the following excerpt as one self-contained statement of at most 120 "
    "characters, in the excerpt's language, keeping concrete values. Output the "
    "statement only."
)
RETRY_SUFFIX = "\n\nYour previous answer was too long. At most 120 characters."

MAX_NODE_CHARS = 120
MAX_TOKENS = 96
# Item I (2026-09-18): a node text must be ONE line, carry no CD marker syntax
# (it would be re-serialized inside the <CONTEXT> block and confuse the marker
# scan) and come from a completion that finished on its own (finish_reason
# "stop", not "length" / missing).  Any violation is treated like over-length:
# one retry, then SummarizerStop.
MARKER_SYNTAX = ("[SN]", "[PN", "[RN]", "<CONTEXT>", "</CONTEXT>")
FINISH_STOP = "stop"

Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, bytes]]


def node_text_problem(text: str, finish_reason: object) -> str | None:
    """None when ``text`` is an acceptable node text, else a short reason."""
    if finish_reason != FINISH_STOP:
        return f"finish_reason={finish_reason!r} (expected {FINISH_STOP!r})"
    if not text:
        return "empty"
    if len(text) > MAX_NODE_CHARS:
        return f"length {len(text)} > {MAX_NODE_CHARS}"
    if "\n" in text:
        return "multi-line"
    for marker in MARKER_SYNTAX:
        if marker in text:
            return f"marker syntax {marker!r}"
    return None


def urllib_transport(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> tuple[int, bytes]:
    """Default transport: one POST via urllib; non-2xx is returned, not raised."""
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            payload: bytes = response.read()
            return status, payload
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


class SummarizerClient:
    """One chat-completions call per chunk with JSONL accounting (D5)."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001/v1",
        model: str = "summarizer",
        *,
        accounting_path: Path | str,
        timeout_s: float = 120,
        transport: Transport | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.accounting_path = Path(accounting_path)
        self.timeout_s = float(timeout_s)
        self._transport: Transport = urllib_transport if transport is None else transport
        # Item 14: one lock per client instance guards the JSONL append.
        self._accounting_lock = threading.Lock()

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    # ------------------------------------------------------------------ public
    def summarize(self, excerpt: str) -> str:
        """Return a non-empty node text of <= 120 characters, or raise ``SummarizerStop``."""
        if not isinstance(excerpt, str) or not excerpt.strip():
            raise ValueError("excerpt must be a non-blank string")
        replies: list[str] = []
        problems: list[str] = []
        for retried in (False, True):
            content = INSTRUCTION + "\n\n" + excerpt
            if retried:
                content += RETRY_SUFFIX
            text, finish_reason = self._call(content, retried=retried)
            replies.append(text)
            problem = node_text_problem(text, finish_reason)
            if problem is None:
                return text
            problems.append(problem)
        raise SummarizerNodeTextUnusable(
            "summarizer produced no usable node text after one retry: "
            f"first={replies[0]!r} ({problems[0]}) second={replies[1]!r} ({problems[1]})"
        )

    # ----------------------------------------------------------------- private
    def _call(self, content: str, *, retried: bool) -> tuple[str, object]:
        """(stripped content, finish_reason) of one completion call."""
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": MAX_TOKENS,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        t0 = time.perf_counter()
        try:
            status, raw = self._transport(self.endpoint, headers, body, self.timeout_s)
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._record(None, None, latency_ms, retried, None)
            raise SummarizerStop(f"summarizer transport error: {exc!r}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if not 200 <= status < 300:
            self._record(None, None, latency_ms, retried, None)
            raise SummarizerStop(
                f"summarizer HTTP {status}: {raw[:500].decode('utf-8', errors='replace')}"
            )
        # Item 8: usage first, so a content failure still books its tokens.
        try:
            envelope = _parse_body(raw)
            prompt_tokens, completion_tokens = _parse_usage(envelope)
        except SummarizerStop:
            self._record(None, None, latency_ms, retried, None)
            raise
        try:
            text = _parse_content(envelope)
        except SummarizerStop:
            self._record(prompt_tokens, completion_tokens, latency_ms, retried, None)
            raise
        self._record(prompt_tokens, completion_tokens, latency_ms, retried, len(text))
        return text, _finish_reason(envelope)

    def _record(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        latency_ms: float,
        retried: bool,
        chars: int | None,
    ) -> None:
        line = {
            "ts": _utc_now_iso(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "retried": retried,
            "chars": chars,
        }
        text = json.dumps(line, ensure_ascii=False) + "\n"
        with self._accounting_lock:
            self.accounting_path.parent.mkdir(parents=True, exist_ok=True)
            with self.accounting_path.open("a", encoding="utf-8") as fh:
                fh.write(text)


def _parse_body(raw: bytes) -> dict[str, Any]:
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SummarizerStop(f"summarizer 2xx body is not JSON: {raw[:500]!r}") from exc
    if not isinstance(envelope, dict):
        raise SummarizerStop("summarizer body is not a JSON object")
    return envelope


def _parse_usage(envelope: dict[str, Any]) -> tuple[int, int]:
    """``usage.prompt_tokens`` / ``usage.completion_tokens`` as ints (D5)."""
    usage = envelope.get("usage")
    if not isinstance(usage, dict):
        raise SummarizerStop("summarizer response lacks 'usage' object")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not _is_int(prompt_tokens) or not _is_int(completion_tokens):
        raise SummarizerStop("summarizer 'usage' lacks integer prompt_tokens/completion_tokens")
    return prompt_tokens, completion_tokens


def _parse_content(envelope: dict[str, Any]) -> str:
    """``choices[0].message.content`` (stripped)."""
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise SummarizerStop("summarizer response lacks 'choices[0]'")
    message: Any = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise SummarizerStop("summarizer response lacks 'choices[0].message.content' string")
    content: str = message["content"]
    return content.strip()


def _finish_reason(envelope: dict[str, Any]) -> object:
    """``choices[0].finish_reason`` (None when absent; validated by the caller)."""
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    return choices[0].get("finish_reason")


def _parse_completion(raw: bytes) -> tuple[str, int, int]:
    """Extract ``choices[0].message.content`` (stripped) and ``usage`` token counts."""
    envelope = _parse_body(raw)
    prompt_tokens, completion_tokens = _parse_usage(envelope)
    return _parse_content(envelope), prompt_tokens, completion_tokens
