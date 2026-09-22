"""Jev (TypeSafe) client for mcbuild-bench — DESIGN.md §3 / §6, DECISIONS D1, D3, D5.

Only the request/response fields quoted in DESIGN.md §3 are used. Every failure
raises ``JevStop``; there are no default answers and no silent fallbacks.

HTTP goes through an injectable ``transport(url, headers, body_bytes, timeout)
-> (status, body_bytes)`` so unit tests never touch the network. The default
transport uses only ``urllib.request``.
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Collection
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeGuard

from benchmark.mcbuild_bench.errors import JevStop

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"

# D1: "429/529 -> exponential backoff, 3 attempts, then STOP". Sleeps between attempts.
BACKOFF_SECONDS: tuple[float, ...] = (1.0, 4.0, 16.0)
RETRY_STATUSES: frozenset[int] = frozenset({429, 529})

# D5: "Cost = input_tokens x $0.042/M (published price, recorded as published_price_per_mtok)".
published_price_per_mtok: float = 0.042

Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, bytes]]
SleepFn = Callable[[float], None]


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


class JevClient:
    """Stateless Jev caller with per-attempt JSONL accounting (D5)."""

    def __init__(
        self,
        api_key: str,
        model: str = "jev-latest",
        timeout_s: float = 60,
        max_attempts: int = 3,
        *,
        accounting_path: Path | str,
        transport: Transport | None = None,
        sleep_fn: SleepFn = time.sleep,
    ) -> None:
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("api_key must be a non-empty string")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not 1 <= max_attempts <= len(BACKOFF_SECONDS) + 1:
            raise ValueError(
                f"max_attempts must be in 1..{len(BACKOFF_SECONDS) + 1} "
                f"(backoff schedule {BACKOFF_SECONDS}), got {max_attempts}"
            )
        self._api_key = api_key
        self.model = model
        self.timeout_s = float(timeout_s)
        self.max_attempts = max_attempts
        self.accounting_path = Path(accounting_path)
        self._transport: Transport = urllib_transport if transport is None else transport
        self._sleep = sleep_fn
        # Item 14: one lock per client instance guards the JSONL append.
        self._accounting_lock = threading.Lock()

    # ------------------------------------------------------------------ public
    def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """POST one Jev request; return ``JevResult`` (DESIGN.md §6) or raise ``JevStop``."""
        if not isinstance(state, str) or not state:
            raise ValueError("state must be a non-empty string")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a non-empty dict")
        question_types: dict[str, str] = {}
        for qid, question in questions.items():
            if not isinstance(qid, str) or not isinstance(question, dict):
                raise ValueError("questions must map str id -> dict question")
            qtype = question.get("type")
            if not isinstance(qtype, str):
                raise ValueError(f"question {qid!r} lacks a string 'type'")
            question_types[qid] = qtype
        question_ids = sorted(questions)

        body = json.dumps(
            {"state": state, "model": self.model, "questions": questions}
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(self.max_attempts):
            t0 = time.perf_counter()
            try:
                status, raw = self._transport(JEV_ENDPOINT, headers, body, self.timeout_s)
            except Exception as exc:  # network / timeout: no retry, no fallback
                latency_ms = (time.perf_counter() - t0) * 1000.0
                self._record(
                    question_ids, question_types, len(state), None, None,
                    latency_ms, None, attempt, None,
                )
                raise JevStop(f"Jev transport error on attempt {attempt}: {exc!r}") from exc
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if status in RETRY_STATUSES:
                self._record(
                    question_ids, question_types, len(state), None, None,
                    latency_ms, status, attempt, None,
                )
                if attempt + 1 < self.max_attempts:
                    self._sleep(BACKOFF_SECONDS[attempt])
                    continue
                raise JevStop(
                    f"Jev HTTP {status} after {self.max_attempts} attempts: "
                    f"{raw[:500].decode('utf-8', errors='replace')}"
                )

            if not 200 <= status < 300:
                self._record(
                    question_ids, question_types, len(state), None, None,
                    latency_ms, status, attempt, None,
                )
                raise JevStop(
                    f"Jev HTTP {status}: {raw[:500].decode('utf-8', errors='replace')}"
                )

            # H10 / D5: usage FIRST (accounting completeness), then the answers.
            try:
                envelope = _parse_body(raw)
                input_tokens, output_tokens = _parse_usage(envelope)
            except JevStop:
                self._record(
                    question_ids, question_types, len(state), None, None,
                    latency_ms, status, attempt, None,
                )
                raise
            try:
                answers = _parse_answers(envelope, question_ids)
            except JevStop:
                self._record(
                    question_ids, question_types, len(state), input_tokens, output_tokens,
                    latency_ms, status, attempt, None,
                )
                raise
            self._record(
                question_ids, question_types, len(state), input_tokens, output_tokens,
                latency_ms, status, attempt, answers,
            )
            return {
                "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                "latency_ms": latency_ms,
                "http_status": status,
                "retries": attempt,
            }
        raise AssertionError("unreachable: attempt loop exhausted without raising")

    # ----------------------------------------------------------------- private
    def _record(
        self,
        question_ids: list[str],
        question_types: dict[str, str],
        state_chars: int,
        input_tokens: int | None,
        output_tokens: int | None,
        latency_ms: float,
        http_status: int | None,
        retry_index: int,
        answers: dict[str, Any] | None,
    ) -> None:
        line = {
            "ts": _utc_now_iso(),
            "question_ids": question_ids,
            "question_types": question_types,
            "state_chars": state_chars,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "http_status": http_status,
            "retry_index": retry_index,
            "answers": answers,
        }
        text = json.dumps(line, ensure_ascii=False) + "\n"
        with self._accounting_lock:
            self.accounting_path.parent.mkdir(parents=True, exist_ok=True)
            with self.accounting_path.open("a", encoding="utf-8") as fh:
                fh.write(text)


def _parse_body(raw: bytes) -> dict[str, Any]:
    """Decode the 2xx body; anything but a JSON object is ``JevStop``."""
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JevStop(f"Jev 2xx body is not JSON: {raw[:500]!r}") from exc
    if not isinstance(envelope, dict):
        raise JevStop("Jev 2xx body is not a JSON object")
    return envelope


def _parse_usage(envelope: dict[str, Any]) -> tuple[int, int]:
    """``usage.{input,output}_tokens`` as ints (H10 / D5).

    An int, or a float with ``is_integer()``, is accepted and converted;
    a missing ``usage`` object or any other value (string, non-integral float,
    bool, None) is an accounting hole and stops the run.
    """
    usage = envelope.get("usage")
    if not isinstance(usage, dict):
        raise JevStop("Jev 2xx response lacks a 'usage' object (H10: accounting is mandatory)")
    return (
        _token_count(usage.get("input_tokens"), "usage.input_tokens"),
        _token_count(usage.get("output_tokens"), "usage.output_tokens"),
    )


def _parse_answers(envelope: dict[str, Any], question_ids: list[str]) -> dict[str, Any]:
    """Validate the DESIGN 3 ``answers`` object strictly (D1)."""
    answers = envelope.get("answers")
    if not isinstance(answers, dict):
        raise JevStop("Jev response lacks 'answers' object")
    missing = [qid for qid in question_ids if qid not in answers]
    if missing:
        raise JevStop(f"Jev response missing answers for question ids {missing}")
    for qid in question_ids:
        if not isinstance(answers[qid], dict):
            raise JevStop(f"Jev answer for {qid!r} is not an object")
    return answers


def _token_count(value: object, name: str) -> int:
    """Accounting token count: int, or an integral float -> int; else ``JevStop`` (H10)."""
    if _is_int(value):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise JevStop(f"Jev {name} is not an integer token count: {value!r} (H10)")


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ------------------------------------------------------------ pure parsing helpers (D3)
def _probability(value: object, what: str) -> float:
    """A finite number in [0, 1] (Astra round 3 item 5), else ``JevStop``."""
    if not _is_number(value):
        raise JevStop(f"{what} is not a number: {value!r}")
    prob = float(value)
    if not math.isfinite(prob):
        raise JevStop(f"{what} is not finite: {value!r}")
    if not 0.0 <= prob <= 1.0:
        raise JevStop(f"{what} {prob} outside [0, 1]")
    return prob


def choice_of(answer: dict[str, Any], options: Collection[str] | None = None) -> str:
    """Choice decision = ``choice`` (D3). Requires ``type == "choice"``.

    When ``options`` (the offered criteria keys) is given, ``choice`` must be
    one of them and every key of the optional ``probabilities`` object must be
    one of them; the probabilities, if present, must be finite numbers in
    [0, 1] with at least one > 0 (Astra round 3 item 5).
    """
    if answer.get("type") != "choice":
        raise JevStop(f"expected a choice answer, got type={answer.get('type')!r}")
    choice = answer.get("choice")
    if not isinstance(choice, str) or not choice:
        raise JevStop("choice answer lacks a non-empty 'choice' string")
    if options is not None and choice not in options:
        raise JevStop(f"choice {choice!r} is not an offered option")
    probabilities = answer.get("probabilities")
    if probabilities is not None:
        if not isinstance(probabilities, dict) or not probabilities:
            raise JevStop("choice 'probabilities' is not a non-empty object")
        any_positive = False
        for key, value in probabilities.items():
            if not isinstance(key, str):
                raise JevStop(f"choice probability key {key!r} is not a string")
            if options is not None and key not in options:
                raise JevStop(f"choice probability key {key!r} is not an offered option")
            if _probability(value, f"choice probability for {key!r}") > 0.0:
                any_positive = True
        if not any_positive:
            raise JevStop("choice probabilities are all zero")
    return choice


def argmax_level(answer: dict[str, Any], n_levels: int | None = None) -> int:
    """Level = argmax over the keys PRESENT in ``probabilities`` (a missing
    level is probability 0); keys are level-index strings; tie -> lower index.

    At least one key is required and every probability must be a finite
    number in [0, 1] with at least one > 0.  When ``n_levels`` is given EVERY
    key must be an index ``< n_levels`` (the caller's level table), else
    ``JevStop`` (Astra round 3 item 5).
    """
    if answer.get("type") != "score":
        raise JevStop(f"expected a score answer, got type={answer.get('type')!r}")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        raise JevStop("score answer lacks a non-empty 'probabilities' object")
    parsed: list[tuple[int, float]] = []
    for key, value in probabilities.items():
        if not isinstance(key, str) or not key.isdigit():
            raise JevStop(f"score probability key {key!r} is not a level-index string")
        index = int(key)
        if n_levels is not None and not index < n_levels:
            raise JevStop(
                f"score probability key {key!r} is outside the {n_levels}-level table"
            )
        parsed.append((index, _probability(value, f"score probability for level {key!r}")))
    if not any(prob > 0.0 for _, prob in parsed):
        raise JevStop("score probabilities are all zero")
    parsed.sort(key=lambda item: item[0])
    best_index, best_prob = parsed[0]
    for index, prob in parsed[1:]:
        if prob > best_prob:  # strictly greater: ties keep the lower index
            best_index, best_prob = index, prob
    return best_index


def noul_of(answer: dict[str, Any]) -> float:
    """Raw noul probability: a finite number in [0, 1]. Requires ``type == "noul"``."""
    if answer.get("type") != "noul":
        raise JevStop(f"expected a noul answer, got type={answer.get('type')!r}")
    noul = answer.get("noul")
    if not _is_number(noul):
        raise JevStop("noul answer lacks a numeric 'noul'")
    return _probability(noul, "noul value")


def read_price_per_mtok() -> float:
    """Published Jev price, USD per million input tokens (DECISIONS D5; recorded, not asserted)."""
    return published_price_per_mtok


def cost_usd(input_tokens: int) -> float:
    """Jev cost per D5: ``input_tokens x published_price_per_mtok / 1e6``."""
    if not _is_int(input_tokens) or input_tokens < 0:
        raise ValueError("input_tokens must be a non-negative int")
    return input_tokens * read_price_per_mtok() / 1_000_000
