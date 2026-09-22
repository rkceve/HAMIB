"""Fake Jev / summarizer objects for the mcbuild tests.  No network, no model.

``FakeJev.ask`` returns replies shaped EXACTLY like the DESIGN.md 3 verbatim
JSON (noul / choice / score answer objects, ``usage`` block), driven by simple
text rules so the same fake works for keep, the three axes, same/belongs and
the sun choice.
"""

from __future__ import annotations

from typing import Any, Callable

# Tags a test puts into a chunk to steer the fake's level decision.
TAG_SUN = "[[SUN]]"
TAG_PLANET = "[[PLANET]]"
TAG_DROP = "[[DROP]]"


def score_answer(levels: dict[str, float]) -> dict[str, Any]:
    """A `score` answer over 5 levels with the given probabilities."""
    probs = {str(k): 0.0 for k in range(5)}
    probs.update({str(k): float(v) for k, v in levels.items()})
    weighted = sum(int(k) * p for k, p in probs.items())
    return {
        "type": "score",
        "score": weighted,
        "legend": {str(k): f"level {k}" for k in range(5)},
        "probabilities": probs,
        "confidence": 0.8,
    }


def noul_answer(value: float) -> dict[str, Any]:
    return {"type": "noul", "noul": float(value)}


def choice_answer(choice: str, options: list[str]) -> dict[str, Any]:
    probs = {o: (0.9 if o == choice else 0.1 / max(1, len(options) - 1)) for o in options}
    return {"type": "choice", "choice": choice, "probabilities": probs, "confidence": 0.82}


def _topic_word(text: str) -> str:
    """First capitalised word of a chunk: the fake's notion of a topic key."""
    for word in text.replace("\n", " ").split():
        w = word.strip("[]:.,")
        if w[:1].isupper() and w.isalpha():
            return w
    return ""


class FakeJev:
    """Rule-driven Jev stand-in.

    keep     : False when TAG_DROP or "[tool_result]" is in the state
    axes     : TAG_SUN -> comprehensiveness top, TAG_PLANET -> independence top,
               else detail top (level 4 on the winning axis, level 0 elsewhere)
    same     : noul (single candidate): True iff A == B (verbatim);
               choice (H23 batch, keys mN): first mN whose text's topic word
               occurs in the state, else none (the same rule as pN)
    belongs  : True iff the topic word of B occurs in A
    sun      : first sN whose text's topic word occurs in the state, else new_topic
    planet   : first pN whose text's topic word occurs in the state, else none
    ``override`` lets a test replace the whole reply for one question id.
    """

    def __init__(self, override: dict[str, Callable[[str, dict], Any]] | None = None) -> None:
        self.requests: list[tuple[str, dict]] = []
        self.override = dict(override or {})

    def _axes(self, state: str) -> dict[str, Any]:
        top = "detail"
        if TAG_SUN in state:
            top = "comprehensiveness"
        elif TAG_PLANET in state:
            top = "independence"
        out = {}
        for axis in ("comprehensiveness", "independence", "detail"):
            out[axis] = score_answer({"4": 0.7, "0": 0.3} if axis == top else {"0": 0.7, "1": 0.3})
        return out

    def ask(self, state: str, questions: dict[str, dict]) -> dict:
        self.requests.append((state, questions))
        answers: dict[str, Any] = {}
        axes = None
        for qid, q in questions.items():
            if qid in self.override:
                answers[qid] = self.override[qid](state, q)
                continue
            if qid == "keep":
                answers[qid] = noul_answer(0.1 if (TAG_DROP in state or "[tool_result]" in state) else 0.9)
            elif qid in ("comprehensiveness", "independence", "detail"):
                axes = axes if axes is not None else self._axes(state)
                answers[qid] = axes[qid]
            elif qid == "same" and q.get("type") == "noul":
                a, b = state.split("\nB: ", 1)
                answers[qid] = noul_answer(0.95 if a[len("A: "):] == b else 0.05)
            elif qid == "belongs":
                a, b = state.split("\nB: ", 1)
                key = _topic_word(b)
                answers[qid] = noul_answer(0.9 if key and key in a else 0.1)
            elif qid in ("sun", "planet", "same"):
                # H22 (a): sun batches end with new_topic, planet batches with none;
                # H23: `same` Choice batches (keys mN) end with none too.
                none_key = "new_topic" if qid == "sun" else "none"
                options = list(q["criteria"].keys())
                pick = none_key
                for key, text in q["criteria"].items():
                    if key == none_key:
                        continue
                    word = _topic_word(text)
                    if word and word in state:
                        pick = key
                        break
                answers[qid] = choice_answer(pick, options)
            else:
                raise AssertionError(f"FakeJev: unexpected question id {qid!r}")
        return {
            "answers": answers,
            "usage": {"input_tokens": 100 + len(state) // 4, "output_tokens": 12},
            "latency_ms": 42.0,
            "http_status": 200,
            "retries": 0,
        }


class FakeSummarizer:
    """Returns the chunk with tags removed, cut to 120 chars (never raises)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def summarize(self, excerpt: str) -> str:
        self.calls.append(excerpt)
        text = excerpt
        for tag in (TAG_SUN, TAG_PLANET, TAG_DROP):
            text = text.replace(tag, "")
        return " ".join(text.split())[:120]
