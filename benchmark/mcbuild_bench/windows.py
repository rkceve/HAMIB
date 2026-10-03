"""Compose the reader's context window for one (arm, W) cell.

This module sits between the manager phase and the reader: build_cd produces
the correlation diagram (CD), compaction_c produces baseline C's summary, and
``build_window`` turns those plus the session's round trips into the context
that run_arms hands to the reader.  Questions are asked once, after the whole
session, so one window is built per cell and shared by all of its questions.

The arms:

- ``A``         the whole transcript, no budget (the full-context reference);
- ``B``         the most recent round trips only;
- ``C``         a compaction summary plus the most recent round trips;
- ``proposed``  the CD block plus the most recent round trips.

Composition::

    [CD block (proposed only)] + [the most recent WHOLE round trips that fit,
    in chronological order]; the reader template then adds the question.

``W`` bounds the token count of the FINAL reader prompt, the exact string the
reader tokenizes, not just the context.  That string is built by
``benchmark.bineval.run_reader.build_prompt`` around the context block returned
here (run_reader adds the instruction and the ``Question:`` / ``Answer:``
lines), so every fit test tokenizes ``build_prompt(context, question)`` with
``add_special_tokens=False``.  This module only produces the context text,
wrapped in ``<context>`` / ``</context>``; the rest of the prompt is the same
for every arm.

Round trips are rendered as::

    "### Human\\n" + human + "\\n" + "\\n".join(f"### {kind}\\n{text}" for each event)

Baseline C passes its compaction summary as the first round-trip-like block;
build it with ``summary_block(text)`` (rendered as ``### Summary\\n<text>``).
"""

from __future__ import annotations

from typing import Any, Callable

from benchmark.bineval.run_reader import build_prompt
from communication.cd_serializer import CDSerializer

ARMS = ("A", "B", "C", "proposed")

CONTEXT_OPEN = "<context>"
CONTEXT_CLOSE = "</context>"
BLOCK_SEPARATOR = "\n\n"
SUMMARY_KIND = "summary"

# Marker prefix of a planet line in a level-marker CD block ([PN{mass}] text).
PLANET_MARKER = "[PN"

# Safety valve for the eviction loop: each pass shrinks the CD budget by the
# measured overshoot, so a handful of passes is already generous.
MAX_EVICTION_PASSES = 20


def count_tokens(tokenizer: Any, text: str) -> int:
    """``len(tokenizer(text, add_special_tokens=False)["input_ids"])``."""
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def make_token_counter(tokenizer: Any) -> Callable[[str], int]:
    return lambda text: count_tokens(tokenizer, text)


def summary_block(text: str) -> dict:
    """The baseline-C compaction summary as a round-trip-like block (idx -1)."""
    return {"idx": -1, "kind": SUMMARY_KIND, "text": text}


def is_summary_block(rt: dict) -> bool:
    return rt.get("kind") == SUMMARY_KIND and "text" in rt


def render_round_trip(rt: dict) -> str:
    """Fixed text rendering of one round trip (or of a summary block)."""
    if is_summary_block(rt):
        return "### Summary\n" + rt["text"]
    events = rt.get("events", [])
    return (
        "### Human\n"
        + rt["human"]
        + "\n"
        + "\n".join(f"### {e['kind']}\n{e['text']}" for e in events)
    )


def assemble_context(cd_block: str | None, blocks: list[str]) -> str:
    """``<context>`` + [cd_block] + blocks (already chronological) + ``</context>``."""
    parts: list[str] = []
    if cd_block:
        parts.append(cd_block)
    parts.extend(blocks)
    body = BLOCK_SEPARATOR.join(parts)
    return f"{CONTEXT_OPEN}\n{body}\n{CONTEXT_CLOSE}"


def count_planet_lines(cd_block: str) -> int:
    return sum(1 for ln in cd_block.splitlines() if ln.strip().startswith(PLANET_MARKER))


def _full_prompt(cd_block: str | None, blocks: list[str], question: str,
                 tokenizer: Any = None) -> str:
    """The full reader prompt.  With ``tokenizer`` it is wrapped in the model's
    chat template with thinking turned off (without the template Qwen3.8
    answers inside a ``<think>`` block); W is then measured on that string."""
    return build_prompt(assemble_context(cd_block, blocks), question, tokenizer=tokenizer)


def build_window(
    arm: str,
    W: int | None,
    cd_block: str | None,
    round_trips: list[dict],
    tokenizer: Any,
    question: str,
    *,
    cd: Any = None,
    chat_template: bool = False,
) -> dict:
    """Compose one cell's window; see the module docstring for the rules.

    Returns ``{"prompt", "window_tokens", "n_recent_rts", "recent_idx", "cd_tokens",
    "evicted_planets", "prompt_context"}``:

    - ``prompt``          the FULL reader prompt (run_reader template, ``question``
                          included) whose token count ``W`` bounds;
    - ``prompt_context``  the context block only, the argument for
                          ``run_reader.run_reader(llm, context_block, ...)``, which
                          rebuilds exactly ``prompt`` around it;
    - ``window_tokens``   tokens of ``prompt``;
    - ``n_recent_rts``    number of real round trips kept (a baseline-C summary
                          block is not counted);
    - ``recent_idx``      their ``idx`` values in chronological order;
                          ``min(recent_idx)`` is the oldest round trip inside the
                          window, which run_arms uses to drop the questions whose
                          answer the window holds verbatim;
    - ``cd_tokens``       tokens of the CD block actually used (0 without CD);
    - ``evicted_planets`` ``[PN`` lines dropped to make the CD fit (0 if none).

    ``W=None`` is arm A: every round trip, no CD.  Arms B and C never carry a CD.
    For ``proposed``, when the CD block alone (plus scaffold and question) exceeds
    ``W``, nodes are evicted lightest first (see ``_evict_cd_to_fit``); the ``cd``
    CorrelationDiagram is then required.
    """
    if arm not in ARMS:
        raise ValueError("arm must be one of %r, got %r" % (ARMS, arm))
    if arm != "proposed" and cd_block:
        raise ValueError("arm %r never includes a CD block" % arm)
    if arm == "A" and W is not None:
        raise ValueError("arm A is the full-context reference: W must be None")
    if arm != "A" and W is None:
        raise ValueError("arm %r needs an integer budget W" % arm)
    if arm == "proposed" and not cd_block:
        raise ValueError("arm 'proposed' needs cd_block")

    rendered = [render_round_trip(rt) for rt in round_trips]
    tok_for_prompt = tokenizer if chat_template else None

    def full_prompt(cd_b: str | None, blks: list[str]) -> str:
        return _full_prompt(cd_b, blks, question, tokenizer=tok_for_prompt)

    # Arm A: everything, chronological, no budget.
    if W is None:
        prompt = full_prompt(None, rendered)
        return _window(prompt, count_tokens(tokenizer, prompt), round_trips, rendered,
                       None, 0, tokenizer)

    # Proposed: the CD block comes first; if it alone does not fit next to the
    # scaffold and the question, evict planets by mass until it does.
    evicted_planets = 0
    if cd_block and count_tokens(tokenizer, full_prompt(cd_block, [])) > W:
        if cd is None:
            raise ValueError(
                "CD block alone exceeds W=%d; pass cd= (CorrelationDiagram) so "
                "it can be evicted by planet mass" % W
            )
        evicted = _evict_cd_to_fit(cd, W, tokenizer, full_prompt)
        evicted_planets = count_planet_lines(cd_block) - count_planet_lines(evicted)
        cd_block = evicted

    # Baseline C: a leading summary block is the compaction result and is always
    # kept (the window is summary + recent round trips).
    pinned: list[int] = []
    if round_trips and is_summary_block(round_trips[0]):
        pinned = [0]
        if count_tokens(tokenizer, full_prompt(cd_block, [rendered[0]])) > W:
            raise ValueError(
                "the summary block alone does not fit W=%d next to the scaffold: "
                "rerun compaction with a larger reserve" % W
            )

    # Newest-first, whole round trips only, stop at the first that does not fit.
    kept: list[int] = list(pinned)  # indices into round_trips
    for i in range(len(round_trips) - 1, len(pinned) - 1, -1):
        trial = sorted(kept + [i])
        prompt = full_prompt(cd_block, [rendered[j] for j in trial])
        if count_tokens(tokenizer, prompt) > W:
            break
        kept.append(i)
    chrono = sorted(kept)
    blocks = [rendered[j] for j in chrono]
    prompt = full_prompt(cd_block, blocks)
    window_tokens = count_tokens(tokenizer, prompt)
    if window_tokens > W:
        raise ValueError(
            "window exceeds the budget even without round trips: %d > W=%d"
            % (window_tokens, W)
        )
    return _window(prompt, window_tokens, [round_trips[j] for j in chrono], blocks,
                   cd_block, evicted_planets, tokenizer)


def _evict_cd_to_fit(
    cd: Any, W: int, tokenizer: Any, full_prompt: Callable[[str | None, list[str]], str],
) -> str:
    """A CD block that fits ``W`` next to the scaffold and the question, built by
    ``CDSerializer.to_context_block_budgeted`` with the "mass" policy (nodes are
    kept in descending mass order, so the lightest are dropped first)."""
    scaffold_tokens = count_tokens(tokenizer, full_prompt("", []))
    budget = W - scaffold_tokens
    if budget <= 0:
        raise ValueError(
            "W=%d leaves no room for a CD block after the %d-token "
            "scaffold" % (W, scaffold_tokens)
        )
    serializer = CDSerializer(level_markers=True)
    counter = make_token_counter(tokenizer)
    for _ in range(MAX_EVICTION_PASSES):
        candidate = serializer.to_context_block_budgeted(
            cd, budget_tokens=budget, policy="mass", token_counter=counter
        )
        overshoot = count_tokens(tokenizer, full_prompt(candidate, [])) - W
        if overshoot <= 0:
            return candidate
        # Token counts are not additive across the block boundary; shrink the
        # block budget by the measured overshoot and retry.
        budget -= max(1, overshoot)
        if budget <= 0:
            raise ValueError("CD eviction could not fit W=%d" % W)
    raise ValueError("CD eviction did not converge after %d passes" % MAX_EVICTION_PASSES)


def _window(
    prompt: str, window_tokens: int, kept_rts: list[dict], blocks: list[str],
    cd_block: str | None, evicted_planets: int, tokenizer: Any,
) -> dict:
    """The ``build_window`` result for the kept round trips (in window order)."""
    real_rts = [rt for rt in kept_rts if not is_summary_block(rt)]
    return {
        "prompt": prompt,
        "window_tokens": window_tokens,
        "n_recent_rts": len(real_rts),
        "recent_idx": [int(rt["idx"]) for rt in real_rts],
        "cd_tokens": count_tokens(tokenizer, cd_block) if cd_block else 0,
        "evicted_planets": evicted_planets,
        "prompt_context": assemble_context(cd_block, blocks),
    }
