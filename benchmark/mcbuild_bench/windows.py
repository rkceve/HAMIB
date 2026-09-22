"""windows.py — window composition for the mcbuild-bench arms (DESIGN.md §6, DECISIONS C6).

One window per (arm, W) cell, built ONCE and shared by every question of the cell
(post-hoc design, B3).  Composition rule (C6)::

    [CD block (proposed only)] + [most recent WHOLE round trips that fit, emitted in
    chronological order] ; the question is appended by the reader scaffold.

Token budget ``W`` bounds the FINAL reader prompt, i.e. the string the reader
actually tokenizes.  That string is built by ``benchmark.bineval.run_reader``
(``READER_PROMPT`` / ``build_prompt``) around the ``context_block`` this module
returns, so the budget is measured on ``run_reader.build_prompt(context, question)``
with ``tokenizer(text, add_special_tokens=False)`` — never on the body alone.

Reconciliation with DESIGN.md §8 (documented, not hidden): ``run_reader`` already
appends the instruction + ``Question:`` + ``Answer:`` lines and does NOT apply a
chat template.  Editing run_reader is out of scope, so this module (a) produces
ONLY the context text, wrapped in the §8 ``<context>`` / ``</context>`` lines, and
(b) counts the scaffold with run_reader's real template.  The final prompt is
therefore identical across arms (C8) and W bounds its token count.

Round-trip rendering is fixed (work-package contract)::

    "### Human\\n" + human + "\\n" + "\\n".join(f"### {kind}\\n{text}" for each event)

Baseline C passes its compaction summary as the FIRST round-trip-like block; use
``summary_block(text)`` to build it (rendered as ``### Summary\\n<text>``).
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
    """The full reader prompt. ``tokenizer`` (H6 alternative / H30) wraps it in the
    chat template with thinking off; W is then measured on that same string."""
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
    - ``prompt_context``  the context block only — the argument for
                          ``run_reader.run_reader(llm, context_block, ...)``, which
                          rebuilds exactly ``prompt`` around it;
    - ``window_tokens``   tokens of ``prompt``;
    - ``n_recent_rts``    number of real round trips kept (a baseline-C summary
                          block is not counted);
    - ``recent_idx``      their ``idx`` values in chronological order (H22 (d):
                          ``min(recent_idx)`` is the oldest round trip whose facts
                          lie INSIDE the window);
    - ``cd_tokens``       tokens of the CD block actually used (0 without CD);
    - ``evicted_planets`` ``[PN`` lines dropped by budgeted eviction (0 if none).

    ``W=None`` is arm A: every round trip, no CD.  Arms B and C never carry a CD.
    For ``proposed``, when the CD block alone (plus scaffold and question) exceeds
    ``W``, it is replaced by ``CDSerializer(level_markers=True)
    .to_context_block_budgeted(cd, W - scaffold_tokens, "mass", counter)``; the
    ``cd`` CorrelationDiagram is then required.
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
    evicted_planets = 0
    tok_for_prompt = tokenizer if chat_template else None

    def full_prompt(cd_b: str | None, blks: list[str]) -> str:
        return _full_prompt(cd_b, blks, question, tokenizer=tok_for_prompt)

    # Arm A: everything, chronological, no budget.
    if W is None:
        prompt = full_prompt(None, rendered)
        return {
            "prompt": prompt,
            "window_tokens": count_tokens(tokenizer, prompt),
            "n_recent_rts": sum(1 for rt in round_trips if not is_summary_block(rt)),
            "recent_idx": [int(rt["idx"]) for rt in round_trips if not is_summary_block(rt)],
            "cd_tokens": 0,
            "evicted_planets": 0,
            "prompt_context": assemble_context(None, rendered),
        }

    # Proposed: the CD block is pinned first; evict by planet mass if it alone
    # does not fit next to the scaffold and the question.
    if cd_block:
        if count_tokens(tokenizer, full_prompt(cd_block, [])) > W:
            if cd is None:
                raise ValueError(
                    "CD block alone exceeds W=%d; pass cd= (CorrelationDiagram) so "
                    "it can be evicted by planet mass" % W
                )
            planets_before = count_planet_lines(cd_block)
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
                    break
                # Token counts are not additive across the block boundary; shrink
                # the block budget by the measured overshoot and retry.
                budget -= max(1, overshoot)
                if budget <= 0:
                    raise ValueError("CD eviction could not fit W=%d" % W)
            else:
                raise ValueError(
                    "CD eviction did not converge after %d passes" % MAX_EVICTION_PASSES
                )
            cd_block = candidate
            evicted_planets = planets_before - count_planet_lines(cd_block)

    # Baseline C: a leading summary block is the compaction result and is PINNED
    # (C4: window = summary + recent round trips); it is never evicted.
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
    return {
        "prompt": prompt,
        "window_tokens": window_tokens,
        "n_recent_rts": sum(1 for j in chrono if not is_summary_block(round_trips[j])),
        "recent_idx": [int(round_trips[j]["idx"]) for j in chrono if not is_summary_block(round_trips[j])],
        "cd_tokens": count_tokens(tokenizer, cd_block) if cd_block else 0,
        "evicted_planets": evicted_planets,
        "prompt_context": assemble_context(cd_block, blocks),
    }
