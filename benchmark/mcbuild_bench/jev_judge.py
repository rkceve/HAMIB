"""Jev-backed hooks that answer the spec manager's decisions (manager phase).

``SpecManager`` (management/harness/spec_manager.py) builds the correlation
diagram one chunk at a time and needs three kinds of decisions: is a chunk worth
keeping and how general is it, which existing node does a new node belong under,
and does a new node state the same matter as an existing one.  In this
experiment Jev answers all of them.  The adapters here turn each decision into
Jev questions and Jev's answers back into what the manager expects; build_cd
wires them into the manager.

The ``jev`` object is duck-typed against ``jev_client.JevClient``::

    jev.ask(state: str, questions: dict[str, dict]) -> {"answers": dict,
        "usage": {"input_tokens": int, "output_tokens": int},
        "latency_ms": float, "http_status": int, "retries": int}

and the ``summarizer`` against ``summarizer_client.SummarizerClient``::

    summarizer.summarize(excerpt: str) -> str    (<= 120 chars, or SummarizerStop)

How an answer becomes a decision:

- score  -> the level with the highest probability (a missing level counts as
            0, a tie goes to the lower level), mapped to the axis score
            10 + 20 * level on the manager's 0-100 scale;
- noul   -> ``noul >= 0.5``;
- choice -> the chosen option.

There are no default answers.  Anything missing or malformed raises
``JevStop`` (after being counted as ``unparsed``), so build_cd can stop and
still write its partial diagram.

One request per chunk: the keep question and the three axis questions are sent
together, with the raw chunk as the state.  ``JevKeepFn`` sends the request
through ``JevNodeFn.ask_chunk``, which parses and caches the answer;
``JevNodeFn`` then takes the scores from the cache and asks the summarizer for
the node text (only kept chunks get that far).  This relies on the manager
calling ``keep_fn(text)`` and then ``node_fn(text)`` on the same text; anything
else raises RuntimeError rather than paying for a second request.

Placement (``belongs``) and same-matter (``same``) decisions over several
candidates are asked as ONE Choice question per batch of at most 254
candidates (Jev allows 255 options; one is the "none" option).  Asking one Noul
per candidate would cost hundreds of thousands of requests over this corpus.  A
single candidate is asked as one pairwise Noul.  Every request is counted in
the shared ``JevCounters`` under its decision kind, so
``SpecManager.call_totals()`` reports it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from benchmark.mcbuild_bench.errors import JevStop, SummarizerNodeTextUnusable
from benchmark.mcbuild_bench.jev_client import argmax_level as _argmax_level
from benchmark.mcbuild_bench.jev_client import choice_of, noul_of
from management.harness.prompts import K_BELONGS, K_NODE, K_SAME
from management.harness.spec_manager import AXES

# -- fixed question texts (the experiment design fixes them; do not edit) -----

MAX_SUN_CHOICES = 254  # choice <= 255 options, one of which is new_topic / none
NEW_TOPIC = "new_topic"
NONE_ITEM = "none"
K_KEEP = "keep"
K_SUN = "sun"
K_PLANET = "planet"

# Axis score for level index k in 0..4.
LEVEL_SCORES: tuple[int, ...] = (10, 30, 50, 70, 90)
N_LEVELS = len(LEVEL_SCORES)

# Facts often sit inside tool output (values, paths, settings), so the keep
# question keeps them; only content with no lasting information is dropped.
Q_KEEP: dict[str, Any] = {
    "type": "noul",
    "instructions": (
        "Does this excerpt contain information worth remembering later - a fact, "
        "value, setting, result, decision, definition, instruction or requirement "
        "- even if it appears inside code, command output or logs? Answer no only "
        "for content with no lasting information (boilerplate, progress noise, "
        "repeated listings)."
    ),
    "criteria": {
        "true": (
            "Contains at least one concrete fact, value, decision or instruction "
            "worth recalling later"
        ),
        "false": "No lasting information: boilerplate, progress noise, or repetition",
    },
}

Q_AXES: dict[str, dict[str, Any]] = {
    "comprehensiveness": {
        "type": "score",
        "instructions": "How broad is the matter this excerpt states?",
        "criteria": [
            "A single detail of something larger",
            "A minor point",
            "A self-standing point",
            "A major theme with several parts",
            "The overarching topic of a whole discussion",
        ],
    },
    "independence": {
        "type": "score",
        "instructions": "Can this excerpt be understood on its own?",
        "criteria": [
            "Meaningless without its surrounding context",
            "Mostly dependent on context",
            "Partly self-contained",
            "Mostly self-contained",
            "Fully self-contained",
        ],
    },
    "detail": {
        "type": "score",
        "instructions": "How specific is this excerpt?",
        "criteria": [
            "Very general",
            "General",
            "Moderately specific",
            "Specific",
            "A precise concrete detail",
        ],
    },
}

Q_SUN_INSTRUCTIONS = (
    "Which existing topic does this excerpt belong to? Pick new_topic if none fits."
)
NEW_TOPIC_CRITERION = "None of the listed topics"
Q_PLANET_INSTRUCTIONS = (
    "Which of the listed items is this excerpt a detail or sub-point of? Pick none "
    "if it belongs under none of them."
)
NONE_ITEM_CRITERION = "None of the listed items"
# The `same` Choice batch, used when a node has more than one candidate.
Q_SAME_CHOICE_INSTRUCTIONS = (
    "Which of the listed items states the same matter as this excerpt? Pick none "
    "if no item states the same matter."
)
NONE_SAME_CRITERION = "No listed item states the same matter"

Q_PAIR: dict[str, dict[str, Any]] = {
    K_SAME: {"type": "noul", "instructions": "Do A and B state the same matter?"},
    K_BELONGS: {
        "type": "noul",
        "instructions": "Is A a detail or sub-point that belongs under topic B?",
    },
}


def pair_state(a: str, b: str) -> str:
    return f"A: {a}\nB: {b}"


def _choice_question(
    texts: list[str], *, prefix: str, offset: int, instructions: str,
    none_key: str, none_criterion: str,
) -> dict[str, Any]:
    """One Choice batch: ``{prefix}{offset + i}`` -> text, plus the none option.
    A batch above ``MAX_SUN_CHOICES`` is a programming error (``most_similar``
    slices the candidates), not a Jev stop."""
    if len(texts) > MAX_SUN_CHOICES:
        raise ValueError(
            "a Choice batch holds at most %d candidates, got %d" % (MAX_SUN_CHOICES, len(texts))
        )
    criteria = {f"{prefix}{offset + i}": text for i, text in enumerate(texts)}
    criteria[none_key] = none_criterion
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def sun_question(sun_texts: list[str], offset: int = 0) -> dict[str, Any]:
    """The `sun` choice question over one batch of existing sun node texts +
    new_topic; keys ``s<offset + i>``."""
    return _choice_question(
        sun_texts, prefix="s", offset=offset, instructions=Q_SUN_INSTRUCTIONS,
        none_key=NEW_TOPIC, none_criterion=NEW_TOPIC_CRITERION,
    )


def planet_question(texts: list[str], offset: int = 0) -> dict[str, Any]:
    """The `planet` choice question over one batch of planet / satellite /
    provisional-sun candidate texts + none; keys ``p<offset + i>``."""
    return _choice_question(
        texts, prefix="p", offset=offset, instructions=Q_PLANET_INSTRUCTIONS,
        none_key=NONE_ITEM, none_criterion=NONE_ITEM_CRITERION,
    )


def same_question(texts: list[str], offset: int = 0) -> dict[str, Any]:
    """The `same` choice question over one batch of candidate node texts +
    none; keys ``m<offset + i>``."""
    return _choice_question(
        texts, prefix="m", offset=offset, instructions=Q_SAME_CHOICE_INSTRUCTIONS,
        none_key=NONE_ITEM, none_criterion=NONE_SAME_CRITERION,
    )


# -- answer parsing (pure; no defaults) ----------------------------------------


def _answer(answers: Any, qid: str) -> dict[str, Any]:
    if not isinstance(answers, dict) or qid not in answers:
        raise JevStop(f"answer for {qid!r} missing")
    answer = answers[qid]
    if not isinstance(answer, dict):
        raise JevStop(f"answer for {qid!r} is not an object")
    return answer


def argmax_level(answer: dict[str, Any], n_levels: int = N_LEVELS) -> int:
    """``jev_client.argmax_level`` bound to this module's level table: the
    winning index must be < ``n_levels`` (default len(LEVEL_SCORES) = 5)."""
    return _argmax_level(answer, n_levels=n_levels)


__all__ = ["argmax_level", "choice_of", "noul_of"]  # re-exported jev_client parsers


# -- shared counters -----------------------------------------------------------


class JevCounters:
    """Per-kind call counters in the shape SpecManager reads from a judge
    runner (``calls`` / ``unparsed`` / ``defaulted``; ``retried`` as in
    JudgeRunner), plus the token usage of the completed requests.

    ``defaulted`` stays empty: Jev answers are never replaced by defaults.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: dict[str, int] = {}
        self.unparsed: dict[str, int] = {}
        self.defaulted: dict[str, int] = {}
        self.retried: dict[str, int] = {}
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def _bump(self, counter: dict[str, int], kind: str, delta: int = 1) -> None:
        if delta:
            with self._lock:
                counter[kind] = counter.get(kind, 0) + delta

    def total_calls(self) -> int:
        return sum(self.calls.values())

    def total_unparsed(self) -> int:
        return sum(self.unparsed.values())

    def total_defaulted(self) -> int:
        return sum(self.defaulted.values())

    def total_retried(self) -> int:
        return sum(self.retried.values())

    def record(self, kind: str, result: Any) -> None:
        """Account one completed ``jev.ask`` under ``kind``."""
        self._bump(self.calls, kind)
        if isinstance(result, dict):
            usage = result.get("usage") or {}
            with self._lock:
                self.input_tokens += int(usage.get("input_tokens", 0) or 0)
                self.output_tokens += int(usage.get("output_tokens", 0) or 0)
            self._bump(self.retried, kind, int(result.get("retries", 0) or 0))


def _ask(jev: Any, counters: JevCounters, kind: str, state: str, questions: dict) -> dict:
    """One Jev request, counted under ``kind``; returns the ``answers`` map.

    A malformed reply (no ``answers`` object) is ``unparsed`` and stops the run.
    A JevStop raised by ``jev.ask`` itself (HTTP failure after the retries,
    validation error) is also booked under ``unparsed[kind]`` before it
    propagates, so the counters show which decision kind failed.
    """
    try:
        result = jev.ask(state, questions)
    except JevStop:
        counters._bump(counters.unparsed, kind)
        raise
    counters.record(kind, result)
    answers = result.get("answers") if isinstance(result, dict) else None
    if not isinstance(answers, dict):
        counters._bump(counters.unparsed, kind)
        raise JevStop("Jev reply without answers")
    return answers


def _parse(counters: JevCounters, kind: str, fn, *args):
    """Run a pure parser; a JevStop from it counts as ``unparsed`` for ``kind``."""
    try:
        return fn(*args)
    except JevStop:
        counters._bump(counters.unparsed, kind)
        raise


# -- adapters ------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkAnswer:
    """The parsed answer of the one request made for a raw chunk."""

    keep: bool
    scores: dict[str, int]


class JevNodeFn:
    """``SpecManager(node_fn=...)``: the node text and axis scores of a kept chunk.

    The scores come from the chunk's one Jev request (keep + the three axes,
    sent by ``ask_chunk``, normally through ``JevKeepFn``), mapped to
    10 + 20 * level; the node text comes from the local summarizer.  A missing
    Jev answer raises JevStop and an unreachable summarizer raises
    SummarizerStop.  When the summarizer cannot produce a usable node text,
    ``__call__`` returns None; ``SpecManager._node_or_dropped`` then counts a
    ``node_fallback`` and uses the truncated chunk text, as it always has.
    """

    def __init__(self, jev: Any, summarizer: Any, counters: JevCounters | None = None) -> None:
        self.jev = jev
        self.summarizer = summarizer
        self.counters = counters if counters is not None else JevCounters()
        self.summarizer_calls: int = 0
        self._lock = threading.Lock()
        # Single-slot cache: (raw chunk text, parsed answer) of the LAST request.
        self._last: tuple[str, ChunkAnswer] | None = None

    def ask_chunk(self, text: str) -> ChunkAnswer:
        """ONE Jev request for ``text``; every answer parsed now (a malformed
        one stops the run here, under K_NODE); the result is cached for
        ``__call__``."""
        questions: dict[str, Any] = {K_KEEP: Q_KEEP, **Q_AXES}
        answers = _ask(self.jev, self.counters, K_NODE, text, questions)
        keep_value = _parse(self.counters, K_NODE, lambda: noul_of(_answer(answers, K_KEEP)))
        scores: dict[str, int] = {}
        for axis in AXES:
            level = _parse(self.counters, K_NODE, lambda a=axis: argmax_level(_answer(answers, a)))
            scores[axis] = LEVEL_SCORES[level]
        chunk = ChunkAnswer(keep=keep_value >= 0.5, scores=scores)
        with self._lock:
            self._last = (text, chunk)
        return chunk

    def cached(self, text: str) -> ChunkAnswer | None:
        with self._lock:
            last = self._last
        return last[1] if last is not None and last[0] == text else None

    def __call__(self, text: str) -> tuple[str, dict[str, int]] | None:
        chunk = self.cached(text)
        if chunk is None:
            raise RuntimeError(
                "JevNodeFn called for a chunk that keep_fn did not ask first: the spec "
                "manager calls keep_fn(text) then node_fn(text) on the same text (H1 / D3 "
                "one-request contract); no second Jev request is made"
            )
        try:
            summary = self.summarizer.summarize(text)
        except SummarizerNodeTextUnusable:
            with self._lock:
                self.summarizer_calls += 1
            return None
        with self._lock:
            self.summarizer_calls += 1
        return summary, chunk.scores


class JevKeepFn:
    """``SpecManager(keep_fn=...)``: the `keep` answer of the chunk's one Jev
    request, which is sent here through ``JevNodeFn.ask_chunk``.  Tool output is
    fed to the manager too, and this question is what lets it drop noise."""

    def __init__(self, node_fn: JevNodeFn) -> None:
        self.node_fn = node_fn
        self.jev = node_fn.jev
        self.counters = node_fn.counters

    def __call__(self, text: str) -> bool:
        return self.node_fn.ask_chunk(text).keep


class JevSimilarityJudge:
    """``SpecManager(similarity=...)``: the belongs / same-matter decisions.

    ``most_similar(query, candidates, kind)`` returns ``(index, 1.0)`` for the
    chosen candidate or ``(-1, 0.0)`` for none:

    - no candidates: ``(-1, 0.0)`` without a request;
    - one candidate: one pairwise Noul (`same` or `belongs`) on
      ``"A: <query>\nB: <candidate>"``; ``noul >= 0.5`` picks it;
    - several candidates: one Choice request per batch of <= 254 candidates,
      in candidate order; the first batch whose answer is not the none option
      wins.  K_SAME uses the `same` wording.  K_BELONGS uses the `sun` wording
      (+ new_topic) when every candidate is a current sun text, otherwise the
      `planet` wording (+ none).

    ``sun_texts_fn`` returns the texts of the suns currently in the diagram
    being built (build_cd wires ``{se.sun.text for se in cd.suns}``); without
    it every K_BELONGS batch uses the planet wording.  ``runner`` exposes the
    counters SpecManager reads per turn.
    """

    def __init__(
        self,
        jev: Any,
        counters: JevCounters | None = None,
        *,
        sun_texts_fn: Callable[[], set[str]] | None = None,
    ) -> None:
        self.jev = jev
        self._counters = counters if counters is not None else JevCounters()
        self._sun_texts_fn = sun_texts_fn

    @property
    def runner(self) -> JevCounters:
        return self._counters

    @property
    def calls(self) -> dict[str, int]:
        return self._counters.calls

    @property
    def unparsed(self) -> dict[str, int]:
        return self._counters.unparsed

    @property
    def defaulted(self) -> dict[str, int]:
        return self._counters.defaulted

    @property
    def retried(self) -> dict[str, int]:
        return self._counters.retried

    def most_similar(
        self, query: str, candidates: list[str], kind: str = K_SAME
    ) -> tuple[int, float]:
        if kind not in Q_PAIR:
            raise ValueError(f"unknown pairwise kind {kind!r}")
        if not candidates:
            return -1, 0.0
        if len(candidates) > 1:
            return self._choose(query, list(candidates), kind)
        question = {kind: Q_PAIR[kind]}
        for idx, candidate in enumerate(candidates):
            answers = _ask(self.jev, self._counters, kind, pair_state(query, candidate), question)
            value = _parse(self._counters, kind, lambda: noul_of(_answer(answers, kind)))
            if value >= 0.5:
                return idx, 1.0
        return -1, 0.0

    def _all_suns(self, candidates: list[str]) -> bool:
        """True iff every candidate is a current sun node text (selects the
        `sun` wording; otherwise the `planet` wording is used)."""
        if self._sun_texts_fn is None:
            return False
        suns = self._sun_texts_fn()
        return all(c in suns for c in candidates)

    def _choose(self, query: str, candidates: list[str], kind: str) -> tuple[int, float]:
        """Choice batches of <= MAX_SUN_CHOICES over ``candidates``, counted
        under ``kind`` (K_BELONGS or K_SAME)."""
        if kind == K_SAME:
            qid, build, prefix, none_key = K_SAME, same_question, "m", NONE_ITEM
        elif self._all_suns(candidates):
            qid, build, prefix, none_key = K_SUN, sun_question, "s", NEW_TOPIC
        else:
            qid, build, prefix, none_key = K_PLANET, planet_question, "p", NONE_ITEM
        for start in range(0, len(candidates), MAX_SUN_CHOICES):
            batch = candidates[start:start + MAX_SUN_CHOICES]
            question = build(batch, start)
            answers = _ask(self.jev, self._counters, kind, query, {qid: question})
            # A choice (or probability key) outside the offered options stops the run.
            options = list(question["criteria"])
            choice = _parse(
                self._counters, kind, lambda: choice_of(_answer(answers, qid), options=options)
            )
            if choice == none_key:
                continue
            if choice.startswith(prefix) and choice[len(prefix):].isdigit():
                idx = int(choice[len(prefix):])
                if start <= idx < start + len(batch):
                    return idx, 1.0
            self._counters._bump(self._counters.unparsed, kind)
            raise JevStop(f"{qid} choice {choice!r} is not an offered option")
        return -1, 0.0


# -- request projection (sizing a run before spending money) ------------------


def choice_batches(n_candidates: int) -> int:
    """Requests of one K_BELONGS or K_SAME decision over ``n_candidates``:
    ceil(n / 254) Choice batches; 1 candidate is one `belongs` / `same` Noul,
    also one request; 0 candidates cost nothing."""
    return -(-int(n_candidates) // MAX_SUN_CHOICES)


def within_turn_requests(n_chunks: int) -> int:
    """Worst case of ``SpecManager.build_provisional`` for a turn of ``n_chunks``
    nodes split into S provisional suns, P planets and T satellites (S + P + T
    = n): each planet asks ONE K_BELONGS decision over the S suns
    (``choice_batches(S)`` requests), each satellite one over the P planets
    (``choice_batches(P)``).  Maximised over every (S, P)."""
    n = int(n_chunks)
    best = 0
    for suns in range(n + 1):
        for planets in range(n - suns + 1):
            satellites = n - suns - planets
            cost = planets * choice_batches(suns) + satellites * choice_batches(planets)
            if cost > best:
                best = cost
    return best


def project_requests(n_chunks: int, n_suns: int, n_planets: int, n_satellites: int = 0) -> int:
    """Worst-case number of Jev REQUESTS (``jev.ask`` calls; 429/529 retries
    are attempts inside one request and are not counted) that ONE round trip
    of ``n_chunks`` chunks can cost against a diagram that holds ``n_suns``
    suns, ``n_planets`` planets and ``n_satellites`` satellites BEFORE the
    round trip, following the request routing of jev_judge, spec_manager and
    graph_merger.  Every chunk is assumed kept (a dropped chunk costs exactly
    its classification request, never more).

    Terms, each an upper bound of one code path::

        classification  n_chunks
            ONE request {keep, 3 axes} per chunk (JevNodeFn.ask_chunk).
        within-turn     within_turn_requests(n_chunks)
            SpecManager.build_provisional: each planet-level node asks ONE
            `belongs` decision over the provisional suns of the turn, each
            satellite-level node one over the planet-level nodes of the turn;
            a decision over k candidates is choice_batches(k) requests.
        merge           n_chunks * (same + belongs)
            same    = choice_batches(k),  k = n_suns + n_planets + n_satellites + n_chunks
                Each node is merged once and asks ONE `same` decision
                (Choice batches) over one candidate set:
                a sun node against every current sun
                (GraphMerger._find_matching_sun_idx); a planet under a matched
                sun against that sun's planets; a satellite under a matched
                planet against its satellites; an orphan planet against ALL
                planets; an orphan satellite against ALL satellites.  ``k``
                bounds every one of those sets; nodes added earlier in the
                same round trip are candidates too, hence the ``+ n_chunks``.
            belongs = choice_batches(n_planets + n_chunks) + choice_batches(n_suns + n_chunks)
                An orphan satellite asks ONE `belongs` decision over ALL
                planets (Choice batches) and then one sun Choice; an orphan
                planet only the sun Choice (bounded by the same sum).

    Because both decisions use Choice batches, the merge term per node grows
    like ceil(diagram / 254) instead of linearly (one `same` Noul per
    candidate) or quadratically (one `belongs` Noul per candidate pair).
    """
    for name, value in (
        ("n_chunks", n_chunks), ("n_suns", n_suns),
        ("n_planets", n_planets), ("n_satellites", n_satellites),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    classification = n_chunks
    within_turn = within_turn_requests(n_chunks)
    same = choice_batches(n_suns + n_planets + n_satellites + n_chunks)
    belongs = choice_batches(n_planets + n_chunks) + choice_batches(n_suns + n_chunks)
    merge = n_chunks * (same + belongs)
    return classification + within_turn + merge


# -- input-token / USD estimate (an ESTIMATE, not the budget guard) ------------


def _decision_cost(
    n_candidates: int, avg_option_tokens: int, instruction_tokens: int
) -> tuple[int, int]:
    """(requests, input_tokens) of ONE decision over ``n_candidates`` with
    Choice batching: ceil(n / 254) requests, every candidate text sent once,
    plus instructions + state (one node text) per request.  A single-candidate
    Noul has the same shape (instructions + two texts)."""
    requests = choice_batches(n_candidates)
    tokens = requests * (instruction_tokens + avg_option_tokens) + n_candidates * avg_option_tokens
    return requests, tokens


def estimate_input_tokens(
    n_chunks: int,
    kept_fraction: float,
    avg_option_tokens: int = 50,
    *,
    avg_chunk_tokens: int = 100,
    question_tokens: int = 150,
    instruction_tokens: int = 40,
    n_round_trips: int = 1,
    price_per_mtok: float = 0.042,
) -> dict[str, Any]:
    """ESTIMATE of the Jev input tokens (and USD at ``price_per_mtok``) of one
    build_cd run over ``n_chunks`` chunks of which ``kept_fraction`` become
    nodes.  This is an expected-spend figure to read BEFORE a run; the budget
    guard (build_cd --max-jev-input-tokens) is what actually stops a run.
    Nothing here is measured; every number is an assumption:

      * classification: one request per chunk, ``avg_chunk_tokens`` of state
        + ``question_tokens`` for the four fixed questions {keep, 3 axes}.
      * kept = round(n_chunks * kept_fraction) nodes, each ``avg_option_tokens``
        long (node texts are <= 120 chars) whether it is a state or an option.
      * within-turn: the kept nodes are spread evenly over ``n_round_trips``
        turns; each kept node asks ONE decision over about half of its turn's
        kept nodes (build_provisional: planets over the turn's suns, satellites
        over the turn's planets).
      * same: the i-th kept node (i from 0) asks ONE `same` decision over i
        candidates (the diagram grows by one node per kept chunk; Choice
        batches, ceil(i / 254) requests, every candidate text sent once).
      * belongs: the i-th kept node asks two `belongs` decisions (planet
        Choice, sun Choice) over the same i candidates split in half
        (ceil(i / 2) each); every kept node is assumed to reach both (an upper
        estimate: a node placed by `same` asks none).
      * a decision over 0 candidates costs nothing; over 1 candidate it is one
        Noul with the same token shape (instructions + two texts).

    Returns {"estimate": True, "kept_nodes", "terms": {name: {"requests",
    "input_tokens"}}, "requests_total", "input_tokens_total",
    "price_per_mtok_usd", "usd", "assumptions": {...}}.
    """
    if not isinstance(n_chunks, int) or isinstance(n_chunks, bool) or n_chunks < 0:
        raise ValueError(f"n_chunks must be a non-negative int, got {n_chunks!r}")
    if not (0.0 <= float(kept_fraction) <= 1.0):
        raise ValueError(f"kept_fraction must lie in [0, 1], got {kept_fraction!r}")
    if n_round_trips < 1:
        raise ValueError(f"n_round_trips must be >= 1, got {n_round_trips!r}")
    kept = int(round(n_chunks * float(kept_fraction)))
    terms: dict[str, dict[str, int]] = {}

    terms["classification"] = {
        "requests": n_chunks,
        "input_tokens": n_chunks * (avg_chunk_tokens + question_tokens),
    }

    per_node_candidates = int(round(kept / n_round_trips / 2))
    wt_req, wt_tok = _decision_cost(per_node_candidates, avg_option_tokens, instruction_tokens)
    terms["within_turn"] = {"requests": kept * wt_req, "input_tokens": kept * wt_tok}

    same_req = same_tok = 0
    bel_req = bel_tok = 0
    for i in range(kept):
        r, t = _decision_cost(i, avg_option_tokens, instruction_tokens)
        same_req += r
        same_tok += t
        r, t = _decision_cost(-(-i // 2), avg_option_tokens, instruction_tokens)
        bel_req += 2 * r
        bel_tok += 2 * t
    terms["same"] = {"requests": same_req, "input_tokens": same_tok}
    terms["belongs"] = {"requests": bel_req, "input_tokens": bel_tok}

    requests_total = sum(t["requests"] for t in terms.values())
    tokens_total = sum(t["input_tokens"] for t in terms.values())
    return {
        "estimate": True,
        "kept_nodes": kept,
        "terms": terms,
        "requests_total": requests_total,
        "input_tokens_total": tokens_total,
        "price_per_mtok_usd": price_per_mtok,
        "usd": tokens_total * price_per_mtok / 1_000_000,
        "assumptions": {
            "n_chunks": n_chunks,
            "kept_fraction": float(kept_fraction),
            "avg_option_tokens": avg_option_tokens,
            "avg_chunk_tokens": avg_chunk_tokens,
            "question_tokens": question_tokens,
            "instruction_tokens": instruction_tokens,
            "n_round_trips": n_round_trips,
        },
    }


class JevJudgeLLM:
    """The ``JudgeLLM`` handed to SpecManager in this experiment.

    Every decision is routed through node_fn / keep_fn / similarity, so a text
    prompt must never reach the backend.  Reaching it is a bug, not a fallback.
    """

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        raise NotImplementedError(
            "JevJudgeLLM.complete() must never be called: the spec manager's "
            "decisions are routed through the Jev hooks"
        )
