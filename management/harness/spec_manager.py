"""SpecManager: the manager procedure exactly as the specification describes
it, with one node per chunk.

This is the default experiment path; ``manager.HarnessManager`` is the optional
mode with extra checks.  Compared with it, SpecManager has no statement
extraction, no support check, no within-turn de-duplication, no LLM boundary
merge, and no evaluation step.

Per turn:
  1. chunk      split_candidates on each side, no merging
  2. one node   one Q_NODE call per chunk -> summary + three axis scores
  3. link       this turn's nodes to each other only
  4. merge      into the base diagram (GraphMerger case 1 / 2 / 3)
  5. normalize  exactly once, in a ``finally``, with the spec's planet mass floor

Skipping turns that only ask a question is the caller's job.  Nothing here
loads a model: the judge is injected (``make_spec_fake_judge`` for
``--judge fake``).
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Sequence

import jsonschema

from management.graph_merger import GraphMerger
from management.harness.backends import FakeJudge, first_fenced_span
from management.harness.chunking import split_candidates
from management.harness.judge import JudgeCache, JudgeLLM, JudgeRunner
from management.harness.manager import (
    ProvisionalStructure,
    link_within_turn,
    merge_into_base,
)
from management.harness.prompts import (
    ALL_KINDS,
    DEFAULT_NODE_SCORES,
    FALLBACK_NODE_CHARS,
    K_BELONGS,
    K_NODE,
    Q_NODE,
    RETRY_JSON_OBJECT_SUFFIX,
)
from management.harness.similarity_judge import MATCH_SCORE, EmbedFn, SimilarityJudge
from models.correlation_diagram import CorrelationDiagram
from models.node import Node, NodeLevel
from utils.config import get

# The three classification axes, in reporting order.
AXES: tuple[str, ...] = ("comprehensiveness", "independence", "detail")

# Scores outside 0..100 are clamped, not rejected: a model answering 120 still
# ranked the axis, and rejecting the reply would cost a retry and maybe a
# fallback node.  The reply's shape is checked by NODE_OBJECT_SCHEMA.
SCORE_MIN = 0
SCORE_MAX = 100

NODE_OBJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "minLength": 1},
        "comprehensiveness": {"type": "integer"},
        "independence": {"type": "integer"},
        "detail": {"type": "integer"},
    },
    "required": ["summary", "comprehensiveness", "independence", "detail"],
}

# Answer-quality counters, reported per turn and summed in ``totals``.
SPEC_QUALITY_KEYS: tuple[str, ...] = (
    "unparsed",
    "defaulted",
    "node_fallback",
    "vanished",
    "attached",
)


@dataclass
class SpecChunk:
    """One chunk of a turn.

    Same fields as ``management.text_chunker.Chunk``, declared here so this
    module does not import the Japanese-only TextChunker.
    """

    text: str
    source: str  # "user" | "assistant"
    turn: int


# -- config ------------------------------------------------------------------


@dataclass
class SpecConfig:
    """The ``spec_manager:`` section of config.yaml."""

    chunk_max_chars: int = 400
    max_node_chars: int = 120
    shortlist_k: int = 5
    judge_max_tokens: int = 256
    node_max_tokens: int = 200
    max_retries: int = 1
    # The spec gives a planet with no satellites mass 0.  The diagram's own
    # default floor (1.0) is kept for every other caller.
    planet_mass_floor: float = 0.0
    # Threads for the Q_NODE calls of one turn; 1 = sequential.  Safe because
    # each chunk's node is independent of the others.  Linking and merging stay
    # sequential: they mutate the diagram and their order decides the result.
    max_workers: int = 1

    @classmethod
    def from_config(cls) -> "SpecConfig":
        """Read config.yaml; each value is cast to the type of its default."""
        defaults = cls()
        values: dict[str, Any] = {}
        for f in fields(cls):
            default = getattr(defaults, f.name)
            values[f.name] = type(default)(get("spec_manager", f.name, default))
        return cls(**values)


# -- report ------------------------------------------------------------------


@dataclass
class SpecTurnReport:
    """Per-turn statistics."""

    calls: dict[str, int] = field(default_factory=dict)
    cache_hits: int = 0
    chunks: int = 0
    nodes: int = 0
    node_fallback: int = 0
    unparsed: int = 0
    defaulted: int = 0
    vanished: int = 0
    attached: int = 0
    promoted: int = 0
    added: int = 0
    # Chunks the optional ``keep_fn`` rejected before a node was made (the
    # benchmark drops code, tool output and logs, which are not worth
    # remembering).  Always 0 without a keep_fn.
    dropped: int = 0

    def total_calls(self) -> int:
        return sum(self.calls.values())


# -- parsing helpers ---------------------------------------------------------


def clamp_score(value: int) -> int:
    return max(SCORE_MIN, min(SCORE_MAX, int(value)))


def first_json_object(raw: str) -> str | None:
    """The first balanced ``{...}`` span of ``raw``, or None.

    Taking the first ``{`` to the last ``}`` instead would glue together a
    reply that holds two objects (say, the answer and an example) into
    something that does not parse.  Braces inside JSON strings are ignored.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0:
                    return raw[start : i + 1]
    return None


def _coerce_score(value: Any) -> int | None:
    """Accept 90, "90", 90.0 and " 90 " as a score; None for anything else.

    Models often answer with strings or floats, and rejecting those would throw
    away a correct ranking.
    """
    if isinstance(value, bool):  # bool is an int subclass, but not a score
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def _lift_nested_scores(parsed: Any) -> Any:
    """Flatten ``{"summary": ..., "scores": {...}}`` into one object.

    Only axis keys missing at the top level are taken from ``scores``, so
    top-level values win.
    """
    if not isinstance(parsed, dict):
        return parsed
    nested = parsed.get("scores")
    if not isinstance(nested, dict):
        return parsed
    lifted = dict(parsed)
    lifted.pop("scores", None)
    for axis in AXES:
        if axis not in lifted and axis in nested:
            lifted[axis] = nested[axis]
    return lifted


def parse_node_object(raw: str, max_chars: int) -> tuple[str, dict[str, int]] | None:
    """Parse a Q_NODE reply into ``(summary, scores)``, or None if unusable.

    Takes the first JSON object, flattens a nested ``scores`` block, coerces
    string/float scores, validates against NODE_OBJECT_SCHEMA, trims the
    summary to ``max_chars`` and clamps the scores to 0..100.  On None the
    caller counts the reply as unparsed, retries, and finally falls back.
    """
    span = first_json_object(raw)
    if span is None:
        return None
    try:
        parsed = json.loads(span)
    except Exception:
        return None
    parsed = _lift_nested_scores(parsed)
    if isinstance(parsed, dict):
        parsed = dict(parsed)
        for axis in AXES:
            if axis in parsed:
                value = _coerce_score(parsed[axis])
                if value is None:
                    return None  # e.g. "hi": not a score in any shape
                parsed[axis] = value
    try:
        jsonschema.validate(parsed, NODE_OBJECT_SCHEMA)
    except Exception:
        return None
    summary = str(parsed["summary"]).strip()[:max_chars].strip()
    if not summary:
        return None
    scores = {axis: clamp_score(parsed[axis]) for axis in AXES}
    return summary, scores


def level_from_scores(scores: dict[str, int]) -> NodeLevel:
    """The level of the highest-scoring axis; ties go to the more specific
    level (detail > independence > comprehensiveness).

    Ties go down because most nodes in the hand-built reference diagram are
    satellites (207 of 240).

        detail   indep    compr    level
        top      -        -        satellite   (detail wins outright and on ties)
        <        top      -        planet
        <        <        top      sun
        0        0        0        satellite
    """
    detail = scores["detail"]
    independence = scores["independence"]
    comprehensiveness = scores["comprehensiveness"]
    best = max(detail, independence, comprehensiveness)
    if best <= 0:
        return NodeLevel.SATELLITE
    if detail == best:
        return NodeLevel.SATELLITE
    if independence == best:
        return NodeLevel.PLANET
    return NodeLevel.SUN


# -- fake judge for --judge fake runs ----------------------------------------

# Substrings that identify a spec-mode question in a raw prompt.
_NODE_MARKER = "JSON object with the keys summary"
_BELONGS_MARKER = "belong under this topic"

# Scores the fake judge gives chunks in order of first appearance: sun,
# planet, satellite, satellite, then repeat.  Giving every chunk the same
# level would make the smoke diagram a flat list with no planet lines, and the
# smoke run exists to exercise that serialization.
_FAKE_LEVEL_CYCLE: tuple[dict[str, int], ...] = (
    {"comprehensiveness": 90, "independence": 10, "detail": 10},  # -> sun
    {"comprehensiveness": 10, "independence": 90, "detail": 10},  # -> planet
    {"comprehensiveness": 10, "independence": 10, "detail": 90},  # -> satellite
    {"comprehensiveness": 10, "independence": 10, "detail": 90},  # -> satellite
)


def make_spec_fake_judge(max_chars: int = 120) -> FakeJudge:
    """Deterministic judge for ``--extractor spec --judge fake`` runs.

    * Q_NODE: the chunk itself is the summary, scored by _FAKE_LEVEL_CYCLE in
      order of the chunk's first appearance (remembered per text, so a repeated
      chunk always gets the same level, cached or not).  The diagram therefore
      has suns, planets and satellites.
    * Q_BELONGS: "yes" for the first candidate ever offered for a query, "no"
      afterwards, so each turn forms one sun -> planet -> satellite tree while
      the later merge probes still decline and run the full merge path.
    * Q_SAME: always "no", so the node count depends only on the chunk count.
    """
    order: dict[str, int] = {}
    belongs_seen: set[str] = set()
    lock = threading.Lock()

    def policy(prompt: str) -> str | None:
        if _NODE_MARKER in prompt:
            text = first_fenced_span(prompt).strip()
            with lock:
                index = order.setdefault(text, len(order))
            payload: dict[str, Any] = {"summary": text[:max_chars]}
            payload.update(_FAKE_LEVEL_CYCLE[index % len(_FAKE_LEVEL_CYCLE)])
            return json.dumps(payload, ensure_ascii=False)
        if _BELONGS_MARKER in prompt:
            query = first_fenced_span(prompt).strip()
            with lock:
                first = query not in belongs_seen
                belongs_seen.add(query)
            return "yes" if first else "no"
        return "no"

    return FakeJudge(policy=policy)


# -- manager -----------------------------------------------------------------


class SpecManager:
    def __init__(
        self,
        judge: JudgeLLM,
        *,
        config: SpecConfig | None = None,
        embed_fn: EmbedFn | None = None,
        cache: JudgeCache | None = None,
        node_fn: Callable[[str], tuple[str, dict[str, int]] | None] | None = None,
        keep_fn: Callable[[str], bool] | None = None,
        similarity: SimilarityJudge | None = None,
    ) -> None:
        """Optional hooks; without them the built-in behaviour is used.

        node_fn     ``text -> (summary, axis scores) | None``, used instead of
                    the Q_NODE prompt.  None means a fallback node, as for an
                    unparsable Q_NODE reply.
        keep_fn     ``text -> bool``; chunks answered False get no node and are
                    counted in ``SpecTurnReport.dropped``.
        similarity  used instead of the built-in SimilarityJudge.  Must provide
                    ``most_similar(query, candidates, kind) -> (index, score)``
                    and a ``runner`` with ``calls`` / ``unparsed`` /
                    ``defaulted`` dicts keyed by question kind, so its calls
                    are counted.
        """
        self.config = config if config is not None else SpecConfig.from_config()
        self.cache = cache if cache is not None else JudgeCache()
        self.runner = JudgeRunner(
            judge, self.cache, max_tokens=self.config.judge_max_tokens
        )
        self._node_fn = node_fn
        self._keep_fn = keep_fn
        if similarity is not None:
            self.similarity = similarity
        else:
            self.similarity = SimilarityJudge(
                judge,
                shortlist_k=self.config.shortlist_k,
                use_embedding_shortlist=self.config.shortlist_k > 0,
                embed_fn=embed_fn,
                runner=self.runner,
            )
        # Merge decisions go through counting wrappers so the turn report can
        # tell vanish (Q_SAME) events from attach (Q_BELONGS) events.
        self._merger = GraphMerger(
            similarity_fn=self._counting_similarity,
            attach_fn=self._counting_attach,
        )
        self._vanished = 0
        self._attached = 0
        # Chunks rejected by keep_fn in the last _nodes_for_chunks() call.
        self._dropped = 0
        # Run totals: call counts per question kind plus the SPEC_QUALITY_KEYS
        # counters.  Same shape as HarnessManager.totals, so build_cd_offline's
        # checkpoint code works for both.
        self.totals: dict[str, int] = {}
        self.total_cache_hits = 0
        self.normalize_calls = 0
        self.node_fallback = 0
        # node_fallback is the only counter this class updates from worker
        # threads; JudgeRunner and JudgeCache lock their own.
        self._fallback_lock = threading.Lock()

    # -- totals views -------------------------------------------------------

    def call_totals(self) -> dict[str, int]:
        return {k: v for k, v in self.totals.items() if k in ALL_KINDS}

    def quality_totals(self) -> dict[str, int]:
        return {k: self.totals.get(k, 0) for k in SPEC_QUALITY_KEYS}

    def load_totals(self, totals: dict[str, int], cache_hits: int = 0) -> None:
        """Add counters restored from a checkpoint."""
        for key, value in totals.items():
            self.totals[key] = self.totals.get(key, 0) + int(value)
        self.node_fallback += int(totals.get("node_fallback", 0))
        self.total_cache_hits += int(cache_hits)

    def _add_total(self, key: str, delta: int) -> None:
        if delta:
            self.totals[key] = self.totals.get(key, 0) + delta

    # -- counter sources ------------------------------------------------------

    def _counter_sources(self) -> list[Any]:
        """Objects whose per-kind ``calls`` / ``unparsed`` / ``defaulted`` dicts
        update() counts.

        Usually just ``self.runner``, which the built-in SimilarityJudge
        shares.  An injected ``similarity`` that brings its own ``.runner`` is
        added so its calls are counted too.
        """
        sources: list[Any] = [self.runner]
        other = getattr(self.similarity, "runner", None)
        if other is not None and other is not self.runner:
            sources.append(other)
        return sources

    def _calls_snapshot(self) -> dict[str, int]:
        snapshot: dict[str, int] = {}
        for source in self._counter_sources():
            for kind, total in source.calls.items():
                snapshot[kind] = snapshot.get(kind, 0) + total
        return snapshot

    def _counter_total(self, name: str) -> int:
        return sum(
            sum(getattr(source, name).values()) for source in self._counter_sources()
        )

    # -- merge decision counters --------------------------------------------

    def _counting_similarity(
        self, query: str, candidates: list[str]
    ) -> tuple[int, float]:
        """GraphMerger's similarity_fn (Q_SAME).  A match means the incoming
        node vanished into an existing one."""
        idx, score = self.similarity.most_similar(query, candidates)
        if score >= MATCH_SCORE:
            self._vanished += 1
        return idx, score

    def _counting_attach(self, query: str, candidates: list[str]) -> tuple[int, float]:
        """GraphMerger's attach_fn (Q_BELONGS)."""
        idx, score = self.similarity.most_similar(query, candidates, K_BELONGS)
        if score >= MATCH_SCORE:
            self._attached += 1
        return idx, score

    # -- step 1: chunking ---------------------------------------------------

    def chunk(self, user_text: str, assistant_text: str, turn: int) -> list[SpecChunk]:
        """Chunks of both sides, with no LLM involvement and no merging."""
        chunks: list[SpecChunk] = []
        for source, text in (("user", user_text), ("assistant", assistant_text)):
            if not text.strip():
                continue
            for piece in split_candidates(text, self.config.chunk_max_chars):
                chunks.append(SpecChunk(text=piece, source=source, turn=turn))
        return chunks

    # -- step 2: one node per chunk -----------------------------------------

    def node_for_text(self, text: str, turn: int = -1) -> Node:
        """The node for one chunk: one Q_NODE call plus up to ``max_retries``
        reformat retries.

        If no reply parses, the node is the chunk's first 80 characters with
        every score 0.  That fallback is counted in ``node_fallback`` and not
        cached, since it is not an answer.
        """
        # With a node_fn hook the answer cache is bypassed, so every chunk gets
        # a fresh answer from the hook.
        node_fn = self._node_fn
        use_cache = node_fn is None
        cached = self.cache.get(K_NODE, text, "") if use_cache else None
        if cached is not None:
            summary, scores = cached
            scores = dict(scores)
        else:
            parsed = node_fn(text) if node_fn is not None else self._ask_node(text)
            if parsed is None:
                with self._fallback_lock:
                    self.node_fallback += 1
                summary = text[:FALLBACK_NODE_CHARS].strip()
                scores = dict(DEFAULT_NODE_SCORES)
            else:
                summary, scores = parsed
                if use_cache:
                    self.cache.put(K_NODE, text, "", (summary, dict(scores)))
        if not summary:
            summary = text.strip()[: self.config.max_node_chars]
        level = level_from_scores(scores)
        # The mass is set later: normalize() derives every planet's mass from
        # its satellite count at the end of the turn.
        return Node(text=summary, level=level, mass=0.0, created_turn=turn)

    def _ask_node(self, text: str) -> tuple[str, dict[str, int]] | None:
        """Ask Q_NODE, retrying with a format reminder; None if no reply parses.

        Every unusable reply counts as ``unparsed`` and giving up counts as
        ``defaulted``, in the runner's counters.
        """
        base_prompt = Q_NODE.format(text=text, max_chars=self.config.max_node_chars)
        attempts = 1 + max(0, self.config.max_retries)
        for attempt in range(attempts):
            prompt = (
                base_prompt
                if attempt == 0
                else base_prompt + "\n" + RETRY_JSON_OBJECT_SUFFIX
            )
            raw = self.runner.ask_raw(
                K_NODE, prompt, max_tokens=self.config.node_max_tokens
            )
            parsed = parse_node_object(raw, self.config.max_node_chars)
            if parsed is not None:
                return parsed
            self.runner._bump(self.runner.unparsed, K_NODE)
            if attempt + 1 < attempts:
                self.runner._bump(self.runner.retried, K_NODE)
        self.runner._bump(self.runner.defaulted, K_NODE)
        return None

    def _nodes_for_chunks(self, chunks: Sequence[SpecChunk], turn: int) -> list[Node]:
        """One node per chunk, in chunk order; chunks rejected by ``keep_fn``
        get none, and their count is left in ``self._dropped``.

        With ``max_workers > 1`` the Q_NODE calls run on threads.
        ``Executor.map`` keeps input order, so the result matches a sequential
        run; two identical chunks in one turn may both miss the cache, which
        only costs an extra call.
        """
        workers = int(self.config.max_workers)
        if workers <= 1 or len(chunks) <= 1:
            results = [self._node_or_dropped(chunk, turn) for chunk in chunks]
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(chunks))) as pool:
                results = list(pool.map(lambda c: self._node_or_dropped(c, turn), chunks))
        nodes = [node for node in results if node is not None]
        self._dropped = len(results) - len(nodes)
        return nodes

    def _node_or_dropped(self, chunk: SpecChunk, turn: int) -> Node | None:
        """The chunk's node, or None when ``keep_fn`` rejects the chunk."""
        if self._keep_fn is not None and not self._keep_fn(chunk.text):
            return None
        return self.node_for_text(chunk.text, turn)

    # -- step 3: provisional structure --------------------------------------

    def build_provisional(self, nodes: Sequence[Node]) -> ProvisionalStructure:
        """Link this turn's nodes to each other only, via Q_BELONGS.

        Planets attach to this turn's suns, satellites to this turn's planets;
        whatever does not attach is an orphan for the merge step.
        """
        return link_within_turn(nodes, self.similarity.most_similar)

    # -- step 4: merge ------------------------------------------------------

    def _merge(self, base: CorrelationDiagram, structure: ProvisionalStructure) -> int:
        """Merge into ``base``; returns the number of orphan promotions.

        GraphMerger.merge() normalizes with the diagram's default mass floor;
        update() normalizes again at the end with the spec's floor, so the
        spec's floor is what remains.
        """
        return merge_into_base(self._merger, base, structure)

    # -- public entry point -------------------------------------------------

    def update(
        self,
        base: CorrelationDiagram,
        user_text: str,
        assistant_text: str,
        turn: int,
    ) -> SpecTurnReport:
        """Process one user/assistant round trip, mutating ``base``.

        Normalize and accounting run in ``finally``, so a judge error mid-turn
        still leaves a consistent diagram and correct counts; the error then
        propagates to the caller.
        """
        calls_before = self._calls_snapshot()
        unparsed_before = self._counter_total("unparsed")
        defaulted_before = self._counter_total("defaulted")
        fallback_before = self.node_fallback
        hits_before = self.cache.hits
        nodes_before = len(base)
        self._vanished = 0
        self._attached = 0
        self._dropped = 0
        report = SpecTurnReport()

        try:
            chunks = self.chunk(user_text, assistant_text, turn)
            report.chunks = len(chunks)

            nodes = self._nodes_for_chunks(chunks, turn)
            report.nodes = len(nodes)

            structure = self.build_provisional(nodes)
            report.promoted = self._merge(base, structure)
        finally:
            # Exactly one normalize per update, with the spec's mass floor.
            base.normalize(planet_mass_floor=self.config.planet_mass_floor)
            self.normalize_calls += 1

            calls: dict[str, int] = {}
            for kind, total in self._calls_snapshot().items():
                delta = total - calls_before.get(kind, 0)
                if delta > 0:
                    calls[kind] = delta
                    self._add_total(kind, delta)
            report.calls = calls
            report.cache_hits = self.cache.hits - hits_before
            self.total_cache_hits += report.cache_hits
            report.added = len(base) - nodes_before
            report.vanished = self._vanished
            report.attached = self._attached
            report.dropped = self._dropped
            report.unparsed = self._counter_total("unparsed") - unparsed_before
            report.defaulted = self._counter_total("defaulted") - defaulted_before
            report.node_fallback = self.node_fallback - fallback_before
            for key in SPEC_QUALITY_KEYS:
                self._add_total(key, getattr(report, key))
            # Summed in totals but not a SPEC_QUALITY_KEYS member, so
            # quality_totals() keeps its shape.
            self._add_total("dropped", report.dropped)

        return report
