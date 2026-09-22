"""SpecManager: the specification-faithful manager procedure (0030, 0037-0062).

Design: SPEC_FAITHFUL_DESIGN.md Stream S1.  This is the DEFAULT experiment path;
``manager.HarnessManager`` stays as the optional enhanced mode.

The difference from HarnessManager is what is NOT here.  Ryosuke's directive 2
(2026-09-07) is **1 chunk = 1 node**, so this module has:

  * no statement extraction (Q_EXTRACT),
  * no faithfulness check (Q_SUPPORTED),
  * no within-turn de-duplication,
  * no LLM boundary merge (Q_BOUNDARY),

and the evaluation unit stays off (directive 1).

Per turn:
  1. chunk        chunking.split_candidates on each side, no merging  0037-0038
  2. one node     ONE Q_NODE call per chunk -> summary + 3 axis scores 0030/0039-0040
  3. link         this turn's nodes to each other only                0041
  4. merge        GraphMerger case 1 / 2 / 3 into the base diagram    0042-0061
  5. normalize    exactly once, in a `finally`, with the mass floor   0030/0062

Step 0 (D-7 query-turn skip) is the CALLER's job, exactly as for HarnessManager.

Machine rule: nothing here loads a model.  The judge is injected; the driver's
`--judge fake` path uses :func:`make_spec_fake_judge`.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import jsonschema

from management.graph_merger import GraphMerger
from management.harness.backends import FakeJudge, first_fenced_span
from management.harness.chunking import split_candidates
from management.harness.judge import JudgeCache, JudgeLLM, JudgeRunner
from management.harness.manager import ProvisionalStructure
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
from models.correlation_diagram import CorrelationDiagram, PlanetEntry
from models.node import Node, NodeLevel
from utils.config import get

# The three 0039 axes, in the order they are documented and reported.
AXES: tuple[str, ...] = ("comprehensiveness", "independence", "detail")

# Score range of 0039.  Values outside it are CLAMPED rather than rejected:
# a model answering 120 has still ranked the axis, and failing the whole reply
# would cost a retry and possibly a fallback node.  The SHAPE (four keys, a
# non-empty string summary, integer scores) is enforced by jsonschema.
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

# Quality counter keys, reported per turn and accumulated in `totals`.
SPEC_QUALITY_KEYS: tuple[str, ...] = (
    "unparsed",
    "defaulted",
    "node_fallback",
    "vanished",
    "attached",
)


@dataclass
class SpecChunk:
    """One meaning unit of 0037-0038.

    Same three fields as ``management.text_chunker.Chunk``; declared here so the
    spec path never imports the legacy Japanese-only ``TextChunker`` module.
    """

    text: str
    source: str  # "user" | "assistant"
    turn: int


# -- config ------------------------------------------------------------------


@dataclass
class SpecConfig:
    """Mirrors the ``spec_manager:`` section of config.yaml (S1.1)."""

    chunk_max_chars: int = 400
    max_node_chars: int = 120
    shortlist_k: int = 5
    judge_max_tokens: int = 256
    node_max_tokens: int = 200
    max_retries: int = 1
    # 0062 literally: a planet with no satellites has mass 0.  The diagram's own
    # default (floor 1.0) is kept for every other caller.
    planet_mass_floor: float = 0.0
    # B5 (2026-09-07): the Q_NODE calls of ONE turn are independent of each
    # other (1 chunk = 1 node, no cross-chunk state), so they may run on a
    # ThreadPoolExecutor.  Linking and merging stay strictly sequential -- they
    # mutate the diagram and their order decides the result.  1 = sequential
    # (the default, and what every existing test asserts).
    max_workers: int = 1

    @classmethod
    def from_config(cls) -> "SpecConfig":
        d = cls()
        return cls(
            chunk_max_chars=int(
                get("spec_manager", "chunk_max_chars", d.chunk_max_chars)
            ),
            max_node_chars=int(get("spec_manager", "max_node_chars", d.max_node_chars)),
            shortlist_k=int(get("spec_manager", "shortlist_k", d.shortlist_k)),
            judge_max_tokens=int(
                get("spec_manager", "judge_max_tokens", d.judge_max_tokens)
            ),
            node_max_tokens=int(get("spec_manager", "node_max_tokens", d.node_max_tokens)),
            max_retries=int(get("spec_manager", "max_retries", d.max_retries)),
            planet_mass_floor=float(
                get("spec_manager", "planet_mass_floor", d.planet_mass_floor)
            ),
            max_workers=int(get("spec_manager", "max_workers", d.max_workers)),
        )


# -- report ------------------------------------------------------------------


@dataclass
class SpecTurnReport:
    """Per-turn statistics (S1.2 step 6)."""

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
    # Chunks removed by the optional ``keep_fn`` hook BEFORE any node was made
    # for them (mcbuild_bench D3 `keep`: code / tool output / logs are not
    # memory items).  0 whenever no keep_fn is installed.
    dropped: int = 0

    def total_calls(self) -> int:
        return sum(self.calls.values())


# -- parsing helpers ---------------------------------------------------------


def clamp_score(value: int) -> int:
    return max(SCORE_MIN, min(SCORE_MAX, int(value)))


def first_json_object(raw: str) -> str | None:
    """The FIRST balanced ``{...}`` span of ``raw``, or None.

    M8 (2026-09-07 review): the old rule was FIRST ``{`` .. LAST ``}``.  A reply
    that contains two objects ("here is the object: {...}  and an example
    {...}") produced the concatenation of both, which never parses, so a
    perfectly good first object was thrown away and the chunk fell back.  Braces
    inside JSON strings (and their backslash escapes) are respected, so a summary
    containing ``{`` no longer breaks the scan.
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
    """M8: accept 90, "90", 90.0 and " 90 "; reject anything else.

    Models answer the three axes as strings or as floats often enough that
    rejecting those shapes costs a retry and then a fallback node for a reply
    that carried the right ranking all along.
    """
    if isinstance(value, bool):  # bool is an int subclass; not a score
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
    """M8: lift a nested ``{"summary": ..., "scores": {...}}`` to a flat object.

    Only the three axis keys are lifted, and only when they are ABSENT at the
    top level, so a model that answers both shapes keeps its top-level values.
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
    """Parse a Q_NODE reply into ``(summary, scores)`` or None.

    The object is the FIRST BALANCED ``{...}`` span (M8; see
    :func:`first_json_object`), optionally with a nested ``scores`` block lifted
    into the top level, with the three axis values coerced from string/float
    shapes.  The result is validated against :data:`NODE_OBJECT_SCHEMA`, the
    summary is trimmed to ``max_chars`` and the scores are clamped into 0..100.
    """
    span = first_json_object(raw)
    if span is None:
        return None
    try:
        parsed = json.loads(span)
    except Exception:
        # Not silent: None makes the caller count `unparsed`, retry, and finally
        # count `defaulted` + `node_fallback`.
        return None
    parsed = _lift_nested_scores(parsed)
    if isinstance(parsed, dict):
        coerced = dict(parsed)
        for axis in AXES:
            if axis in coerced:
                value = _coerce_score(coerced[axis])
                if value is None:
                    return None  # e.g. "hi": not a score in any shape
                coerced[axis] = value
        parsed = coerced
    try:
        jsonschema.validate(parsed, NODE_OBJECT_SCHEMA)
    except Exception:
        return None  # same accounting as above
    summary = str(parsed["summary"]).strip()[:max_chars].strip()
    if not summary:
        return None
    scores = {axis: clamp_score(parsed[axis]) for axis in AXES}
    return summary, scores


def level_from_scores(scores: dict[str, int]) -> NodeLevel:
    """0040 argmax with the documented tie rule detail > independence >
    comprehensiveness.

    Specific beats general: the oracle CD is 1 sun / 32 planets / 207 satellites,
    so an ambiguous node belongs at the bottom.  All-zero also yields satellite.

        detail   indep    compr    level
        top      -        -        satellite   (detail wins outright and on ties)
        <        top      -        planet
        <        <        top      sun
        0        0        0        satellite   (0040 fallback)
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


# -- the driver's fake judge (S1.3) ------------------------------------------

# Distinctive substrings used to recognise a spec-mode question in a raw prompt.
_NODE_MARKER = "JSON object with the keys summary"
_BELONGS_MARKER = "belong under this topic"

# M2 (2026-09-07 review): the fake judge's score cycle.  Chunk 0 of the run is a
# sun, chunk 1 a planet, chunks 2 and 3 satellites, then the cycle repeats.  The
# old policy scored EVERY chunk detail=100, so the smoke diagram was a flat list
# of promoted suns with no ``[PN`` line at all -- and the whole point of the
# smoke run is to exercise the marker serialization the reader then scans.
_FAKE_LEVEL_CYCLE: tuple[dict[str, int], ...] = (
    {"comprehensiveness": 90, "independence": 10, "detail": 10},  # -> sun
    {"comprehensiveness": 10, "independence": 90, "detail": 10},  # -> planet
    {"comprehensiveness": 10, "independence": 10, "detail": 90},  # -> satellite
    {"comprehensiveness": 10, "independence": 10, "detail": 90},  # -> satellite
)


def make_spec_fake_judge(max_chars: int = 120) -> FakeJudge:
    """The ``--extractor spec --judge fake`` policy (S1.3 + M2).

    Deterministic, no network, no model:

    * **Q_NODE** returns the chunk itself as the summary and the scores of
      :data:`_FAKE_LEVEL_CYCLE` indexed by the chunk's ORDER OF FIRST APPEARANCE
      in the run (a per-text memo, so an identical chunk always gets the same
      level whether or not the answer cache served it).  The run therefore
      produces suns, planets AND satellites, and the smoke correlation diagram
      really carries ``[SN]`` / ``[PN{mass}]`` / ``[RN]`` lines.
    * **Q_BELONGS** answers "yes" for the FIRST candidate ever offered for a
      given query and "no" afterwards, so a planet attaches to the turn's first
      sun and a satellite to the turn's first planet -- one three-level tree per
      turn -- while later re-asks (the case 2 / case 3 merge probes) decline and
      leave the merge path exercised rather than short-circuited.
    * **Q_SAME** always answers "no": nothing vanishes, so the node count is a
      function of the chunk count only.
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
        """Optional hooks (mcbuild_bench DESIGN.md 5.1), each defaulting to the
        behaviour documented above:

        node_fn     ``text -> (summary, axis scores) | None``; replaces the
                    Q_NODE text prompt of :meth:`_ask_node`.  A ``None`` return
                    is a node fallback exactly like an unparsable Q_NODE reply.
        keep_fn     ``text -> bool``; chunks answered ``False`` are dropped
                    before node creation and counted in ``SpecTurnReport.dropped``.
        similarity  replaces the internally built :class:`SimilarityJudge`.  It
                    must provide ``most_similar(query, candidates, kind) ->
                    (index, score)`` and, for the per-turn accounting, a
                    ``runner`` attribute carrying ``calls`` / ``unparsed`` /
                    ``defaulted`` dicts keyed by question kind (see
                    :meth:`_counter_sources`).
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
        # Every merge decision goes through the counting wrappers so the report
        # can attribute vanish (Q_SAME) and attach (Q_BELONGS) events.
        self._merger = GraphMerger(
            similarity_fn=self._counting_similarity,
            attach_fn=self._counting_attach,
        )
        self._vanished = 0
        self._attached = 0
        # Per-call drop count of the last _nodes_for_chunks() (keep_fn hook).
        self._dropped = 0
        # Cumulative totals over the whole run: call counts by question kind PLUS
        # the SPEC_QUALITY_KEYS counters.  Same shape as HarnessManager.totals so
        # build_cd_offline's checkpoint block is unchanged.
        self.totals: dict[str, int] = {}
        self.total_cache_hits = 0
        self.normalize_calls = 0
        self.node_fallback = 0
        # B5: node_fallback is incremented from worker threads when
        # config.max_workers > 1.  JudgeRunner's counters and JudgeCache are
        # already lock-protected (judge.py: JudgeRunner._lock, JudgeCache._lock);
        # this is the only manager-owned counter on the parallel path.
        self._fallback_lock = threading.Lock()

    # -- totals views -------------------------------------------------------

    def call_totals(self) -> dict[str, int]:
        return {k: v for k, v in self.totals.items() if k in ALL_KINDS}

    def quality_totals(self) -> dict[str, int]:
        return {k: self.totals.get(k, 0) for k in SPEC_QUALITY_KEYS}

    def load_totals(self, totals: dict[str, int], cache_hits: int = 0) -> None:
        """Restore cumulative counters from a checkpoint (H12)."""
        for key, value in totals.items():
            self.totals[key] = self.totals.get(key, 0) + int(value)
        self.node_fallback += int(totals.get("node_fallback", 0))
        self.total_cache_hits += int(cache_hits)

    def _add_total(self, key: str, delta: int) -> None:
        if delta:
            self.totals[key] = self.totals.get(key, 0) + delta

    # -- counter sources ------------------------------------------------------

    def _counter_sources(self) -> list[Any]:
        """Runner-like objects whose per-kind counters update() accounts.

        The internal SimilarityJudge shares ``self.runner`` (one ``calls`` dict
        for the whole turn).  An injected ``similarity`` may bring its own
        counter object as ``.runner``; it is read here so ``call_totals()`` and
        ``quality_totals()`` still cover every same/belongs decision.  Required
        fields on such an object: ``calls``, ``unparsed``, ``defaulted`` as
        ``dict[str, int]``.
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
        """GraphMerger's similarity_fn: Q_SAME.  A "yes" means the incoming node
        VANISHED into an existing one (0045/0046/0048/0056/0059)."""
        idx, score = self.similarity.most_similar(query, candidates)
        if score >= MATCH_SCORE:
            self._vanished += 1
        return idx, score

    def _counting_attach(self, query: str, candidates: list[str]) -> tuple[int, float]:
        """GraphMerger's attach_fn: Q_BELONGS, for the attach decisions of
        0057 / 0060 / 0061."""
        idx, score = self.similarity.most_similar(query, candidates, K_BELONGS)
        if score >= MATCH_SCORE:
            self._attached += 1
        return idx, score

    # -- step 1: chunking (0037-0038) ---------------------------------------

    def chunk(self, user_text: str, assistant_text: str, turn: int) -> list[SpecChunk]:
        """Meaning units of both sides.  No LLM involvement, no merging."""
        chunks: list[SpecChunk] = []
        for source, text in (("user", user_text), ("assistant", assistant_text)):
            if not text.strip():
                continue
            for piece in split_candidates(text, self.config.chunk_max_chars):
                chunks.append(SpecChunk(text=piece, source=source, turn=turn))
        return chunks

    # -- step 2: one node per chunk (0030, 0039-0040) ------------------------

    def node_for_text(self, text: str, turn: int = -1) -> Node:
        """ONE Q_NODE call (plus at most ``max_retries`` reformat retries).

        On total failure the node text is ``text[:80]`` with all scores 0
        (counted as ``node_fallback``, and NOT cached: a fallback is the absence
        of an answer, exactly like JudgeRunner's defaulted yes/no).
        """
        # mcbuild_bench Astra round 3 item 3: with a node_fn hook the K_NODE
        # answer cache is bypassed entirely (fresh Jev answers must be used);
        # the default text-prompt path keeps caching.
        cached = None if self._node_fn is not None else self.cache.get(K_NODE, text, "")
        if cached is not None:
            summary, scores = cached
            scores = dict(scores)
        else:
            parsed = (
                self._node_fn(text) if self._node_fn is not None else self._ask_node(text)
            )
            if parsed is None:
                with self._fallback_lock:
                    self.node_fallback += 1
                summary = text[:FALLBACK_NODE_CHARS].strip()
                scores = dict(DEFAULT_NODE_SCORES)
            else:
                summary, scores = parsed
                if self._node_fn is None:
                    self.cache.put(K_NODE, text, "", (summary, dict(scores)))
        if not summary:
            summary = text.strip()[: self.config.max_node_chars]
        level = level_from_scores(scores)
        # Mass at creation is irrelevant: normalize() overwrites every planet
        # mass from the satellite count (0062) at the end of the turn.
        return Node(text=summary, level=level, mass=0.0, created_turn=turn)

    def _ask_node(self, text: str) -> tuple[str, dict[str, int]] | None:
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
            # Same accounting as JudgeRunner.ask_yes_no: the first unusable reply
            # is `unparsed`, and never getting one is `defaulted`.
            self.runner._bump(self.runner.unparsed, K_NODE)
            if attempt + 1 < attempts:
                self.runner._bump(self.runner.retried, K_NODE)
        self.runner._bump(self.runner.defaulted, K_NODE)
        return None

    def _nodes_for_chunks(self, chunks: Sequence[SpecChunk], turn: int) -> list[Node]:
        """One node per chunk, in chunk order.

        B5: with ``config.max_workers > 1`` the Q_NODE calls run on a
        ThreadPoolExecutor.  ``Executor.map`` yields in INPUT order, so the node
        list -- and therefore the linking and merging that follow -- is
        identical to the sequential run.  The only observable difference is that
        two IDENTICAL chunks inside one turn can both miss the answer cache and
        ask twice (a cost difference, not a result difference).

        With a ``keep_fn`` hook, a chunk answered ``False`` yields no node; the
        number of such chunks of THIS call is left in ``self._dropped`` for the
        turn report (same pattern as ``_vanished`` / ``_attached``).
        """
        workers = max(1, int(self.config.max_workers))
        if workers <= 1 or len(chunks) <= 1:
            results = [self._node_or_dropped(chunk, turn) for chunk in chunks]
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(chunks))) as pool:
                results = list(pool.map(lambda c: self._node_or_dropped(c, turn), chunks))
        nodes = [node for node in results if node is not None]
        self._dropped = len(results) - len(nodes)
        return nodes

    def _node_or_dropped(self, chunk: SpecChunk, turn: int) -> Node | None:
        """keep_fn gate (None = dropped), then the ONE node of the chunk."""
        if self._keep_fn is not None and not self._keep_fn(chunk.text):
            return None
        return self.node_for_text(chunk.text, turn)

    # -- step 3: provisional structure (0041) --------------------------------

    def build_provisional(self, nodes: Sequence[Node]) -> ProvisionalStructure:
        """Link this turn's nodes to EACH OTHER only, via Q_BELONGS.

        Planets attach to this turn's suns, satellites to this turn's planets;
        whatever does not attach is an orphan handed to case 2 / case 3.
        """
        cd = CorrelationDiagram()
        sun_texts: list[str] = []
        sun_ids: list[str] = []
        for node in nodes:
            if node.level is not NodeLevel.SUN:
                continue
            if cd.add_sun(node):
                sun_texts.append(node.text)
                sun_ids.append(node.node_id)

        orphan_planets: list[PlanetEntry] = []
        planet_texts: list[str] = []
        # Either the id of a planet already inside cd, or an orphan PlanetEntry.
        planet_refs: list[str | PlanetEntry] = []
        for node in nodes:
            if node.level is not NodeLevel.PLANET:
                continue
            attached = False
            if sun_texts:
                idx, score = self.similarity.most_similar(
                    node.text, sun_texts, K_BELONGS
                )
                if score >= MATCH_SCORE and cd.add_planet(node, sun_ids[idx]):
                    planet_refs.append(node.node_id)
                    attached = True
            if not attached:
                node.level = NodeLevel.PLANET
                node.parent_id = None
                entry = PlanetEntry(planet=node)
                orphan_planets.append(entry)
                planet_refs.append(entry)
            planet_texts.append(node.text)

        orphan_satellites: list[Node] = []
        for node in nodes:
            if node.level is not NodeLevel.SATELLITE:
                continue
            attached = False
            if planet_texts:
                idx, score = self.similarity.most_similar(
                    node.text, planet_texts, K_BELONGS
                )
                if score >= MATCH_SCORE:
                    ref = planet_refs[idx]
                    if isinstance(ref, str):
                        attached = cd.add_satellite(node, ref)
                    else:
                        node.level = NodeLevel.SATELLITE
                        node.parent_id = ref.planet.node_id
                        ref.satellites.append(node)
                        attached = True
            if not attached:
                orphan_satellites.append(node)

        return ProvisionalStructure(
            suns=cd.suns,
            orphan_planets=orphan_planets,
            orphan_satellites=orphan_satellites,
        )

    # -- step 4: merge (0042-0061) -------------------------------------------

    def _merge(self, base: CorrelationDiagram, structure: ProvisionalStructure) -> int:
        """Returns the number of orphan promotions observed (0058 / 0061)."""
        promoted = 0
        if structure.suns:
            incoming = CorrelationDiagram()
            incoming.suns = structure.suns
            # NOTE: GraphMerger.merge() normalizes internally with the diagram's
            # OWN default floor; update()'s single normalize in the `finally`
            # then re-runs it with the spec floor, so the end state is the spec's.
            self._merger.merge(base, incoming)

        for entry in structure.orphan_planets:
            before_suns = len(base.suns)
            self._merger.merge_case2_planet(base, entry.planet, entry.satellites)
            if len(base.suns) > before_suns:
                promoted += 1  # 0058: planet promoted to sun

        for sat in structure.orphan_satellites:
            before_suns = len(base.suns)
            before_planets = sum(len(se.planets) for se in base.suns)
            self._merger.merge_case3_satellite(base, sat)
            after_planets = sum(len(se.planets) for se in base.suns)
            if len(base.suns) > before_suns or after_planets > before_planets:
                promoted += 1  # 0061: satellite promoted to planet or sun
        return promoted

    # -- public entry point --------------------------------------------------

    def update(
        self,
        base: CorrelationDiagram,
        user_text: str,
        assistant_text: str,
        turn: int,
    ) -> SpecTurnReport:
        """One conversation round-trip (0036).  Mutates ``base`` in place.

        The normalize and the accounting run in a `finally`, so a judge that
        raises mid-turn still leaves the diagram consistent and still costs what
        it cost.  The exception propagates to the caller (the driver counts it).
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

            # 1 chunk = 1 node (directive 2).
            nodes = self._nodes_for_chunks(chunks, turn)
            report.nodes = len(nodes)

            structure = self.build_provisional(nodes)
            report.promoted = self._merge(base, structure)
        finally:
            # 0030 + 0062: exactly one manager-level normalize per update.
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
            # Not a SPEC_QUALITY_KEYS member: quality_totals() keeps its shape.
            self._add_total("dropped", report.dropped)

        return report
