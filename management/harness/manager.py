"""HarnessManager: the manager procedure written as code, with an LLM that only
answers small, checkable questions.  (SpecManager in ``spec_manager.py`` is the
default path; this is the optional mode with extra checks.)

Per turn:
  1. chunk: split_candidates, then merge neighbours the judge says share a
     topic (Q_BOUNDARY)
  2. extract self-contained statements from each chunk (JSON array)
  3. drop statements the chunk does not support (Q_SUPPORTED)
  4. classify each statement on three yes/no axes -> sun / planet / satellite
  5. drop this turn's duplicate statements
  6. link this turn's nodes to each other only
  7. merge that structure into the base diagram (GraphMerger)
  8. normalize once (planet mass = satellite count, coordinates)

Skipping turns that only ask a question is the caller's job.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Iterable, Sequence

import jsonschema

from management.graph_merger import GraphMerger
from management.harness.chunking import split_candidates
from management.harness.judge import JudgeCache, JudgeLLM, JudgeRunner
from management.harness.prompts import (
    ALL_KINDS,
    K_BELONGS,
    K_BOUNDARY,
    K_COMPREHENSIVE,
    K_DETAIL,
    K_EXTRACT,
    K_INDEPENDENT,
    K_SAME,
    K_SUPPORTED,
    NO_TOPICS,
    Q_BOUNDARY,
    Q_COMPREHENSIVE,
    Q_DETAIL,
    Q_EXTRACT,
    Q_INDEPENDENT,
    Q_SUPPORTED,
    RETRY_JSON_SUFFIX,
)
from management.harness.similarity_judge import MATCH_SCORE, EmbedFn, SimilarityJudge
from management.node_classifier import Action, NodeProposal
from management.text_chunker import Chunk
from models.correlation_diagram import CorrelationDiagram, PlanetEntry, SunEntry
from models.node import Node, NodeLevel
from utils.config import get

# The extraction reply must be a JSON array of non-empty strings.
STATEMENT_ARRAY_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {"type": "string", "minLength": 1},
}

# Length of the fallback statement when extraction never returns valid JSON
# (the same 80 characters build_cd_offline uses).
FALLBACK_STATEMENT_CHARS = 80

# Answer-quality counters, reported per turn and summed in ``totals``.
QUALITY_KEYS: tuple[str, ...] = (
    "unparsed",
    "defaulted",
    "extract_fallback",
    "extract_salvaged",
    "dedup_dropped",
    "vanished",
    "attached",
)

_LEVEL_ACTION = {
    NodeLevel.SUN: Action.NEW_SUN,
    NodeLevel.PLANET: Action.NEW_PLANET,
    NodeLevel.SATELLITE: Action.NEW_SATELLITE,
}

# Every complete double-quoted JSON string (escapes included).
_QUOTED_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')

# ``most_similar(query, candidates, kind) -> (index, score)``.
MostSimilarFn = Callable[[str, list[str], str], tuple[int, float]]


def salvage_string_array(raw: str) -> list[str]:
    """Recover the complete strings of a JSON array cut off by the token limit.

    ``["a", "b", "c`` -> ``["a", "b"]``.  Returns [] when there is no '[' or
    nothing complete.
    """
    start = raw.find("[")
    if start == -1:
        return []
    out: list[str] = []
    for m in _QUOTED_RE.finditer(raw[start:]):
        try:
            value = json.loads('"' + m.group(1) + '"')
        except ValueError:
            continue
        if isinstance(value, str) and value.strip():
            out.append(value)
    return out


def normalize_for_dedup(text: str) -> str:
    """Duplicate-detection key: NFKC, casefolded, with punctuation, whitespace
    and control characters removed."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        ch for ch in folded if not unicodedata.category(ch).startswith(("P", "Z", "C"))
    )


# -- config ------------------------------------------------------------------


@dataclass
class HarnessConfig:
    """The ``harness:`` section of config.yaml."""

    boundary_check: bool = True
    faithfulness_check: bool = True
    shortlist_k: int = 5
    max_retries: int = 2
    max_statement_chars: int = 120
    judge_max_tokens: int = 256
    # A JSON array of statements needs far more room than a one-word answer;
    # 256 tokens cut multi-statement replies short.
    extract_max_tokens: int = 1024
    # Max chunk length, also the cap for the Q_BOUNDARY merge.
    chunk_max_chars: int = 800
    # How many existing sun texts the classification questions are shown.
    topics_in_prompt: int = 8
    # Threads for the per-statement questions; 1 = sequential.
    max_workers: int = 1

    @classmethod
    def from_config(cls) -> "HarnessConfig":
        """Read config.yaml; each value is cast to the type of its default."""
        defaults = cls()
        values: dict[str, Any] = {}
        for f in fields(cls):
            default = getattr(defaults, f.name)
            values[f.name] = type(default)(get("harness", f.name, default))
        return cls(**values)


# -- report / provisional structure ------------------------------------------


@dataclass
class HarnessTurnReport:
    """Per-turn statistics."""

    calls: dict[str, int] = field(default_factory=dict)
    cache_hits: int = 0
    statements: int = 0
    dropped_unsupported: int = 0
    added: int = 0
    merged: int = 0
    promoted: int = 0
    chunks: int = 0
    # ``merged`` = vanished + attached; kept for older readers.
    vanished: int = 0
    attached: int = 0
    # Answer-quality counters (QUALITY_KEYS).
    unparsed: int = 0
    defaulted: int = 0
    extract_fallback: int = 0
    extract_salvaged: int = 0
    dedup_dropped: int = 0

    def total_calls(self) -> int:
        return sum(self.calls.values())


@dataclass
class ProvisionalStructure:
    """One turn's nodes, linked only to each other: suns with whatever attached
    to them, plus the planets and satellites that found no parent."""

    suns: list[SunEntry] = field(default_factory=list)
    orphan_planets: list[PlanetEntry] = field(default_factory=list)
    orphan_satellites: list[Node] = field(default_factory=list)


def link_within_turn(
    nodes: Sequence[Node], most_similar: MostSimilarFn
) -> ProvisionalStructure:
    """Link one turn's nodes to each other only, via Q_BELONGS.

    Planets attach to this turn's suns and satellites to this turn's planets;
    whatever finds no parent is returned as an orphan for the merge step.
    """
    cd = CorrelationDiagram()
    sun_texts: list[str] = []
    sun_ids: list[str] = []
    for node in nodes:
        if node.level is NodeLevel.SUN and cd.add_sun(node):
            sun_texts.append(node.text)
            sun_ids.append(node.node_id)

    orphan_planets: list[PlanetEntry] = []
    planet_texts: list[str] = []
    # Per planet: its id when it joined a sun in ``cd``, else its orphan entry.
    planet_refs: list[str | PlanetEntry] = []
    for node in nodes:
        if node.level is not NodeLevel.PLANET:
            continue
        attached = False
        if sun_texts:
            idx, score = most_similar(node.text, sun_texts, K_BELONGS)
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
            idx, score = most_similar(node.text, planet_texts, K_BELONGS)
            if score >= MATCH_SCORE:
                ref = planet_refs[idx]
                if isinstance(ref, str):
                    attached = cd.add_satellite(node, ref)
                else:  # the planet is itself an orphan: hang it on its entry
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


def merge_into_base(
    merger: GraphMerger, base: CorrelationDiagram, structure: ProvisionalStructure
) -> int:
    """Merge one turn's structure into ``base``; return how many orphans were
    promoted (a planet to a sun, or a satellite to a planet or sun)."""
    if structure.suns:
        incoming = CorrelationDiagram()
        incoming.suns = structure.suns
        # merge() also normalizes; the managers still normalize once at the end.
        merger.merge(base, incoming)

    promoted = 0
    for entry in structure.orphan_planets:
        suns_before = len(base.suns)
        merger.merge_case2_planet(base, entry.planet, entry.satellites)
        if len(base.suns) > suns_before:
            promoted += 1

    for sat in structure.orphan_satellites:
        suns_before = len(base.suns)
        planets_before = sum(len(se.planets) for se in base.suns)
        merger.merge_case3_satellite(base, sat)
        planets_after = sum(len(se.planets) for se in base.suns)
        if len(base.suns) > suns_before or planets_after > planets_before:
            promoted += 1
    return promoted


# -- manager -----------------------------------------------------------------


class HarnessManager:
    def __init__(
        self,
        judge: JudgeLLM,
        *,
        config: HarnessConfig | None = None,
        embed_fn: EmbedFn | None = None,
        cache: JudgeCache | None = None,
    ) -> None:
        self.config = config if config is not None else HarnessConfig.from_config()
        self.cache = cache if cache is not None else JudgeCache()
        self.runner = JudgeRunner(
            judge, self.cache, max_tokens=self.config.judge_max_tokens
        )
        self.similarity = SimilarityJudge(
            judge,
            shortlist_k=self.config.shortlist_k,
            use_embedding_shortlist=self.config.shortlist_k > 0,
            embed_fn=embed_fn,
            runner=self.runner,
        )
        # Merge decisions go through counting wrappers so the turn report can
        # tell vanish events from attach events.
        self._merger = GraphMerger(
            similarity_fn=self._counting_similarity,
            attach_fn=self._counting_attach,
        )
        self._vanished = 0
        self._attached = 0
        self._sun_mass = float(get("graph", "default_sun_mass", 1.0))
        self._planet_mass = float(get("graph", "default_planet_mass", 0.5))
        self._satellite_mass = float(get("graph", "default_satellite_mass", 0.1))
        # Run totals (read by build_cd_offline): call counts per question kind
        # plus the QUALITY_KEYS counters.
        self.totals: dict[str, int] = {}
        self.total_cache_hits = 0
        self.normalize_calls = 0
        self.extract_fallback = 0
        self.extract_salvaged = 0
        self.dedup_dropped = 0

    # -- totals views -------------------------------------------------------

    def call_totals(self) -> dict[str, int]:
        """Call counts per question kind (what ``harness_calls`` reports)."""
        return {k: v for k, v in self.totals.items() if k in ALL_KINDS}

    def quality_totals(self) -> dict[str, int]:
        """The answer-quality counters of ``totals``."""
        return {k: self.totals.get(k, 0) for k in QUALITY_KEYS}

    def load_totals(self, totals: dict[str, int]) -> None:
        """Add counters restored from a checkpoint."""
        for key, value in totals.items():
            self.totals[key] = self.totals.get(key, 0) + int(value)
        self.extract_fallback += int(totals.get("extract_fallback", 0))
        self.extract_salvaged += int(totals.get("extract_salvaged", 0))
        self.dedup_dropped += int(totals.get("dedup_dropped", 0))

    def _add_total(self, key: str, delta: int) -> None:
        if delta:
            self.totals[key] = self.totals.get(key, 0) + delta

    # -- LLM plumbing -------------------------------------------------------

    def _ask(self, kind: str, prompt: str, *, max_tokens: int | None = None) -> str:
        return self.runner.ask_raw(kind, prompt, max_tokens=max_tokens)

    def _ask_bool(self, kind: str, prompt: str, key_a: str, key_b: str = "") -> bool:
        return self.runner.ask_yes_no(kind, prompt, key_a, key_b)

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
        """GraphMerger's attach_fn (Q_BELONGS: "does this belong under that
        topic?").  Asking Q_SAME here instead made every orphan fail to attach
        and get promoted, inflating the sun count."""
        idx, score = self.similarity.most_similar(query, candidates, K_BELONGS)
        if score >= MATCH_SCORE:
            self._attached += 1
        return idx, score

    # -- step 1: chunking ---------------------------------------------------

    def _chunk(self, user_text: str, assistant_text: str, turn: int) -> list[Chunk]:
        """Split both sides into candidates, then merge neighbours the judge
        says stay on the same topic."""
        max_chars = self.config.chunk_max_chars
        chunks: list[Chunk] = []
        for source, text in (("user", user_text), ("assistant", assistant_text)):
            if not text.strip():
                continue
            for piece in split_candidates(text, max_chars):
                chunks.append(Chunk(text=piece, source=source, turn=turn))

        if not self.config.boundary_check or len(chunks) < 2:
            return chunks

        merged: list[Chunk] = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            # The question compares the original neighbours, not the growing
            # merged chunk, so its answer stays cacheable.
            if prev.source == cur.source and prev.turn == cur.turn:
                joined = merged[-1].text + " " + cur.text
                # Without this cap, a chain of merges rebuilds the whole turn
                # as one chunk.
                if len(joined) <= max_chars:
                    prompt = Q_BOUNDARY.format(a=prev.text, b=cur.text)
                    changes = self._ask_bool(K_BOUNDARY, prompt, prev.text, cur.text)
                    if not changes:
                        merged[-1] = Chunk(
                            text=joined,
                            source=merged[-1].source,
                            turn=merged[-1].turn,
                        )
                        continue
            merged.append(cur)
        return merged

    # -- step 2: statement extraction ---------------------------------------

    @staticmethod
    def _normalize_payload(parsed: Any) -> list[str] | None:
        """Turn the reply shapes models actually produce into a list of strings.

        Accepts ``["a"]``, ``{"statements": [...]}`` (or facts / items /
        results), ``[{"statement": "a"}]`` and ``[{"text": "a"}]``.  Blank
        elements are dropped rather than failing the reply; None for any other
        shape.
        """
        if isinstance(parsed, dict):
            for key in ("statements", "facts", "items", "results"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break
            else:
                return None
        if not isinstance(parsed, list):
            return None
        out: list[str] = []
        for element in parsed:
            if isinstance(element, str):
                out.append(element)
            elif isinstance(element, dict):
                value = element.get("statement", element.get("text"))
                if not isinstance(value, str):
                    return None
                out.append(value)
            else:
                return None
        return [s for s in (t.strip() for t in out) if s]

    @classmethod
    def _parse_statement_array(cls, raw: str) -> list[str] | None:
        """Statements from the JSON array (or object) in ``raw``, or None."""
        for open_ch, close_ch in (("[", "]"), ("{", "}")):
            start = raw.find(open_ch)
            end = raw.rfind(close_ch) + 1
            if start == -1 or end <= start:
                continue
            try:
                parsed = json.loads(raw[start:end])
            except Exception:
                continue
            normalized = cls._normalize_payload(parsed)
            if normalized is None:
                continue
            try:
                jsonschema.validate(normalized, STATEMENT_ARRAY_SCHEMA)
            except Exception:
                continue
            return normalized
        return None

    def extract_statements(self, text: str) -> list[str]:
        """Self-contained statements from one chunk (cached).

        Unparsable replies are retried; if every reply fails, the chunk's first
        80 characters become its only statement.
        """
        cached = self.cache.get(K_EXTRACT, text, "")
        if cached is not None:
            return list(cached)
        base_prompt = Q_EXTRACT.format(
            text=text, max_chars=self.config.max_statement_chars
        )
        parsed: list[str] | None = None
        for attempt in range(1 + max(0, self.config.max_retries)):
            prompt = (
                base_prompt if attempt == 0 else base_prompt + "\n" + RETRY_JSON_SUFFIX
            )
            raw = self._ask(
                K_EXTRACT, prompt, max_tokens=self.config.extract_max_tokens
            )
            parsed = self._parse_statement_array(raw)
            if parsed is not None:
                break
            # A reply cut off by the token limit still holds whole statements;
            # retrying would only cut it off again.
            salvaged = salvage_string_array(raw)
            if salvaged:
                self.extract_salvaged += 1
                parsed = salvaged
                break
        if parsed is None:
            self.extract_fallback += 1
            fallback = text[:FALLBACK_STATEMENT_CHARS].strip()
            parsed = [fallback] if fallback else []
        limit = self.config.max_statement_chars
        result = [s.strip()[:limit] for s in parsed]
        result = [s for s in result if s]
        self.cache.put(K_EXTRACT, text, "", result)
        return list(result)

    # -- step 3: faithfulness -----------------------------------------------

    def is_supported(self, statement: str, source_text: str) -> bool:
        """Is ``statement`` supported by ``source_text`` alone?"""
        prompt = Q_SUPPORTED.format(text=source_text, statement=statement)
        return self._ask_bool(K_SUPPORTED, prompt, statement, source_text)

    # -- step 4: classification ---------------------------------------------

    @staticmethod
    def level_from_axes(
        comprehensive: bool, independent: bool, detail: bool
    ) -> NodeLevel:
        """Map the three yes/no axes to a level: detail wins, then independent,
        then comprehensive; all-no is a satellite.

        Ties go to the most specific level because most nodes in the hand-built
        reference diagram are satellites (207 of 240).
        """
        if detail:
            return NodeLevel.SATELLITE
        if independent:
            return NodeLevel.PLANET
        if comprehensive:
            return NodeLevel.SUN
        return NodeLevel.SATELLITE

    @staticmethod
    def _topics_block(topics: Sequence[str] | None) -> str:
        if not topics:
            return NO_TOPICS
        return "\n".join("- " + t for t in topics)

    @staticmethod
    def _context_key(context: str, topics_block: str) -> str:
        """Short hash of the context and topics, used in the cache key of the
        questions that show them."""
        digest = hashlib.sha1(
            (topics_block + "\x00" + context).encode("utf-8")
        ).hexdigest()
        return digest[:16]

    def classify_statement(
        self,
        statement: str,
        turn: int = -1,
        *,
        context: str = "",
        topics: Sequence[str] | None = None,
    ) -> NodeProposal:
        """Ask the three axis questions and return a NodeProposal (its score_*
        fields are 1.0 / 0.0).

        The two topic-level questions also see the source chunk and the current
        topics, since "is this a heading?" can't be answered without them.
        """
        topics_block = self._topics_block(topics)
        key_b = self._context_key(context, topics_block)
        comprehensive = self._ask_bool(
            K_COMPREHENSIVE,
            Q_COMPREHENSIVE.format(
                statement=statement, context=context, topics=topics_block
            ),
            statement,
            key_b,
        )
        independent = self._ask_bool(
            K_INDEPENDENT,
            Q_INDEPENDENT.format(
                statement=statement, context=context, topics=topics_block
            ),
            statement,
            key_b,
        )
        detail = self._ask_bool(
            K_DETAIL, Q_DETAIL.format(statement=statement), statement
        )
        level = self.level_from_axes(comprehensive, independent, detail)
        mass = {
            NodeLevel.SUN: self._sun_mass,
            NodeLevel.PLANET: self._planet_mass,
            NodeLevel.SATELLITE: self._satellite_mass,
        }[level]
        node = Node(text=statement, level=level, mass=mass, created_turn=turn)
        return NodeProposal(
            action=_LEVEL_ACTION[level],
            node=node,
            score_comprehensiveness=1.0 if comprehensive else 0.0,
            score_independence=1.0 if independent else 0.0,
            score_detail=1.0 if detail else 0.0,
        )

    # -- step 5: de-duplication ---------------------------------------------

    def deduplicate(self, proposals: list[NodeProposal]) -> tuple[list[NodeProposal], int]:
        """Drop this turn's duplicates: exact matches of the normalized text,
        then Q_SAME between proposals of the same level, in arrival order.

        Returns (kept, dropped_count).
        """
        kept: list[NodeProposal] = []
        seen: set[str] = set()
        dropped = 0
        for proposal in proposals:
            key = normalize_for_dedup(proposal.node.text)
            if key and key in seen:
                dropped += 1
                continue
            if key:
                seen.add(key)
            kept.append(proposal)

        final: list[NodeProposal] = []
        by_level: dict[NodeLevel, list[str]] = {}
        for proposal in kept:
            level = proposal.node.level
            candidates = by_level.get(level, [])
            if candidates:
                _idx, score = self.similarity.most_similar(
                    proposal.node.text, candidates, K_SAME
                )
                if score >= MATCH_SCORE:
                    dropped += 1
                    continue
            by_level.setdefault(level, []).append(proposal.node.text)
            final.append(proposal)
        return final, dropped

    # -- step 6: provisional structure --------------------------------------

    def build_provisional(self, proposals: list[NodeProposal]) -> ProvisionalStructure:
        """Link this turn's nodes to each other only, via Q_BELONGS."""
        return link_within_turn(
            [p.node for p in proposals], self.similarity.most_similar
        )

    # -- step 7: merge ------------------------------------------------------

    def _merge(self, base: CorrelationDiagram, structure: ProvisionalStructure) -> int:
        """Merge into ``base``; returns the number of orphan promotions."""
        return merge_into_base(self._merger, base, structure)

    # -- per-statement questions (optionally on threads) --------------------

    def _process_statements(
        self,
        items: Sequence[tuple[Chunk, str]],
        turn: int,
        topics: Sequence[str],
    ) -> tuple[list[NodeProposal], int]:
        """Support check plus the three axis questions for every statement.

        Returns (proposals, number dropped as unsupported).
        """

        def work(item: tuple[Chunk, str]) -> NodeProposal | None:
            chunk, statement = item
            if self.config.faithfulness_check and not self.is_supported(
                statement, chunk.text
            ):
                return None
            return self.classify_statement(
                statement, turn, context=chunk.text, topics=topics
            )

        results: Iterable[NodeProposal | None]
        if self.config.max_workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
                results = list(pool.map(work, items))
        else:
            results = [work(item) for item in items]

        proposals = [r for r in results if r is not None]
        return proposals, len(items) - len(proposals)

    # -- public entry point -------------------------------------------------

    def update(
        self,
        base: CorrelationDiagram,
        user_text: str,
        assistant_text: str,
        turn: int,
    ) -> HarnessTurnReport:
        """Process one user/assistant round trip, mutating ``base``.

        Normalize and accounting run in ``finally``, so a judge error mid-turn
        still leaves a consistent diagram and correct counts; the error then
        propagates to the caller.
        """
        calls_before = dict(self.runner.calls)
        unparsed_before = self.runner.total_unparsed()
        defaulted_before = self.runner.total_defaulted()
        fallback_before = self.extract_fallback
        salvaged_before = self.extract_salvaged
        hits_before = self.cache.hits
        nodes_before = len(base)
        self._vanished = 0
        self._attached = 0
        report = HarnessTurnReport()

        try:
            chunks = self._chunk(user_text, assistant_text, turn)
            report.chunks = len(chunks)

            topics = [se.sun.text for se in base.suns][: self.config.topics_in_prompt]
            items: list[tuple[Chunk, str]] = [
                (chunk, statement)
                for chunk in chunks
                for statement in self.extract_statements(chunk.text)
            ]
            report.statements = len(items)

            proposals, dropped = self._process_statements(items, turn, topics)
            report.dropped_unsupported = dropped

            proposals, dedup_dropped = self.deduplicate(proposals)
            self.dedup_dropped += dedup_dropped
            report.dedup_dropped = dedup_dropped

            structure = self.build_provisional(proposals)
            report.promoted = self._merge(base, structure)
        finally:
            # Exactly one normalize per update.
            base.normalize()
            self.normalize_calls += 1

            calls: dict[str, int] = {}
            for kind, total in self.runner.calls.items():
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
            report.merged = self._vanished + self._attached
            report.unparsed = self.runner.total_unparsed() - unparsed_before
            report.defaulted = self.runner.total_defaulted() - defaulted_before
            report.extract_fallback = self.extract_fallback - fallback_before
            report.extract_salvaged = self.extract_salvaged - salvaged_before
            for key in QUALITY_KEYS:
                self._add_total(key, getattr(report, key))

        return report
