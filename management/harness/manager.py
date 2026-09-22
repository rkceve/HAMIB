"""HarnessManager: the coded manager procedure (0036-0062) with an LLM that only
answers small, verifiable questions.

Design: HARNESS_DESIGN.md Stream B, B1/B2/B5/B7.

Per turn:
  1. chunk  (chunking.split_candidates + optional Q_BOUNDARY merge)  0037-0038
  2. extract self-contained statements per chunk (JSON)              0030 / D-4
  3. faithfulness check per statement                                D-4
  4. classify on three yes/no axes                                   0039-0040
  5. de-duplicate this turn's statements (H8)
  6. link this turn's nodes to each other only                       0041
  7. merge the provisional structure into the base CD                0042-0061
  8. normalize once (mass = satellite count, coordinates)            0030 / 0062

Step 0 (D-7 query-turn skip) is the CALLER's job; the harness does not
re-implement it.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

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

# jsonschema for the extraction reply: a JSON array of non-empty strings.
STATEMENT_ARRAY_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {"type": "string", "minLength": 1},
}

# Length of the fallback statement when extraction never returns valid JSON.
# Mirrors build_cd_offline._safe_extractor_fn's text[:80].
FALLBACK_STATEMENT_CHARS = 80

# Quality counter keys (H9), reported per turn and accumulated in `totals`.
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

# Tolerant recovery of a truncated JSON array: every COMPLETE double-quoted
# element, escapes honoured.  Applied only to the text after the leading '['.
_QUOTED_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def salvage_string_array(raw: str) -> list[str]:
    """Recover the complete string elements of a truncated JSON array (H1).

    ``["a", "b", "c`` -> ``["a", "b"]``.  Returns [] when there is no '[' at all
    or when nothing complete can be recovered.
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
    """Casefolded, punctuation- and whitespace-free key for H8 de-duplication."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        ch for ch in folded if not unicodedata.category(ch).startswith(("P", "Z", "C"))
    )


# -- config ------------------------------------------------------------------


@dataclass
class HarnessConfig:
    """Mirrors the harness: section of config.yaml (B5)."""

    boundary_check: bool = True
    faithfulness_check: bool = True
    shortlist_k: int = 5
    max_retries: int = 2
    max_statement_chars: int = 120
    judge_max_tokens: int = 256
    # H1: extraction returns a JSON array and needs far more room than a
    # one-word yes/no answer; 256 tokens truncated multi-statement replies.
    extract_max_tokens: int = 1024
    # H1: chunk length cap, also the ceiling for the Q_BOUNDARY merge.
    chunk_max_chars: int = 800
    # H6: how many existing sun texts are shown to the classification questions.
    topics_in_prompt: int = 8
    # H7: 1 = sequential (default).  Only the per-statement stage is parallel.
    max_workers: int = 1

    @classmethod
    def from_config(cls) -> "HarnessConfig":
        d = cls()
        return cls(
            boundary_check=bool(get("harness", "boundary_check", d.boundary_check)),
            faithfulness_check=bool(
                get("harness", "faithfulness_check", d.faithfulness_check)
            ),
            shortlist_k=int(get("harness", "shortlist_k", d.shortlist_k)),
            max_retries=int(get("harness", "max_retries", d.max_retries)),
            max_statement_chars=int(
                get("harness", "max_statement_chars", d.max_statement_chars)
            ),
            judge_max_tokens=int(get("harness", "judge_max_tokens", d.judge_max_tokens)),
            extract_max_tokens=int(
                get("harness", "extract_max_tokens", d.extract_max_tokens)
            ),
            chunk_max_chars=int(get("harness", "chunk_max_chars", d.chunk_max_chars)),
            topics_in_prompt=int(
                get("harness", "topics_in_prompt", d.topics_in_prompt)
            ),
            max_workers=int(get("harness", "max_workers", d.max_workers)),
        )


# -- report / provisional structure ------------------------------------------


@dataclass
class HarnessTurnReport:
    """Per-turn statistics (B2 step 7)."""

    calls: dict[str, int] = field(default_factory=dict)
    cache_hits: int = 0
    statements: int = 0
    dropped_unsupported: int = 0
    added: int = 0
    merged: int = 0
    promoted: int = 0
    chunks: int = 0
    # H14: `merged` = vanished + attached, kept for compatibility.
    vanished: int = 0
    attached: int = 0
    # H9 quality counters.
    unparsed: int = 0
    defaulted: int = 0
    extract_fallback: int = 0
    extract_salvaged: int = 0
    dedup_dropped: int = 0

    def total_calls(self) -> int:
        return sum(self.calls.values())


@dataclass
class ProvisionalStructure:
    """This turn's nodes, linked to each other only (0041)."""

    suns: list[SunEntry] = field(default_factory=list)
    orphan_planets: list[PlanetEntry] = field(default_factory=list)
    orphan_satellites: list[Node] = field(default_factory=list)


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
        # The merger routes every similarity decision through the counting
        # wrapper so the report can attribute vanish/attach events (B2 step 7).
        self._merger = GraphMerger(
            similarity_fn=self._counting_similarity,
            attach_fn=self._counting_attach,
        )
        self._vanished = 0
        self._attached = 0
        self._sun_mass = float(get("graph", "default_sun_mass", 1.0))
        self._planet_mass = float(get("graph", "default_planet_mass", 0.5))
        self._satellite_mass = float(get("graph", "default_satellite_mass", 0.1))
        # Cumulative totals over the whole run (used by build_cd_offline): call
        # counts by question kind PLUS the quality counters of QUALITY_KEYS.
        self.totals: dict[str, int] = {}
        self.total_cache_hits = 0
        self.normalize_calls = 0
        # Run-level extraction quality counters.
        self.extract_fallback = 0
        self.extract_salvaged = 0
        self.dedup_dropped = 0

    # -- totals views -------------------------------------------------------

    def call_totals(self) -> dict[str, int]:
        """The question-kind part of ``totals`` (what `harness_calls` reports)."""
        return {k: v for k, v in self.totals.items() if k in ALL_KINDS}

    def quality_totals(self) -> dict[str, int]:
        """The H9 quality part of ``totals``."""
        return {k: self.totals.get(k, 0) for k in QUALITY_KEYS}

    def load_totals(self, totals: dict[str, int]) -> None:
        """Restore cumulative counters from a checkpoint (H12)."""
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
        """GraphMerger's similarity_fn: Q_SAME.  A "yes" here means the incoming
        node VANISHED into an existing one (0045/0046/0048/0056/0059)."""
        idx, score = self.similarity.most_similar(query, candidates)
        if score >= MATCH_SCORE:
            self._vanished += 1
        return idx, score

    def _counting_attach(self, query: str, candidates: list[str]) -> tuple[int, float]:
        """GraphMerger's attach_fn: Q_BELONGS ("does S belong under topic T?") for
        the attach decisions of 0057 / 0060 / 0061.  Asking Q_SAME there (the
        pre-fix behaviour) made every orphan planet/satellite fail to attach and
        get promoted, inflating the sun count."""
        idx, score = self.similarity.most_similar(query, candidates, K_BELONGS)
        if score >= MATCH_SCORE:
            self._attached += 1
        return idx, score

    # -- step 1: chunking ---------------------------------------------------

    def _chunk(self, user_text: str, assistant_text: str, turn: int) -> list[Chunk]:
        """0037-0038.  Language-agnostic candidates, then the Q_BOUNDARY merge."""
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
            # The question always uses the ORIGINAL adjacent pair (cache-friendly
            # and closest to 0038's "boundary between two meaning units").
            if prev.source == cur.source and prev.turn == cur.turn:
                joined = merged[-1].text + " " + cur.text
                # H1: an unbounded merge chain rebuilds the whole turn as one
                # chunk, which is exactly the failure this rewrite removes.
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
        """Accept the three shapes seen in the wild and return a string list.

        ``["a", "b"]`` / ``{"statements": [...]}`` / ``[{"statement": "a"}]`` or
        ``[{"text": "a"}]``.  Blank elements are DROPPED rather than failing the
        whole reply (H1).
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
        """0030 / D-4: 0..N self-contained descriptive statements for one chunk."""
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
            # H1: a reply truncated by the token budget still carries whole
            # statements; retrying it verbatim would only truncate again.
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
        prompt = Q_SUPPORTED.format(text=source_text, statement=statement)
        return self._ask_bool(K_SUPPORTED, prompt, statement, source_text)

    # -- step 4: classification ---------------------------------------------

    @staticmethod
    def level_from_axes(
        comprehensive: bool, independent: bool, detail: bool
    ) -> NodeLevel:
        """0040 with the documented tie rule detail > independent > comprehensive.

        Specific beats general: the oracle CD is 1 sun / 32 planets / 207
        satellites, so an ambiguous statement belongs at the bottom.  All-no also
        yields satellite.
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
        """Stable cache key part for the context-carrying questions (H6)."""
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
        """Three binary questions -> one NodeProposal (score_* kept for compat).

        H6: the two topic-level axes also see the discussion the statement came
        from and the topics already in the diagram, so "is this a heading?" is
        answerable at all.  The context is part of their cache key.
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

    # -- step 5: de-duplication (H8) ----------------------------------------

    def deduplicate(self, proposals: list[NodeProposal]) -> tuple[list[NodeProposal], int]:
        """Drop within-turn duplicates: exact (normalized text) then Q_SAME.

        The judge question is asked only between proposals of the SAME level, in
        arrival order, through SimilarityJudge (so the shortlist setting is
        respected).  Returns (kept, dropped_count).
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
        """0041: link this turn's nodes to each other only, via Q_BELONGS."""
        cd = CorrelationDiagram()
        sun_texts: list[str] = []
        sun_ids: list[str] = []
        for p in proposals:
            if p.node.level is not NodeLevel.SUN:
                continue
            if cd.add_sun(p.node):
                sun_texts.append(p.node.text)
                sun_ids.append(p.node.node_id)

        orphan_planets: list[PlanetEntry] = []
        planet_texts: list[str] = []
        # Either the id of a planet already inside cd, or an orphan PlanetEntry.
        planet_refs: list[str | PlanetEntry] = []
        for p in proposals:
            if p.node.level is not NodeLevel.PLANET:
                continue
            attached = False
            if sun_texts:
                idx, score = self.similarity.most_similar(
                    p.node.text, sun_texts, K_BELONGS
                )
                if score >= MATCH_SCORE and cd.add_planet(p.node, sun_ids[idx]):
                    planet_refs.append(p.node.node_id)
                    attached = True
            if not attached:
                p.node.level = NodeLevel.PLANET
                p.node.parent_id = None
                entry = PlanetEntry(planet=p.node)
                orphan_planets.append(entry)
                planet_refs.append(entry)
            planet_texts.append(p.node.text)

        orphan_satellites: list[Node] = []
        for p in proposals:
            if p.node.level is not NodeLevel.SATELLITE:
                continue
            attached = False
            if planet_texts:
                idx, score = self.similarity.most_similar(
                    p.node.text, planet_texts, K_BELONGS
                )
                if score >= MATCH_SCORE:
                    ref = planet_refs[idx]
                    if isinstance(ref, str):
                        attached = cd.add_satellite(p.node, ref)
                    else:
                        p.node.level = NodeLevel.SATELLITE
                        p.node.parent_id = ref.planet.node_id
                        ref.satellites.append(p.node)
                        attached = True
            if not attached:
                orphan_satellites.append(p.node)

        return ProvisionalStructure(
            suns=cd.suns,
            orphan_planets=orphan_planets,
            orphan_satellites=orphan_satellites,
        )

    # -- step 7: merge ------------------------------------------------------

    def _merge(self, base: CorrelationDiagram, structure: ProvisionalStructure) -> int:
        """0042-0061.  Returns the number of orphan promotions observed."""
        promoted = 0
        if structure.suns:
            incoming = CorrelationDiagram()
            incoming.suns = structure.suns
            # NOTE: GraphMerger.merge() normalizes internally; the harness's own
            # single normalize() still runs once at the end of update().
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

    # -- per-statement stage (optionally concurrent, H7) --------------------

    def _process_statements(
        self,
        items: Sequence[tuple[Chunk, str]],
        turn: int,
        topics: Sequence[str],
    ) -> tuple[list[NodeProposal], int]:
        """Q_SUPPORTED + the three axes for every statement of the turn."""

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
        dropped = len(items) - len(proposals)
        return proposals, dropped

    # -- public entry point -------------------------------------------------

    def update(
        self,
        base: CorrelationDiagram,
        user_text: str,
        assistant_text: str,
        turn: int,
    ) -> HarnessTurnReport:
        """One conversation round-trip (0036).  Mutates base in place.

        H4: the normalize and the accounting run in a `finally`, so a judge that
        raises mid-turn still leaves the diagram consistent and still costs what
        it cost.  The exception propagates to the caller (the driver counts it).
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
            # 0030 + 0062: exactly one harness-level normalize per update.
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
