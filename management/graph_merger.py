"""GraphMerger: merge a provisional diagram (built from new dialogue) into the
existing one.

An incoming node "matches" an existing node when the similarity function
scores them at or above config ``management.similarity_threshold``; a matching
incoming node is dropped and the existing one kept. "Attach" decisions (does
this node belong under that parent?) use ``attach_fn`` instead.

Case 1, the incoming tree has a sun (``merge``):
  - the sun matches an existing sun: drop it and merge its planets under the
    existing sun. A planet that matches one there is dropped and its
    satellites merged under the existing planet; a satellite that matches an
    existing satellite is dropped. Everything else is added.
  - otherwise: add the sun with its whole subtree.
Case 2, a loose planet (``merge_case2_planet``):
  - matches an existing planet: drop it and merge its satellites there;
  - else attaches to an existing sun: add it there as a planet;
  - else: it becomes a new sun and its satellites become its planets.
Case 3, a loose satellite (``merge_case3_satellite``):
  - matches an existing satellite: drop it;
  - else attaches to an existing planet: add it there;
  - else attaches to an existing sun: add it there as a planet;
  - else: it becomes a new sun.

Afterwards ``merge`` recomputes masses and coordinates (``normalize``).
"""
from __future__ import annotations
import copy
from typing import Callable

from models.correlation_diagram import CorrelationDiagram, SunEntry, PlanetEntry
from models.node import Node, NodeLevel
from utils.similarity import most_similar_index
from utils.config import get

# (query, candidates) -> (index of best candidate, score)
SimilarityFn = Callable[[str, list[str]], tuple[int, float]]


def _require(inserted: bool, node: Node, parent_id: str | None) -> None:
    """Raise if an insert into the base diagram failed, so a node is never
    dropped silently. ``add_sun`` / ``add_planet`` / ``add_satellite`` return
    False for a missing parent or at the configured limit (config.yaml
    ``graph.max_*``)."""
    if not inserted:
        raise RuntimeError(
            "capacity exceeded: could not add %s %r under %s (parent missing or at "
            "its configured maximum, see config.yaml graph.max_*)"
            % (node.level.value, node.text[:60], parent_id)
        )


def _default_similarity_fn(text: str, candidates: list[str]) -> tuple[int, float]:
    """Embedding similarity. ``most_similar_index`` is looked up on each call,
    so a test can monkeypatch it after the merger has been built."""
    return most_similar_index(text, candidates)


class GraphMerger:
    def __init__(
        self,
        similarity_fn: SimilarityFn | None = None,
        attach_fn: SimilarityFn | None = None,
    ):
        """
        similarity_fn: decides whether two nodes are the same thing.
            Default: utils.similarity.most_similar_index.
        attach_fn: decides whether a node belongs under a candidate parent
            (planet under sun, satellite under planet or sun).
            Default: similarity_fn.
        """
        self._sim_threshold: float = get("management", "similarity_threshold", 0.75)
        self._similarity_fn: SimilarityFn = (
            similarity_fn if similarity_fn is not None else _default_similarity_fn
        )
        self._attach_fn: SimilarityFn = (
            attach_fn if attach_fn is not None else self._similarity_fn
        )

    def merge(
        self, base: CorrelationDiagram, incoming: CorrelationDiagram
    ) -> CorrelationDiagram:
        """Merge every sun of ``incoming`` (with its subtree) into ``base``,
        then recompute masses and coordinates. Returns ``base``."""
        for se_in in list(incoming.suns):
            self._merge_case1(base, se_in)

        # Cases 2 and 3 are not reached from here: an incoming diagram holds
        # everything under suns. They are public for callers that merge loose
        # planets and satellites themselves.

        base.normalize()
        return base

    # ── case 1: the incoming tree has a sun ────────────────────────────

    def _merge_case1(self, base: CorrelationDiagram, se_in: SunEntry) -> None:
        match_idx = self._find_matching_sun_idx(se_in.sun.text, base)
        if match_idx is None:
            self._add_full_subtree_as_new_sun(base, se_in)
            return

        # Same topic as an existing sun: keep that sun, merge the planets into it.
        existing_sun_id = base.suns[match_idx].sun.node_id
        for pe_in in se_in.planets:
            self._merge_planet_into_sun(base, existing_sun_id, pe_in)

    def _add_full_subtree_as_new_sun(
        self, base: CorrelationDiagram, se_in: SunEntry
    ) -> None:
        new_sun = copy.deepcopy(se_in.sun)
        _require(base.add_sun(new_sun), new_sun, None)
        for pe_in in se_in.planets:
            self._add_planet_with_satellites(base, pe_in.planet, pe_in.satellites, new_sun.node_id)

    def _merge_planet_into_sun(
        self, base: CorrelationDiagram, sun_id: str, pe_in: PlanetEntry
    ) -> None:
        """Merge an incoming planet (and its satellites) under an existing sun."""
        match_planet_id = self._find_matching_planet_id(pe_in.planet.text, sun_id, base)
        if match_planet_id is None:
            self._add_planet_with_satellites(base, pe_in.planet, pe_in.satellites, sun_id)
            return
        # Same as an existing planet: keep it, merge only the satellites.
        for sat_in in pe_in.satellites:
            self._merge_satellite_under_planet(base, match_planet_id, sat_in)

    def _merge_satellite_under_planet(
        self, base: CorrelationDiagram, planet_id: str, satellite: Node
    ) -> None:
        """Add ``satellite`` under the planet unless it matches one already there."""
        result = base.find_planet_entry(planet_id)
        if result is None:
            return
        _, pe = result
        if self._best_match(satellite.text, [s.text for s in pe.satellites]) is not None:
            return
        new_sat = copy.deepcopy(satellite)
        _require(base.add_satellite(new_sat, planet_id), new_sat, planet_id)

    def _add_planet_with_satellites(
        self, base: CorrelationDiagram, planet: Node, satellites: list[Node], sun_id: str
    ) -> None:
        """Add copies of ``planet`` and its ``satellites`` under the sun."""
        new_planet = copy.deepcopy(planet)
        _require(base.add_planet(new_planet, sun_id), new_planet, sun_id)
        for sat in satellites:
            new_sat = copy.deepcopy(sat)
            _require(base.add_satellite(new_sat, new_planet.node_id), new_sat, new_planet.node_id)

    # ── case 2: a loose planet (with its satellites) ───────────────────

    def merge_case2_planet(
        self, base: CorrelationDiagram, planet_a: Node, satellites: list[Node]
    ) -> None:
        match_planet = self._find_any_matching_planet(planet_a.text, base)
        if match_planet is not None:
            sun_idx, planet_idx = match_planet
            target_planet_id = base.suns[sun_idx].planets[planet_idx].planet.node_id
            for sat in satellites:
                self._merge_satellite_under_planet(base, target_planet_id, sat)
            return

        match_sun_idx = self._find_matching_sun_idx(planet_a.text, base, attach=True)
        if match_sun_idx is not None:
            target_sun_id = base.suns[match_sun_idx].sun.node_id
            self._add_planet_with_satellites(base, planet_a, satellites, target_sun_id)
            return

        # Unrelated to everything: a new sun, its satellites become its planets.
        promoted_sun = copy.deepcopy(planet_a)
        promoted_sun.level = NodeLevel.SUN
        _require(base.add_sun(promoted_sun), promoted_sun, None)
        for sat in satellites:
            promoted_planet = copy.deepcopy(sat)
            promoted_planet.level = NodeLevel.PLANET
            _require(base.add_planet(promoted_planet, promoted_sun.node_id), promoted_planet, promoted_sun.node_id)

    # ── case 3: a loose satellite ──────────────────────────────────────

    def merge_case3_satellite(
        self, base: CorrelationDiagram, satellite: Node
    ) -> None:
        all_sat_texts = [sat.text for se in base.suns for pe in se.planets for sat in pe.satellites]
        if self._best_match(satellite.text, all_sat_texts) is not None:
            return

        match_planet = self._find_any_matching_planet(satellite.text, base, attach=True)
        if match_planet is not None:
            sun_idx, planet_idx = match_planet
            target_planet_id = base.suns[sun_idx].planets[planet_idx].planet.node_id
            new_sat = copy.deepcopy(satellite)
            _require(base.add_satellite(new_sat, target_planet_id), new_sat, target_planet_id)
            return

        match_sun_idx = self._find_matching_sun_idx(satellite.text, base, attach=True)
        if match_sun_idx is not None:
            promoted_planet = copy.deepcopy(satellite)
            promoted_planet.level = NodeLevel.PLANET
            target_sun_id = base.suns[match_sun_idx].sun.node_id
            _require(base.add_planet(promoted_planet, target_sun_id), promoted_planet, target_sun_id)
            return

        promoted_sun = copy.deepcopy(satellite)
        promoted_sun.level = NodeLevel.SUN
        _require(base.add_sun(promoted_sun), promoted_sun, None)

    # ── matching helpers ───────────────────────────────────────────────

    def _best_match(
        self, text: str, candidates: list[str], *, attach: bool = False
    ) -> int | None:
        """Index of the candidate ``text`` matches (score >= threshold), else None."""
        if not candidates:
            return None
        fn = self._attach_fn if attach else self._similarity_fn
        idx, score = fn(text, candidates)
        return idx if score >= self._sim_threshold else None

    def _find_matching_sun_idx(
        self, text: str, cd: CorrelationDiagram, *, attach: bool = False
    ) -> int | None:
        return self._best_match(text, [se.sun.text for se in cd.suns], attach=attach)

    def _find_matching_planet_id(
        self, text: str, sun_id: str, cd: CorrelationDiagram
    ) -> str | None:
        """Id of the matching planet under the given sun, or None."""
        se = cd.find_sun_entry(sun_id)
        if se is None:
            return None
        idx = self._best_match(text, [pe.planet.text for pe in se.planets])
        return None if idx is None else se.planets[idx].planet.node_id

    def _find_any_matching_planet(
        self, text: str, cd: CorrelationDiagram, *, attach: bool = False
    ) -> tuple[int, int] | None:
        """(sun_idx, planet_idx) of the matching planet anywhere in ``cd``, or None."""
        positions: list[tuple[int, int]] = []
        texts: list[str] = []
        for s_idx, se in enumerate(cd.suns):
            for p_idx, pe in enumerate(se.planets):
                positions.append((s_idx, p_idx))
                texts.append(pe.planet.text)
        idx = self._best_match(text, texts, attach=attach)
        return None if idx is None else positions[idx]
