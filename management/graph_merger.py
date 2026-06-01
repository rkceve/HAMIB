"""
GraphMerger: merges a provisional structure (incoming CD) into the existing
correlation diagram (base CD).

Detailed merge logic:

  Case 1: the provisional structure contains sun nodes
    - Sun B is similar to existing sun A -> B is dropped, A remains; B's
      descendants are merged into A
      - 1) only planets under B
      - 2) planets and satellites under B
      - 3) only satellites under B -> promote each satellite and merge
    - Sun B is standalone (no descendants) -> compared with existing suns
    - Sun B is dissimilar to all existing suns -> added as a new sun

  Case 2: no sun in the provisional structure, only planets / planets+satellites
    - Planet A is similar to existing planet B -> A is dropped, only its
      satellites are added under B
    - Planet A is dissimilar -> compared with all suns
      - a similar sun exists -> A is added as a new planet under it
      - no similar sun -> A is promoted to a sun

  Case 3: only satellites in the provisional structure
    - satellite vs existing satellite -> dropped if similar
    - satellite vs existing planet -> added under it if similar
    - satellite vs existing sun -> promoted to a planet and added if similar
    - dissimilar to all -> promoted to a sun and added

After merging, mass is recomputed from the total satellite-node count.
"""
from __future__ import annotations
import copy

from models.correlation_diagram import CorrelationDiagram, SunEntry, PlanetEntry
from models.node import Node, NodeLevel
from utils.similarity import most_similar_index
from utils.config import get


class GraphMerger:
    def __init__(self):
        self._sim_threshold: float = get("management", "similarity_threshold", 0.75)

    # ── Entry point ────────────────────────────────────────────────────

    def merge(
        self, base: CorrelationDiagram, incoming: CorrelationDiagram
    ) -> CorrelationDiagram:
        """
        Classifies each subtree of `incoming` into one of three cases and
        merges it into `base`.

        If incoming.suns contains a sun -> Case 1.
        If incoming has no sun -> the Case 2/3 helpers are used
          - free planet -> Case 2
          - free satellite -> Case 3

        After merging, mass and coordinates are recomputed.
        """
        # Case 1: provisional structure containing suns
        for se_in in list(incoming.suns):
            self._merge_case1(base, se_in)

        # Case 2/3 are exposed as merge_case2_planet / merge_case3_satellite.
        # In the current NodeClassifier, orphan planets/satellites are promoted
        # to suns and flow into Case 1, so no direct traversal of non-sun
        # entries is needed here.

        # Recompute mass and coordinates
        base.normalize()
        return base

    # ── Case 1: provisional structure contains sun nodes ────────────────

    def _merge_case1(self, base: CorrelationDiagram, se_in: SunEntry) -> None:
        """
        Merges a provisional sun node B (= se_in.sun) into base.
        """
        sun_b = se_in.sun
        match_idx = self._find_matching_sun_idx(sun_b.text, base)

        if match_idx is None:
            # B is dissimilar to every existing sun -> add B as a new sun and
            # carry over its entire subtree
            self._add_full_subtree_as_new_sun(base, se_in)
            return

        # B is similar to existing sun A -> B is dropped, A remains; B's
        # descendants are merged into A
        sun_a_id = base.suns[match_idx].sun.node_id

        if not se_in.planets:
            # B is standalone (no descendants) -> done, A is unchanged
            return

        # B has planets, which may also have satellites attached
        for pe_in in se_in.planets:
            self._merge_planet_into_sun(base, sun_a_id, pe_in)

    def _add_full_subtree_as_new_sun(
        self, base: CorrelationDiagram, se_in: SunEntry
    ) -> None:
        """Adds B as a new sun and carries over its planets and satellites."""
        new_sun = copy.deepcopy(se_in.sun)
        if not base.add_sun(new_sun):
            return  # sun-count limit exceeded -> give up
        new_sun_id = new_sun.node_id
        for pe_in in se_in.planets:
            new_planet = copy.deepcopy(pe_in.planet)
            if base.add_planet(new_planet, new_sun_id):
                for sat_in in pe_in.satellites:
                    base.add_satellite(copy.deepcopy(sat_in), new_planet.node_id)

    def _merge_planet_into_sun(
        self, base: CorrelationDiagram, sun_a_id: str, pe_in: PlanetEntry
    ) -> None:
        """
        Merges planet C (= pe_in.planet) under existing sun A.

        - C is similar to existing planet D -> C is dropped, its satellites are
          added under D (similar satellites are dropped)
        - C is dissimilar -> added as a new planet under A, carrying over its
          satellites
        """
        planet_c = pe_in.planet
        match_planet_id = self._find_matching_planet_id(planet_c.text, sun_a_id, base)

        if match_planet_id is not None:
            # C is similar to D -> C is dropped, D remains; only C's satellites
            # move under D
            for sat_in in pe_in.satellites:
                self._merge_satellite_under_planet(base, match_planet_id, sat_in)
            return

        # C is dissimilar -> add as a new planet under A
        new_planet = copy.deepcopy(planet_c)
        if base.add_planet(new_planet, sun_a_id):
            new_planet_id = new_planet.node_id
            for sat_in in pe_in.satellites:
                base.add_satellite(copy.deepcopy(sat_in), new_planet_id)

    def _merge_satellite_under_planet(
        self, base: CorrelationDiagram, planet_id: str, satellite: Node
    ) -> None:
        """
        Drops satellite E if it is similar to a satellite F already under
        planet D, otherwise adds it.
        """
        result = base.find_planet_entry(planet_id)
        if result is None:
            return
        _, pe = result
        existing_sat_texts = [s.text for s in pe.satellites]
        if existing_sat_texts:
            idx, score = most_similar_index(satellite.text, existing_sat_texts)
            if score >= self._sim_threshold:
                # E is similar to F -> E is dropped, F remains (do nothing)
                return
        # E is dissimilar -> add as a new satellite under D
        base.add_satellite(copy.deepcopy(satellite), planet_id)

    # ── Case 2: no sun, only planets / planets+satellites ───────────────

    def merge_case2_planet(
        self, base: CorrelationDiagram, planet_a: Node, satellites: list[Node]
    ) -> None:
        """
        Handles a provisional structure with no sun, containing only a free
        planet A (and its satellites).
        """
        # A is similar to existing planet B -> A is dropped, only its
        # satellites move under B
        match_planet = self._find_any_matching_planet(planet_a.text, base)
        if match_planet is not None:
            sun_idx, planet_idx = match_planet
            target_planet_id = base.suns[sun_idx].planets[planet_idx].planet.node_id
            for sat in satellites:
                self._merge_satellite_under_planet(base, target_planet_id, sat)
            return

        # A is dissimilar -> compared with all suns
        match_sun_idx = self._find_matching_sun_idx(planet_a.text, base)
        if match_sun_idx is not None:
            # a similar sun exists -> add as a new planet
            target_sun_id = base.suns[match_sun_idx].sun.node_id
            new_planet = copy.deepcopy(planet_a)
            if base.add_planet(new_planet, target_sun_id):
                for sat in satellites:
                    base.add_satellite(copy.deepcopy(sat), new_planet.node_id)
            return

        # no similar sun -> promote A to a sun and its satellites to planets
        promoted_sun = copy.deepcopy(planet_a)
        promoted_sun.level = NodeLevel.SUN
        if not base.add_sun(promoted_sun):
            return
        for sat in satellites:
            promoted_planet = copy.deepcopy(sat)
            promoted_planet.level = NodeLevel.PLANET
            base.add_planet(promoted_planet, promoted_sun.node_id)

    # ── Case 3: only satellites ─────────────────────────────────────────

    def merge_case3_satellite(
        self, base: CorrelationDiagram, satellite: Node
    ) -> None:
        """
        Handles a provisional structure containing only satellite nodes.
        """
        # Compare with all existing satellites
        all_satellites: list[tuple[str, Node]] = []
        for se in base.suns:
            for pe in se.planets:
                for sat in pe.satellites:
                    all_satellites.append((sat.text, sat))
        if all_satellites:
            texts = [t for t, _ in all_satellites]
            idx, score = most_similar_index(satellite.text, texts)
            if score >= self._sim_threshold:
                return  # a similar satellite exists -> dropped

        # Compare with existing planets
        match_planet = self._find_any_matching_planet(satellite.text, base)
        if match_planet is not None:
            sun_idx, planet_idx = match_planet
            target_planet_id = base.suns[sun_idx].planets[planet_idx].planet.node_id
            base.add_satellite(copy.deepcopy(satellite), target_planet_id)
            return

        # Compare with existing suns (promote satellite to a planet)
        match_sun_idx = self._find_matching_sun_idx(satellite.text, base)
        if match_sun_idx is not None:
            promoted_planet = copy.deepcopy(satellite)
            promoted_planet.level = NodeLevel.PLANET
            target_sun_id = base.suns[match_sun_idx].sun.node_id
            base.add_planet(promoted_planet, target_sun_id)
            return

        # dissimilar to all -> promote to a sun
        promoted_sun = copy.deepcopy(satellite)
        promoted_sun.level = NodeLevel.SUN
        base.add_sun(promoted_sun)

    # ── Similarity helpers ──────────────────────────────────────────────

    def _find_matching_sun_idx(
        self, text: str, cd: CorrelationDiagram
    ) -> int | None:
        sun_texts = [se.sun.text for se in cd.suns]
        if not sun_texts:
            return None
        idx, score = most_similar_index(text, sun_texts)
        if score >= self._sim_threshold:
            return idx
        return None

    def _find_matching_planet_id(
        self, text: str, sun_id: str, cd: CorrelationDiagram
    ) -> str | None:
        se = cd.find_sun_entry(sun_id)
        if se is None or not se.planets:
            return None
        planet_texts = [pe.planet.text for pe in se.planets]
        idx, score = most_similar_index(text, planet_texts)
        if score >= self._sim_threshold:
            return se.planets[idx].planet.node_id
        return None

    def _find_any_matching_planet(
        self, text: str, cd: CorrelationDiagram
    ) -> tuple[int, int] | None:
        """Returns (sun_idx, planet_idx) of the most similar planet across all suns."""
        all_planet_texts: list[str] = []
        index_map: list[tuple[int, int]] = []
        for s_idx, se in enumerate(cd.suns):
            for p_idx, pe in enumerate(se.planets):
                all_planet_texts.append(pe.planet.text)
                index_map.append((s_idx, p_idx))
        if not all_planet_texts:
            return None
        idx, score = most_similar_index(text, all_planet_texts)
        if score >= self._sim_threshold:
            return index_map[idx]
        return None
