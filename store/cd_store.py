"""
CDStore: in-memory holding of CorrelationDiagram data plus JSON persistence.

Manages one "current CD" per session and an "evaluation CD" used for evaluation.
The evaluation CD is discarded after the evaluation unit has used it.
"""
from __future__ import annotations
import json
import threading
from pathlib import Path
from typing import Optional

from models.correlation_diagram import CorrelationDiagram


class CDStore:
    def __init__(self, persist_path: Optional[Path] = None):
        self._lock = threading.Lock()
        self._current: CorrelationDiagram = CorrelationDiagram()
        self._eval: Optional[CorrelationDiagram] = None    # evaluation CD (temporary)
        self._round_trip_count: int = 0
        self._persist_path = persist_path

        if persist_path and persist_path.exists():
            self._load()

    # ── Current CD ────────────────────────────────────────────────────

    def get_current(self) -> CorrelationDiagram:
        with self._lock:
            return self._current

    def set_current(self, cd: CorrelationDiagram) -> None:
        with self._lock:
            self._current = cd
            self._save()

    # ── Evaluation CD ─────────────────────────────────────────────────

    def create_eval(self) -> CorrelationDiagram:
        """Create and return an evaluation CD from a deep copy of the current CD."""
        with self._lock:
            self._eval = self._current.clone()
            return self._eval

    def get_eval(self) -> Optional[CorrelationDiagram]:
        with self._lock:
            return self._eval

    def set_eval(self, cd: CorrelationDiagram) -> None:
        """Set the evaluation CD built by EvalGraphBuilder into the store."""
        with self._lock:
            self._eval = cd

    def discard_eval(self) -> None:
        """Discard the evaluation CD (called after evaluation finishes)."""
        with self._lock:
            self._eval = None

    def replace_with_eval(self) -> None:
        """Replace the current CD when the evaluation CD scores better."""
        with self._lock:
            if self._eval is not None:
                self._current = self._eval
                self._eval = None
                self._save()

    # ── Round-trip bookkeeping ────────────────────────────────────────

    def increment_round_trip(self) -> int:
        with self._lock:
            self._round_trip_count += 1
            return self._round_trip_count

    def get_round_trip_count(self) -> int:
        with self._lock:
            return self._round_trip_count

    # ── Persistence ───────────────────────────────────────────────────

    def _save(self) -> None:
        if self._persist_path is None:
            return
        data = {
            "round_trip_count": self._round_trip_count,
            "current": self._current.to_dict(),
        }
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load(self) -> None:
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            self._round_trip_count = data.get("round_trip_count", 0)
            self._current = CorrelationDiagram.from_dict(data.get("current", {}))
        except Exception:
            pass
