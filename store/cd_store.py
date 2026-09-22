"""
CDStore: 相関図データ (CorrelationDiagram) のインメモリ保持 + JSON永続化。

セッションごとに1つの「現行CD」と、評価用の「評価CD」を管理する。
評価CDは評価ユニットが使用した後に削除される（特許§0065参照）。
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
        self._eval: Optional[CorrelationDiagram] = None    # 評価CD（一時的）
        self._round_trip_count: int = 0
        self._persist_path = persist_path

        if persist_path and persist_path.exists():
            self._load()

    # ── 現行CD ────────────────────────────────────────────────────────

    def get_current(self) -> CorrelationDiagram:
        with self._lock:
            return self._current

    def set_current(self, cd: CorrelationDiagram) -> None:
        with self._lock:
            self._current = cd
            self._save()

    # ── 評価CD ────────────────────────────────────────────────────────

    def create_eval(self) -> CorrelationDiagram:
        """現行CDのディープコピーから評価CDを生成して返す。"""
        with self._lock:
            self._eval = self._current.clone()
            return self._eval

    def get_eval(self) -> Optional[CorrelationDiagram]:
        with self._lock:
            return self._eval

    def set_eval(self, cd: CorrelationDiagram) -> None:
        """EvalGraphBuilder が構築した評価CDをストアにセットする。"""
        with self._lock:
            self._eval = cd

    def discard_eval(self) -> None:
        """評価CD を削除する（評価処理終了後に呼ぶ）。"""
        with self._lock:
            self._eval = None

    def replace_with_eval(self) -> None:
        """評価CDのスコアが優れている場合に現行CDを置換する。"""
        with self._lock:
            if self._eval is not None:
                self._current = self._eval
                self._eval = None
                self._save()

    # ── ラウンドトリップ管理 ──────────────────────────────────────────

    def increment_round_trip(self) -> int:
        with self._lock:
            self._round_trip_count += 1
            return self._round_trip_count

    def get_round_trip_count(self) -> int:
        with self._lock:
            return self._round_trip_count

    # ── 永続化 ────────────────────────────────────────────────────────

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
