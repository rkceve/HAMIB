"""
Controller: CMSクライアント側のメインオーケストレーター。

責務:
  1. ユーザー入力をサーバーへ送信（CDペイロード付き）
  2. サーバー応答を受け取り、管理ユニットでCD更新
  3. 5ラウンドトリップごとに評価ユニットを起動
"""
from __future__ import annotations
import httpx

from store.cd_store import CDStore
from communication.cd_serializer import CDSerializer
from management.text_chunker import TextChunker
from management.node_classifier import NodeClassifier
from management.graph_builder import GraphBuilder
from evaluation.eval_graph_builder import EvalGraphBuilder
from evaluation.replacer import Replacer
from utils.config import get, load_config


class Controller:
    def __init__(self, store: CDStore):
        self._store = store
        self._serializer = CDSerializer()
        self._chunker = TextChunker()
        self._classifier = NodeClassifier()
        self._builder = GraphBuilder()
        self._eval_builder = EvalGraphBuilder()
        self._replacer = Replacer(store)

        cfg = load_config()
        server_cfg = cfg.get("server", {})
        client_cfg = cfg.get("client", {})
        host = client_cfg.get("host", "localhost")
        port = client_cfg.get("server_port", 8080)
        self._server_url = f"http://{host}:{port}"
        self._eval_interval: int = get("evaluation", "eval_interval_rounds", 5)

        self._recent_turns: list[tuple[str, str]] = []

    def chat(self, user_text: str) -> str:
        """
        ユーザーの発話を処理してアシスタントの返答文字列を返す。
        """
        cd = self._store.get_current()
        payload = self._serializer.to_api_payload(cd)

        # サーバーへリクエスト
        response_text = self._call_server(user_text, payload)

        # CD更新（管理ユニット）
        self._update_cd(user_text, response_text)

        # ラウンドトリップ管理
        turn = self._store.increment_round_trip()
        self._recent_turns.append((user_text, response_text))
        if len(self._recent_turns) > self._eval_interval:
            self._recent_turns.pop(0)

        # 評価ユニット（5ラウンドごと）
        if turn % self._eval_interval == 0:
            self._run_evaluation(turn)

        return response_text

    def _call_server(self, user_text: str, payload: dict) -> str:
        try:
            resp = httpx.post(
                f"{self._server_url}/chat",
                json={"user_text": user_text, **payload},
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            return f"[Server Error: {e}]"

    def _update_cd(self, user_text: str, assistant_text: str) -> None:
        turn = self._store.get_round_trip_count()
        chunks = self._chunker.chunk_turn(user_text, assistant_text, turn)
        cd = self._store.get_current()
        for chunk in chunks:
            proposals = self._classifier.classify(
                chunk, cd, llm_extract_fn=self._server_extract
            )
            self._builder.apply(cd, proposals)
        self._store.set_current(cd)

    def _run_evaluation(self, turn: int) -> None:
        start_turn = max(0, turn - self._eval_interval)
        eval_cd = self._eval_builder.build(
            self._recent_turns,
            self._store.get_current(),
            llm_extract_fn=self._server_extract,
            start_turn=start_turn,
        )
        self._store.set_eval(eval_cd)
        result = self._replacer.evaluate_and_replace()
        print(f"[Eval turn={turn}] replaced={result['replaced']} "
              f"current={result.get('current_score', {}).get('total', '?')} "
              f"eval={result.get('eval_score', {}).get('total', '?')}")

    def _server_extract(self, text: str) -> list[dict]:
        """サーバーのノード抽出エンドポイントを呼ぶ。"""
        try:
            resp = httpx.post(
                f"{self._server_url}/extract_nodes",
                json={"text": text},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json().get("nodes", [])
        except Exception:
            return []
