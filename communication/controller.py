"""
Controller: HTTP client-side orchestrator (reference implementation showing
the server / management split — see README "Architecture" section).

Status: UNWIRED. Nothing in this repository imports ``Controller``. The
default in-process orchestrator used by the benchmarks and ``server/main.py``
is ``server/hamib_session.py:HAMIBSession``, which runs the same management and
evaluation flow without a network hop. Controller is kept to document the
client / server boundary: the same management pipeline could be deployed as
a separate service that POSTs to the FastAPI inference server.

Responsibilities (when wired in):
  1. Send user input to the server (with the CD payload).
  2. Receive the server response and update the CD via the management unit.
  3. Trigger the evaluation unit every 5 round trips.
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
        client_cfg = cfg.get("client", {})
        host = client_cfg.get("host", "localhost")
        port = client_cfg.get("server_port", 8080)
        self._server_url = f"http://{host}:{port}"
        self._eval_interval: int = get("evaluation", "eval_interval_rounds", 5)

        self._recent_turns: list[tuple[str, str]] = []

    def chat(self, user_text: str) -> str:
        """
        Process the user's utterance and return the assistant's reply string.
        """
        cd = self._store.get_current()
        payload = self._serializer.to_api_payload(cd)

        # Send the request to the server
        response_text = self._call_server(user_text, payload)

        # Update the CD (management unit)
        self._update_cd(user_text, response_text)

        # Round-trip bookkeeping
        turn = self._store.increment_round_trip()
        self._recent_turns.append((user_text, response_text))
        if len(self._recent_turns) > self._eval_interval:
            self._recent_turns.pop(0)

        # Evaluation unit (every 5 rounds)
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
        """Call the server's node extraction endpoint."""
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
