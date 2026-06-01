"""
BenchmarkRunner: runs both systems (HAMIB / Baseline) under the same conditions and collects metrics.

Metrics (server side only):
  - input_tokens    : number of tokens passed to inference
  - inference_ms    : inference time (ms)
  - peak_memory_mb  : peak memory increase (MB)

Metrics (client side, separate):
  - cd_build_ms     : correlation-diagram build time (ms)  (HAMIB only)

Consistency score:
  - recall_hit      : whether the correct keyword appears in the recall-turn response (0/1)
"""
from __future__ import annotations
import time
import httpx
from dataclasses import dataclass, field
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.dataset import Turn, BENCHMARK_TURNS
from store.cd_store import CDStore
from communication.cd_serializer import CDSerializer
from utils.config import load_config


@dataclass
class TurnResult:
    turn_id: int
    turn_type: str
    # Server-side metrics
    input_tokens: int
    inference_ms: float
    peak_memory_mb: float
    # Client side (non-zero for HAMIB only)
    cd_build_ms: float
    # Consistency
    response: str
    recall_hit: int   # 0 or 1 (meaningful for recall turns only, otherwise -1)
    fact_value: str   # expected correct keyword


@dataclass
class BenchmarkResult:
    mode: str          # "hamib" | "baseline"
    turns: list[TurnResult] = field(default_factory=list)


class BenchmarkRunner:
    def __init__(self, server_url: str | None = None):
        cfg = load_config()
        client_cfg = cfg.get("client", {})
        host = client_cfg.get("host", "localhost")
        port = client_cfg.get("server_port", 8080)
        self._server_url = server_url or f"http://{host}:{port}"
        self._http = httpx.Client(timeout=120.0)

    # -- HAMIB mode -------------------------------------------------------

    def run_hamib(self) -> BenchmarkResult:
        result = BenchmarkResult(mode="hamib")
        store = CDStore()
        serializer = CDSerializer()

        # Measure CD update time manually instead of using controller._update_cd
        from management.text_chunker import TextChunker
        from management.node_classifier import NodeClassifier
        from management.graph_builder import GraphBuilder
        chunker = TextChunker()
        classifier = NodeClassifier()
        builder = GraphBuilder()

        assistant_response = ""

        for turn in BENCHMARK_TURNS:
            print(f"  [HAMIB] turn {turn.turn_id:2d} ({turn.type}) ...", end=" ", flush=True)

            cd = store.get_current()
            payload = serializer.to_api_payload(cd)

            # HAMIB inference request to the server
            resp = self._http.post(
                f"{self._server_url}/chat",
                json={
                    "user_text": turn.user,
                    "node_list": payload["node_list"],
                    "context_block": payload["context_block"],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            assistant_response = data["response"]
            m = data["metrics"]

            # Client side: measure CD update time
            t0 = time.perf_counter()
            chunks = chunker.chunk_turn(turn.user, assistant_response, turn.turn_id)
            for chunk in chunks:
                proposals = classifier.classify(
                    chunk, cd, llm_extract_fn=self._server_extract
                )
                builder.apply(cd, proposals)
            store.set_current(cd)
            cd_build_ms = (time.perf_counter() - t0) * 1000

            recall_hit, fact_value = self._eval_recall(turn, assistant_response)
            result.turns.append(TurnResult(
                turn_id=turn.turn_id,
                turn_type=turn.type,
                input_tokens=m["input_tokens"],
                inference_ms=m["inference_ms"],
                peak_memory_mb=m["peak_memory_mb"],
                cd_build_ms=round(cd_build_ms, 1),
                response=assistant_response,
                recall_hit=recall_hit,
                fact_value=fact_value,
            ))
            print(f"tokens={m['input_tokens']}  {m['inference_ms']:.0f}ms  recall={recall_hit}")

        return result

    # -- Baseline mode --------------------------------------------------

    def run_baseline(self) -> BenchmarkResult:
        result = BenchmarkResult(mode="baseline")
        history: list[dict] = []

        for turn in BENCHMARK_TURNS:
            print(f"  [Baseline] turn {turn.turn_id:2d} ({turn.type}) ...", end=" ", flush=True)

            resp = self._http.post(
                f"{self._server_url}/chat_baseline",
                json={"user_text": turn.user, "history": history},
            )
            resp.raise_for_status()
            data = resp.json()
            assistant_response = data["response"]
            m = data["metrics"]

            # Accumulate the full history
            history.append({"role": "user", "content": turn.user})
            history.append({"role": "assistant", "content": assistant_response})

            recall_hit, fact_value = self._eval_recall(turn, assistant_response)
            result.turns.append(TurnResult(
                turn_id=turn.turn_id,
                turn_type=turn.type,
                input_tokens=m["input_tokens"],
                inference_ms=m["inference_ms"],
                peak_memory_mb=m["peak_memory_mb"],
                cd_build_ms=0.0,
                response=assistant_response,
                recall_hit=recall_hit,
                fact_value=fact_value,
            ))
            print(f"tokens={m['input_tokens']}  {m['inference_ms']:.0f}ms  recall={recall_hit}")

        return result

    # -- Helpers --------------------------------------------------------

    def _server_extract(self, text: str) -> list[dict]:
        try:
            resp = self._http.post(
                f"{self._server_url}/extract_nodes", json={"text": text}
            )
            resp.raise_for_status()
            return resp.json().get("nodes", [])
        except Exception:
            return []

    @staticmethod
    def _eval_recall(turn: Turn, response: str) -> tuple[int, str]:
        if turn.type != "recall":
            return -1, ""
        hit = 1 if turn.fact_value in response else 0
        return hit, turn.fact_value

    def close(self):
        self._http.close()
