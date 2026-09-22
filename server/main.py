"""
FastAPI サーバー（友達PC で動かす）

エンドポイント:
  POST /chat           -- CMS推論 (CDペイロード + user_text)
  POST /chat_baseline  -- 通常Gemma推論（全履歴をそのままコンテキストに）
  POST /extract_nodes  -- テキストからノード候補抽出
  GET  /health         -- 疎通確認

起動:
  cd cms_prototype
  uvicorn server.main:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations
import json
import sys
import time
import tracemalloc
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from server.mass_weighted_gemma import MassWeightedGemma
from server.cd_parser import parse_node_list, extract_nodes_prompt, find_pn_positions
from server.mass_vector import positions_to_mass_vector
from utils.config import load_config, get

app = FastAPI(title="CMS LLM Server")

_gemma: MassWeightedGemma | None = None


@app.on_event("startup")
async def startup():
    global _gemma
    _gemma = MassWeightedGemma()
    _gemma.load()


# ── Request / Response モデル ─────────────────────────────────────────

class ServerMetrics(BaseModel):
    input_tokens: int
    inference_ms: float
    peak_memory_mb: float          # サーバー側ピークメモリ増分


class ChatRequest(BaseModel):
    user_text: str
    node_list: list[dict] = []
    context_block: str = ""


class ChatResponse(BaseModel):
    response: str
    metrics: ServerMetrics


class BaselineChatRequest(BaseModel):
    user_text: str
    history: list[dict] = []       # [{"role": "user"|"assistant", "content": "..."}]


class BaselineChatResponse(BaseModel):
    response: str
    metrics: ServerMetrics


class ExtractRequest(BaseModel):
    text: str


class ExtractResponse(BaseModel):
    nodes: list[dict]


# ── ヘルパー ──────────────────────────────────────────────────────────

def _measure_generate(prompt: str, use_m: bool = False, nodes=None, input_ids=None) -> tuple[str, ServerMetrics]:
    tokenizer = _gemma.tokenizer
    ids = input_ids if input_ids is not None else tokenizer.encode(prompt)
    input_tokens = len(ids)

    _gemma.clear_m_matrix()
    _gemma.clear_mass_vector()

    if use_m:
        # 実験L検証済み: 2D M行列（prefill+decode適用）は0%に崩壊するため廃止。
        # 1D マスベクトル（decode専用, seq_q==1 ガード）を使用する。
        pn_positions = find_pn_positions(ids, tokenizer)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # F3: D-3 の上限 min(cap, mass*scale) と max による衝突解決は
        # server/mass_vector.py に一本化されている（手書きの += ループは廃止）。
        vec = positions_to_mass_vector(
            pn_positions,
            len(ids),
            cap=float(get("attention", "mass_cap", 3.0)),
            scale=float(get("attention", "mass_scale", 1.0)),
            device=device,
        )
        if vec is not None:
            _gemma.set_mass_vector(vec)

    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        response_text = _gemma.generate(prompt)
    finally:
        _gemma.clear_mass_vector()
        _gemma.clear_m_matrix()
    inference_ms = (time.perf_counter() - t0) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics = ServerMetrics(
        input_tokens=input_tokens,
        inference_ms=round(inference_ms, 1),
        peak_memory_mb=round(peak / 1024 / 1024, 2),
    )
    return response_text, metrics


# ── エンドポイント ────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": _gemma._model_id if _gemma else "not loaded"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """CMS方式: 相関図トークン + 現在のメッセージのみ送信。"""
    nodes = parse_node_list(req.node_list)

    prompt = ""
    if req.context_block:
        prompt += req.context_block + "\n\n"
    prompt += f"User: {req.user_text}\nAssistant:"

    tokenizer = _gemma.tokenizer
    input_ids = tokenizer.encode(prompt)
    response_text, metrics = _measure_generate(
        prompt, use_m=True, nodes=nodes, input_ids=input_ids
    )
    return ChatResponse(response=response_text, metrics=metrics)


@app.post("/chat_baseline", response_model=BaselineChatResponse)
def chat_baseline(req: BaselineChatRequest):
    """通常Gemma方式: 全会話履歴をそのままコンテキストに積む。"""
    history_block = ""
    for turn in req.history:
        role = "User" if turn["role"] == "user" else "Assistant"
        history_block += f"{role}: {turn['content']}\n"
    prompt = history_block + f"User: {req.user_text}\nAssistant:"

    response_text, metrics = _measure_generate(prompt, use_m=False)
    return BaselineChatResponse(response=response_text, metrics=metrics)


@app.post("/extract_nodes", response_model=ExtractResponse)
def extract_nodes(req: ExtractRequest):
    extraction_prompt = extract_nodes_prompt(req.text)
    _gemma.clear_m_matrix()
    raw = _gemma.generate(extraction_prompt)

    nodes: list[dict] = []
    try:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            nodes = json.loads(raw[start:end])
    except (json.JSONDecodeError, ValueError):
        pass

    return ExtractResponse(nodes=nodes)
