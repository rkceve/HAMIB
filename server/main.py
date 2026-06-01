"""
FastAPI server.

Endpoints:
  POST /chat           -- HAMIB inference (CD payload + user_text)
  POST /chat_baseline  -- plain Gemma inference (full history kept as context)
  POST /extract_nodes  -- extract node candidates from text
  GET  /health         -- connectivity check

Launch:
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

app = FastAPI(title="HAMIB LLM Server")

_gemma: MassWeightedGemma | None = None


@app.on_event("startup")
async def startup():
    global _gemma
    _gemma = MassWeightedGemma()
    _gemma.load()


# ── Request / Response models ─────────────────────────────────────────

class ServerMetrics(BaseModel):
    input_tokens: int
    inference_ms: float
    peak_memory_mb: float          # peak memory increase on the server side


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


# ── Helpers ──────────────────────────────────────────────────────────

def _measure_generate(prompt: str, use_m: bool = False, nodes=None, input_ids=None) -> tuple[str, ServerMetrics]:
    tokenizer = _gemma.tokenizer
    ids = input_ids if input_ids is not None else tokenizer.encode(prompt)
    input_tokens = len(ids)

    _gemma.clear_m_matrix()
    _gemma.clear_mass_vector()

    if use_m:
        # Use the 1D mass vector (decode-only, guarded by seq_q==1) rather than a
        # 2D M matrix applied at prefill+decode.
        pn_positions = find_pn_positions(ids, tokenizer)
        if pn_positions:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            vec = torch.zeros(len(ids), dtype=torch.float32, device=device)
            for pos, mass in pn_positions:
                if 0 <= pos < len(vec):
                    vec[pos] += mass
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


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": _gemma._model_id if _gemma else "not loaded"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """HAMIB mode: send only the correlation-diagram tokens plus the current message."""
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
    """Plain Gemma mode: stack the entire conversation history as context."""
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
