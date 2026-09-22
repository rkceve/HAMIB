"""Minimal OpenAI-compatible chat server for the summarizer (transformers, batch 1).

Why: the pod's driver is CUDA 12.8 and every vLLM wheel that supports Qwen3.5 links CUDA 13
(DECISIONS H27). This serves exactly the subset of `POST /v1/chat/completions` that
`benchmark.mcbuild_bench.summarizer_client.SummarizerClient` uses: `model`, `messages`,
`max_tokens`, `temperature` (0 → greedy), `chat_template_kwargs.enable_thinking`; the reply carries
`choices[0].message.content`, `choices[0].finish_reason` ("stop" | "length") and
`usage.prompt_tokens` / `usage.completion_tokens` (real token counts). `GET /v1/models` lists the
served name so the pod scripts can wait for readiness.

Run (reader venv):
    python -m benchmark.mcbuild_bench.pod.summarizer_server --model-id Qwen/Qwen3.5-4B --port 8123 --served-model-name summarizer
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch


class Engine:
    def __init__(self, model_id: str, dtype: str = "bfloat16", device: str | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, dtype), device_map=self.device, attn_implementation="sdpa"
        )
        self.model.eval()
        self.lock = threading.Lock()  # batch 1: one generation at a time
        self.eos_ids = set()
        gen = self.model.generation_config
        eos = gen.eos_token_id
        if isinstance(eos, int):
            self.eos_ids.add(eos)
        elif eos:
            self.eos_ids.update(int(x) for x in eos)
        if self.tokenizer.eos_token_id is not None:
            self.eos_ids.add(int(self.tokenizer.eos_token_id))

    def chat(self, messages: list[dict], max_tokens: int, temperature: float, enable_thinking: bool) -> dict:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        prompt_tokens = int(enc["input_ids"].shape[1])
        kwargs: dict = {"max_new_tokens": int(max_tokens), "do_sample": temperature > 0}
        if temperature > 0:
            kwargs["temperature"] = float(temperature)
        with self.lock, torch.no_grad():
            t0 = time.perf_counter()
            out = self.model.generate(**enc, **kwargs)
            ms = (time.perf_counter() - t0) * 1000.0
        new_ids = out[0, prompt_tokens:]
        completion_tokens = int(new_ids.shape[0])
        finish = "stop" if (completion_tokens < max_tokens or (completion_tokens and int(new_ids[-1]) in self.eos_ids)) else "length"
        content = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return {
            "content": content,
            "finish_reason": finish,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": ms,
        }


def make_handler(engine: Engine, served_name: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": served_name, "object": "model", "owned_by": "local"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send(404, {"error": "not found"})
                return
            n = int(self.headers.get("Content-Length", "0"))
            try:
                req = json.loads(self.rfile.read(n).decode("utf-8"))
                messages = req["messages"]
                max_tokens = int(req.get("max_tokens", 96))
                temperature = float(req.get("temperature", 0))
                enable_thinking = bool((req.get("chat_template_kwargs") or {}).get("enable_thinking", False))
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"error": f"bad request: {exc}"})
                return
            try:
                r = engine.chat(messages, max_tokens, temperature, enable_thinking)
            except Exception as exc:  # noqa: BLE001 -- the client must get a status, never a dropped connection
                self._send(500, {"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
                return
            self._send(200, {
                "id": f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "model": served_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": r["content"]}, "finish_reason": r["finish_reason"]}],
                "usage": {"prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                          "total_tokens": r["prompt_tokens"] + r["completion_tokens"]},
                "latency_ms": r["latency_ms"],
            })

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--served-model-name", default="summarizer")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    engine = Engine(args.model_id, dtype=args.dtype)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine, args.served_model_name))
    print(f"summarizer_server: {args.model_id} on 127.0.0.1:{args.port} as {args.served_model_name} ({engine.device})", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
