"""
MassWeightedGemma3n: wrapper for Gemma 3n (E2B-it).
  - Inherits the same sdpa monkey-patch as the Gemma 3 wrapper.
  - Gemma 3n is the multimodal Gemma3nForConditionalGeneration (text + vision + audio).
  - E2B-it = effective 2B params (Per-Layer Embedding runs a ~5B model as 2B).
  - ~1.5GB after 4-bit quantization; runs comfortably on a 6GB GPU.
"""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn.functional as F

# With accelerate hooks + torch._dynamo + meta device, compiling pre_forward
# triggers a NotImplementedError from meta tensor.to(), so disable dynamo entirely.
try:
    import torch._dynamo
    torch._dynamo.config.disable = True
except Exception:
    pass

from utils.config import load_config, get


class MassWeightedGemma3n:
    """Wrapper for Gemma 3n E2B-it."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        model_id: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        do_sample: bool | None = None,
    ):
        cfg = load_config(config_path) if config_path else load_config()
        server_cfg = cfg.get("server", {})

        self._model_id: str = model_id or "google/gemma-3n-E2B-it"
        self._quantization: str = server_cfg.get("quantization", "nf4")
        self._device: str = server_cfg.get("device", "cuda")
        self._max_new_tokens: int = max_new_tokens if max_new_tokens is not None else server_cfg.get("max_new_tokens", 512)
        self._temperature: float = temperature if temperature is not None else server_cfg.get("temperature", 0.7)
        self._do_sample: bool = do_sample if do_sample is not None else server_cfg.get("do_sample", True)
        self._mass_weight: float = get("attention", "mass_weight", 1.0)
        self._prefill_mass_scale: float = get("attention", "prefill_mass_scale", 0.0)

        self._qk_norm_mode: str = "off"
        self._qk_norm_alpha: float = 0.5
        self._qk_clip_threshold: float = 2.0

        self._model = None
        self._tokenizer = None
        self._m_matrix: torch.Tensor | None = None
        self._mass_vector: torch.Tensor | None = None
        self._original_sdpa = None

    def load(self) -> None:
        from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3nForConditionalGeneration

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self._quantization,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)

        # Gemma 3n E2B-it weights are 11GB in BF16
        # (vision_tower 548 keys + audio_tower 269 + LM 731).
        # vision/audio are CPU-offloaded as real tensors (not meta) while the LM
        # stays on GPU, so hook-based tensor movement works during generate().
        print("[MassWeightedGemma3n] loading Gemma3nForConditionalGeneration (LM=cuda, vision/audio=cpu) ...")
        max_memory = {0: "4500MiB", "cpu": "24GiB"}
        self._model = Gemma3nForConditionalGeneration.from_pretrained(
            self._model_id,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        self._model.eval()
        self._patch_sdpa()
        print(f"[MassWeightedGemma3n] loaded: {self._model_id}")

    # Unused in the default path: no caller invokes set_m_matrix(). The
    # 2D M-matrix mode is kept for short-context experiments only. The live
    # path uses set_mass_vector() (1D) below.
    def set_m_matrix(self, M: torch.Tensor) -> None:
        self._m_matrix = M

    def clear_m_matrix(self) -> None:
        self._m_matrix = None

    def set_mass_vector(self, v: torch.Tensor) -> None:
        self._mass_vector = v

    def clear_mass_vector(self) -> None:
        self._mass_vector = None

    @property
    def tokenizer(self):
        return self._tokenizer

    def generate(self, prompt: str) -> str:
        # accelerate offload may place embed_tokens on the CPU, so align
        # input_ids with the embedding layer's device.
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            embed = self._model.get_input_embeddings()
            target_device = next(embed.parameters()).device
        except Exception:
            pass
        inputs = self._tokenizer(prompt, return_tensors="pt").to(target_device)
        input_ids = inputs["input_ids"]

        gen_kwargs: dict = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": self._do_sample,
        }
        if self._do_sample:
            gen_kwargs["temperature"] = self._temperature

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_ids = output_ids[0, input_ids.shape[1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True)

    def _patch_sdpa(self) -> None:
        outer = self
        self._original_sdpa = F.scaled_dot_product_attention

        def patched_sdpa(
            query, key, value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            **kwargs,
        ):
            qk_mode = outer._qk_norm_mode
            if qk_mode != "off":
                d_head = query.shape[-1]
                scale_factor = d_head ** 0.5
                if qk_mode == "l2":
                    query = F.normalize(query.float(), p=2, dim=-1).to(query.dtype) * scale_factor
                    key = F.normalize(key.float(), p=2, dim=-1).to(key.dtype) * scale_factor
                elif qk_mode == "l2_soft":
                    alpha = outer._qk_norm_alpha
                    q_norm = F.normalize(query.float(), p=2, dim=-1).to(query.dtype) * scale_factor
                    k_norm = F.normalize(key.float(), p=2, dim=-1).to(key.dtype) * scale_factor
                    query = alpha * q_norm + (1.0 - alpha) * query
                    key = alpha * k_norm + (1.0 - alpha) * key
                elif qk_mode == "clip":
                    max_norm = outer._qk_clip_threshold * scale_factor
                    q_norms = query.float().norm(dim=-1, keepdim=True).clamp(min=1e-9)
                    k_norms = key.float().norm(dim=-1, keepdim=True).clamp(min=1e-9)
                    q_scale = (max_norm / q_norms).clamp(max=1.0).to(query.dtype)
                    k_scale = (max_norm / k_norms).clamp(max=1.0).to(key.dtype)
                    query = query * q_scale
                    key = key * k_scale

            seq_q = query.shape[-2]
            seq_k = key.shape[-2]
            m_bias = None

            M = outer._m_matrix
            mass_vec = outer._mass_vector

            if M is not None:
                # Unused in the default path: outer._m_matrix is never set.
                # 2D M-matrix mode for short-context experiments only.
                m_q = min(seq_q, M.shape[0])
                m_k = min(seq_k, M.shape[1])
                m_slice = outer._mass_weight * M[:m_q, :m_k]
                if m_q < seq_q or m_k < seq_k:
                    full = torch.zeros(seq_q, seq_k, dtype=torch.float32, device=M.device)
                    full[:m_q, :m_k] = m_slice
                    m_slice = full
                m_bias = m_slice.to(dtype=query.dtype, device=query.device).unsqueeze(0).unsqueeze(0)

            elif mass_vec is not None:
                effective_w: float | None = None
                if seq_q == 1:
                    effective_w = outer._mass_weight
                elif outer._prefill_mass_scale > 0.0:
                    effective_w = outer._mass_weight * outer._prefill_mass_scale

                if effective_w is not None:
                    m_k = min(seq_k, mass_vec.shape[0])
                    m_vec = effective_w * mass_vec[:m_k]
                    if m_k < seq_k:
                        full_vec = torch.zeros(seq_k, dtype=torch.float32, device=mass_vec.device)
                        full_vec[:m_k] = m_vec
                        m_vec = full_vec
                    m_bias = m_vec.to(dtype=query.dtype, device=query.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)

            if m_bias is not None:
                if attn_mask is None:
                    if is_causal:
                        causal_mask = _make_causal_mask(seq_q, seq_k, query.dtype, query.device)
                        m_bias = m_bias + causal_mask
                        is_causal = False
                    attn_mask = m_bias
                else:
                    try:
                        attn_mask = attn_mask.to(dtype=query.dtype) + m_bias
                    except RuntimeError:
                        pass

            return outer._original_sdpa(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                **kwargs,
            )

        import torch.nn.functional as _F
        _F.scaled_dot_product_attention = patched_sdpa
        torch.nn.functional.scaled_dot_product_attention = patched_sdpa
        print("[MassWeightedGemma3n] patched scaled_dot_product_attention")

    def restore_sdpa(self) -> None:
        if self._original_sdpa is not None:
            import torch.nn.functional as _F
            _F.scaled_dot_product_attention = self._original_sdpa
            torch.nn.functional.scaled_dot_product_attention = self._original_sdpa

    @property
    def attn_implementation(self) -> str | None:
        if self._model is None:
            return None
        cfg = getattr(self._model, "config", None)
        if cfg is None:
            return None
        text_cfg = getattr(cfg, "text_config", None) or cfg
        return getattr(text_cfg, "_attn_implementation", None)


def _make_causal_mask(seq_q, seq_k, dtype, device):
    mask = torch.full((seq_q, seq_k), float("-inf"), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=seq_k - seq_q + 1)
    return mask.unsqueeze(0).unsqueeze(0)
