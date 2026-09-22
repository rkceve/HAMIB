"""
MassWeightedGemma4: Gemma 4 (E4B-it) 用ラッパー
  - Gemma 3 と同じ sdpa monkey-patch を継承
  - Gemma 4 は Gemma4ForConditionalGeneration (multimodal) なので text-only path で load
  - 42 layers (35 sliding + 7 full attention)、 128K context、 BF16 native

【Gemma 3 との違い】
  - model_type: gemma4 / class: Gemma4ForConditionalGeneration
  - text-only 推論: text_config を直接使用 (vision/audio embedding skip)
  - sliding_window=512 (35/42 layers が sliding、7 layers が full)
  - vocab_size 262144 (Gemma 3: 256K)
  - default torch_dtype: bfloat16

【§77 までの修正は引継ぐ】
  - max_sun_nodes=10000 (silent drop 解消)
  - node_classifier の強制 SUN 昇格廃止
  - sdpa monkey-patch (Phi-3 fix の leading-token-drop も継承)
"""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn.functional as F

from utils.config import load_config, get


class MassWeightedGemma4:
    """Gemma 4 E4B-it 用 wrapper (Gemma 3 wrapper の薄い派生)"""

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

        self._model_id: str = model_id or "google/gemma-4-E4B-it"
        self._quantization: str = server_cfg.get("quantization", "nf4")
        self._device: str = server_cfg.get("device", "cuda")
        self._max_new_tokens: int = max_new_tokens if max_new_tokens is not None else server_cfg.get("max_new_tokens", 512)
        self._temperature: float = temperature if temperature is not None else server_cfg.get("temperature", 0.7)
        self._do_sample: bool = do_sample if do_sample is not None else server_cfg.get("do_sample", True)
        self._mass_weight: float = get("attention", "mass_weight", 1.0)
        self._prefill_mass_scale: float = get("attention", "prefill_mass_scale", 0.0)

        # §53 Exp Q QK-norm retrofit 設定継承 (off がデフォルト)
        self._qk_norm_mode: str = "off"
        self._qk_norm_alpha: float = 0.5
        self._qk_clip_threshold: float = 2.0

        self._model = None
        self._tokenizer = None
        self._m_matrix: torch.Tensor | None = None
        self._mass_vector: torch.Tensor | None = None
        self._original_sdpa = None

    def load(self) -> None:
        from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma4ForConditionalGeneration

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self._quantization,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # Gemma 4 native dtype
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)

        # Gemma 4 E4B-it のチェックポイントは multimodal 配置 (model.language_model.layers.*)
        # Gemma4ForCausalLM では layer key mismatch で全 LM 重みが UNEXPECTED → random init
        # → bnb FP4 quant state が空のまま forward で AssertionError (§78 第1原因)
        # 結論: Gemma4ForConditionalGeneration で読むしかない
        # device_map="auto" 無制約だと weight 量子化中に OOM 静死 (§78 第3原因)
        # post-load strip ループは accelerate dispatch を遅延させる (§78 第4原因)
        # → max_memory で GPU を制限し、 strip は省略
        print(f"[MassWeightedGemma4] loading multimodal Gemma4ForConditionalGeneration (max_memory cap) ...")
        max_memory = {
            0: "4500MiB",       # cuda:0 — language_model のみ載せる
            "cpu": "24GiB",     # vision_tower + audio_tower 退避先
        }
        self._model = Gemma4ForConditionalGeneration.from_pretrained(
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
        print(f"[MassWeightedGemma4] loaded: {self._model_id}")

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
        target_device = self._device
        try:
            target_device = next(self._model.parameters()).device
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
        """sdpa monkey-patch (Gemma 3 wrapper と同じ実装、 Gemma 4 でも attention path で動く)"""
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
            # QK-norm retrofit (off default)
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
        print("[MassWeightedGemma4] patched scaled_dot_product_attention")

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
        # Gemma 4 では text_config.attn_implementation を確認
        text_cfg = getattr(cfg, "text_config", None) or cfg
        return getattr(text_cfg, "_attn_implementation", None)


MassWeightedLLM4 = MassWeightedGemma4


def _make_causal_mask(seq_q, seq_k, dtype, device):
    mask = torch.full((seq_q, seq_k), float("-inf"), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=seq_k - seq_q + 1)
    return mask.unsqueeze(0).unsqueeze(0)
