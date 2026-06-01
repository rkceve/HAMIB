"""
MassWeightedGemma (= MassWeightedLLM): implements "attn_scores += w * M".

Model-agnostic:
  Uses AutoModelForCausalLM + AutoTokenizer, so it works with any Causal LM
  on HuggingFace. Verified with:
    - google/gemma-3-4b-it
    - THUDM/glm-4-9b-chat
    - Qwen/Qwen2.5-7B-Instruct
    - meta-llama/Llama-3.1-8B-Instruct

Injection method:
  Monkey-patches torch.nn.functional.scaled_dot_product_attention.
  That function adds attn_mask (a float tensor) to the logits before softmax,
  so passing M as attn_mask is equivalent to "scores += w * M".

  The transformers library uses the sdpa backend for many architectures, so
  the injection works without changing any model-specific code. It is
  guaranteed to work when `model.config._attn_implementation` is "sdpa".

Usage:
  model = MassWeightedGemma(model_id="THUDM/glm-4-9b-chat")
  model.load()
  M = m_matrix_builder.build(...)
  model.set_m_matrix(M)
  output = model.generate(prompt)
  model.clear_m_matrix()
"""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn.functional as F

from utils.config import load_config, get


class MassWeightedGemma:
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

        self._model_id: str = model_id or server_cfg.get("model_id", "google/gemma-3-4b-it")
        self._quantization: str = server_cfg.get("quantization", "nf4")
        self._device: str = server_cfg.get("device", "cuda")
        self._max_new_tokens: int = max_new_tokens if max_new_tokens is not None else server_cfg.get("max_new_tokens", 512)
        self._temperature: float = temperature if temperature is not None else server_cfg.get("temperature", 0.7)
        self._do_sample: bool = do_sample if do_sample is not None else server_cfg.get("do_sample", True)
        self._mass_weight: float = get("attention", "mass_weight", 1.0)
        # Mass addition scale during prefill (default 0.0 = no mass added during prefill).
        # Raising it to 0.1 or 0.5 also adds mass_vec during prefill.
        # 1.0 adds full mass during prefill, but this can cause representation collapse.
        self._prefill_mass_scale: float = get("attention", "prefill_mass_scale", 0.0)

        # QK-norm retrofit mode for the patched attention.
        # "off"     = no Q/K modification
        # "l2"      = full L2 normalize + sqrt(head_dim) re-scale
        # "l2_soft" = alpha-mix: query = alpha * L2-normalized + (1-alpha) * original
        # "clip"    = shrink outliers only; cap at max_norm = threshold * sqrt(head_dim)
        self._qk_norm_mode: str = "off"
        self._qk_norm_alpha: float = 0.5      # for "l2_soft"
        self._qk_clip_threshold: float = 2.0  # for "clip"

        self._model = None
        self._tokenizer = None
        self._m_matrix: torch.Tensor | None = None
        self._mass_vector: torch.Tensor | None = None  # 1D mass vector (for long context)
        self._original_sdpa = None

    def load(self) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self._quantization,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self._model.eval()
        self._patch_sdpa()
        print(f"[MassWeightedGemma] loaded: {self._model_id}")

    # Unused in the default path: nothing in this repository calls
    # set_m_matrix(). The live HAMIBSession path uses set_mass_vector() (1D)
    # instead. The 2D M-matrix mode is kept so short-context experiments can
    # opt in.
    def set_m_matrix(self, M: torch.Tensor) -> None:
        self._m_matrix = M

    # clear_m_matrix() is called by the default path but only as a no-op
    # safety (sets the already-None _m_matrix back to None).
    def clear_m_matrix(self) -> None:
        self._m_matrix = None

    def set_mass_vector(self, v: torch.Tensor) -> None:
        """Set the 1D mass vector (for long context, shape: (seq_len,))."""
        self._mass_vector = v

    def clear_mass_vector(self) -> None:
        self._mass_vector = None

    @property
    def tokenizer(self):
        return self._tokenizer

    # ── Inference ─────────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        # In case the model is split across devices by device_map="auto",
        # place the input on the device of the first embedding layer.
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

    # ── Core logic: patching scaled_dot_product_attention ─────────────

    def _patch_sdpa(self) -> None:
        """
        Replace torch.nn.functional.scaled_dot_product_attention.

        attn_mask is added to the logits before softmax, so passing M as
        attn_mask realizes
            attn_scores += w * M
        """
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
            # QK-norm retrofit: suppresses the massive-value outliers in Q/K
            # (Sun et al. 2025: arXiv:2502.01563), keeping the pre-softmax
            # logits compact so the additive bias has more leverage.
            qk_mode = outer._qk_norm_mode
            if qk_mode != "off":
                d_head = query.shape[-1]
                scale_factor = d_head ** 0.5

                if qk_mode == "l2":
                    # Full L2 norm + sqrt(d) re-scale.
                    query = F.normalize(query.float(), p=2, dim=-1).to(query.dtype) * scale_factor
                    key = F.normalize(key.float(), p=2, dim=-1).to(key.dtype) * scale_factor

                elif qk_mode == "l2_soft":
                    # alpha-mix: partial normalize to keep coherence while suppressing outliers.
                    alpha = outer._qk_norm_alpha
                    q_norm = F.normalize(query.float(), p=2, dim=-1).to(query.dtype) * scale_factor
                    k_norm = F.normalize(key.float(), p=2, dim=-1).to(key.dtype) * scale_factor
                    query = alpha * q_norm + (1.0 - alpha) * query
                    key = alpha * k_norm + (1.0 - alpha) * key

                elif qk_mode == "clip":
                    # Shrink outliers only: cap at max_norm = threshold * sqrt(d_head).
                    # Normal Q/K is left unchanged for a minimally invasive retrofit.
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
                # Unused in the default path: outer._m_matrix is never set
                # (no caller of set_m_matrix() exists). This branch only
                # fires if a short-context experiment opts in explicitly.
                # 2D M-matrix mode (for short context).
                m_q = min(seq_q, M.shape[0])
                m_k = min(seq_k, M.shape[1])
                m_slice = outer._mass_weight * M[:m_q, :m_k]

                # Zero-pad if KV-cache growth exceeds the M-matrix size.
                if m_q < seq_q or m_k < seq_k:
                    full = torch.zeros(seq_q, seq_k, dtype=torch.float32, device=M.device)
                    full[:m_q, :m_k] = m_slice
                    m_slice = full

                # (seq_q, seq_k) -> (1, 1, seq_q, seq_k)
                m_bias = m_slice.to(dtype=query.dtype, device=query.device).unsqueeze(0).unsqueeze(0)

            elif mass_vec is not None:
                # 1D mass-vector mode (for long context, memory efficient).
                # By default mass is applied only on decode steps (seq_q==1).
                # Adding full mass_weight during prefill can collapse the
                # representation and produce repeated output, so prefill is
                # gated: when prefill_mass_scale > 0.0, mass is applied during
                # prefill at a partial scale.
                effective_w: float | None = None
                if seq_q == 1:
                    # Decode step: apply full weight.
                    effective_w = outer._mass_weight
                elif outer._prefill_mass_scale > 0.0:
                    # Prefill: apply partial scale.
                    effective_w = outer._mass_weight * outer._prefill_mass_scale

                if effective_w is not None:
                    m_k = min(seq_k, mass_vec.shape[0])
                    m_vec = effective_w * mass_vec[:m_k]

                    if m_k < seq_k:
                        full_vec = torch.zeros(seq_k, dtype=torch.float32, device=mass_vec.device)
                        full_vec[:m_k] = m_vec
                        m_vec = full_vec

                    # (seq_k,) -> (1, 1, 1, seq_k) for broadcasting
                    m_bias = m_vec.to(dtype=query.dtype, device=query.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)

            if m_bias is not None:
                if attn_mask is None:
                    # is_causal=True cannot be combined with a float attn_mask,
                    # so drop the is_causal flag and fold the causal mask into m_bias.
                    if is_causal:
                        causal_mask = _make_causal_mask(seq_q, seq_k, query.dtype, query.device)
                        m_bias = m_bias + causal_mask
                        is_causal = False
                    attn_mask = m_bias
                else:
                    try:
                        attn_mask = attn_mask.to(dtype=query.dtype) + m_bias
                    except RuntimeError:
                        pass  # skip if shapes do not match

            return outer._original_sdpa(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                **kwargs,
            )

        # Replace both torch.nn.functional and F references.
        import torch.nn.functional as _F
        _F.scaled_dot_product_attention = patched_sdpa
        torch.nn.functional.scaled_dot_product_attention = patched_sdpa
        print("[MassWeightedGemma] patched scaled_dot_product_attention")

    def restore_sdpa(self) -> None:
        """Restore the original sdpa (e.g. for tests)."""
        if self._original_sdpa is not None:
            import torch.nn.functional as _F
            _F.scaled_dot_product_attention = self._original_sdpa
            torch.nn.functional.scaled_dot_product_attention = self._original_sdpa

    # ── Compatibility check ───────────────────────────────────────────

    @property
    def attn_implementation(self) -> str | None:
        """Return the attention implementation name the model uses
        (sdpa / eager / flash_attention_2, etc.).
        The mass-injection patch takes effect when it is sdpa."""
        if self._model is None:
            return None
        cfg = getattr(self._model, "config", None)
        if cfg is None:
            return None
        return getattr(cfg, "_attn_implementation", None)


# Alias so new code can import this class as MassWeightedLLM.
MassWeightedLLM = MassWeightedGemma


# ── Utilities ─────────────────────────────────────────────────────────

def _make_causal_mask(
    seq_q: int, seq_k: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Causal mask used in place of is_causal=True (masks the future with -inf)."""
    mask = torch.full((seq_q, seq_k), float("-inf"), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=seq_k - seq_q + 1)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_q, seq_k)
