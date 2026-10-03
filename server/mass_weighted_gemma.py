"""
Mass-weighted attention for Hugging Face causal LMs: ``attn_scores += w * M``.

Works with any causal LM loaded through ``AutoModelForCausalLM`` +
``AutoTokenizer``. Tested with google/gemma-3-4b-it, THUDM/glm-4-9b-chat,
Qwen/Qwen2.5-7B-Instruct and meta-llama/Llama-3.1-8B-Instruct.

How it works: ``torch.nn.functional.scaled_dot_product_attention`` is
monkey-patched. SDPA adds a float ``attn_mask`` to the logits before the
softmax, so merging ``w * M`` into the mask is exactly ``scores += w * M``.
transformers routes most architectures through SDPA, so no model code is
changed; the model must run with ``config._attn_implementation == "sdpa"``.

Usage:
  model = MassWeightedGemma(model_id="THUDM/glm-4-9b-chat")
  model.load()
  M = m_matrix_builder.build(...)
  model.set_m_matrix(M)          # or set_mass_vector(v) for long contexts
  output = model.generate(prompt)
  model.clear_m_matrix()
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import time
import torch
import torch.nn.functional as F

from utils.config import load_config, get

# The unpatched torch kernel, captured once at import time. Every wrapper calls
# this rather than the current global, so patching twice can never produce a
# wrapper that calls itself. ``_PATCH_OWNER`` is the instance whose wrapper is
# installed (None when unpatched); only that instance may restore the original.
_ORIGINAL_SDPA = F.scaled_dot_product_attention
_PATCH_OWNER: "MassWeightedGemma | None" = None

# The attention recorder recomputes a float32 softmax row per layer, which is
# only affordable for short probes; longer key sequences are refused.
RECORD_MAX_KEYS = 4096


@lru_cache(maxsize=1)
def _first_token_timer_class():
    """Build the FirstTokenTimer class on first use.

    transformers is imported lazily (as in load()) so that importing this
    module needs only torch; verify_attention_math.py relies on that.
    """
    from transformers import LogitsProcessor

    class FirstTokenTimer(LogitsProcessor):
        """Records ``time.perf_counter()`` on its first call; scores pass through unchanged.

        ``generate()`` runs the logits processors after the prefill forward and
        before the first token is sampled, so the first call marks the
        prefill/decode boundary.
        """

        def __init__(self, on_first_call=None) -> None:
            self.first_call_t: float | None = None
            # Runs once, on the first logits call, i.e. after the last prefill
            # forward (chunked or not); the owner flips its phase flag here.
            self._on_first_call = on_first_call

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            if self.first_call_t is None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.first_call_t = time.perf_counter()
                if self._on_first_call is not None:
                    self._on_first_call()
            return scores

    return FirstTokenTimer


def make_first_token_timer(on_first_call=None):
    """A new ``FirstTokenTimer`` (a ``transformers.LogitsProcessor``).

    ``on_first_call`` (optional, no arguments) runs once, on the first call.
    """
    return _first_token_timer_class()(on_first_call)


BNB_SKIP_MODULES = ["in_proj_a", "in_proj_b", "lm_head"]


def bnb_quant_config(quant_type: str) -> dict:
    """Keyword arguments for ``transformers.BitsAndBytesConfig`` (4-bit load).

    ``quant_type`` must be "nf4" or "fp4". Computation runs in bf16 (the patch
    adds the bias in the query dtype) and double quantization is on. The
    modules in ``BNB_SKIP_MODULES`` (DeltaNet ``in_proj_a`` / ``in_proj_b`` and
    ``lm_head``, matched as substrings of the module path) stay unquantized,
    as in the RedHatAI INT4 recipe.
    """
    if quant_type not in ("nf4", "fp4"):
        raise ValueError("bitsandbytes 4-bit quant type must be 'nf4' or 'fp4', got %r" % (quant_type,))
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": quant_type,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "llm_int8_skip_modules": list(BNB_SKIP_MODULES),
    }


def prefers_native_multimodal_class(architectures: list, is_quantized: bool) -> bool:
    """True if the checkpoint must be loaded with its own multimodal class.

    A pre-quantized ``*ForConditionalGeneration`` checkpoint lists the modules
    it left unquantized (``quantization_config.ignore``) by their paths in the
    full multimodal tree (``model.language_model.layers...``). The text-only
    ``AutoModelForCausalLM`` class drops the ``language_model.`` segment, so
    those rules stop matching and modules get quantized that should not be
    (on RedHatAI/Qwen3.8-27B-INT4 this left 288 weights missing). Unquantized
    checkpoints are unaffected.
    """
    return bool(is_quantized) and any(a.endswith("ForConditionalGeneration") for a in architectures)


class MassWeightedGemma:
    # Class-level defaults, so an instance created with __new__ (without
    # __init__, as verify_attention_math.py does) still has these attributes.
    _record_attention: bool = False
    recorded_attention: list
    # Prefill/decode phase. True outside generate(), so calling the patched
    # function directly treats a one-row query as a decode step. generate()
    # sets it False and the first-token timer sets it True again after the
    # last prefill forward.
    _prefill_done: bool = True
    # Off by default: during the last prefill forward, add w * mass to the
    # final query row only (the row that produces the first answer token).
    # generate() sets ``_prompt_len`` so that forward can be recognised
    # (seq_k == prompt_len); it is None outside generate().
    prefill_last_row: bool = False
    _prompt_len: int | None = None
    bias_applied_prefill_last_row_calls: int = 0

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        model_id: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        do_sample: bool | None = None,
        allow_sliding_layers: bool | None = None,
        quantization: str | None = None,
        prefill_last_row: bool = False,
    ):
        cfg = load_config(config_path) if config_path else load_config()
        server_cfg = cfg.get("server", {})

        self._model_id: str = model_id or server_cfg.get("model_id", "google/gemma-3-4b-it")
        # "none" loads in bf16 without bitsandbytes; anything else is a
        # bitsandbytes 4-bit type ("nf4" by default).
        self._quantization: str = (
            quantization if quantization is not None else server_cfg.get("quantization", "nf4")
        )
        self._device: str = server_cfg.get("device", "cuda")
        self._max_new_tokens: int = max_new_tokens if max_new_tokens is not None else server_cfg.get("max_new_tokens", 512)
        self._temperature: float = temperature if temperature is not None else server_cfg.get("temperature", 0.7)
        self._do_sample: bool = do_sample if do_sample is not None else server_cfg.get("do_sample", True)
        self._mass_weight: float = get("attention", "mass_weight", 1.0)
        # Scale of the mass bias during prefill. The default 0.0 applies the
        # bias on decode steps only: adding the full weight during prefill
        # collapsed the representations in every layer and the output
        # degenerated into repeated strings. Values like 0.1 or 0.5 are
        # experimental.
        self._prefill_mass_scale: float = get("attention", "prefill_mass_scale", 0.0)
        # Upper bound on the effective bias w * mass (None disables it).
        _cap = get("attention", "bias_cap", 3.0)
        self._bias_cap: float | None = float(_cap) if _cap is not None else None

        # Optional Q/K normalisation before attention (see patched_sdpa):
        #   "off"     - no change (default)
        #   "l2"      - full L2 normalisation rescaled by sqrt(head_dim); broke Llama outputs
        #   "l2_soft" - alpha * normalised + (1 - alpha) * original
        #   "clip"    - shrink only vectors whose norm exceeds threshold * sqrt(head_dim)
        self._qk_norm_mode: str = "off"
        self._qk_norm_alpha: float = 0.5      # for "l2_soft"
        self._qk_clip_threshold: float = 2.0  # for "clip"

        # Sliding-window layers (e.g. most Gemma-3 layers) truncate the KV
        # cache, so key index j is no longer absolute position j and the mass
        # vector would land on the wrong tokens. By default that raises; True
        # skips the bias on such layers instead.
        self._allow_sliding_layers: bool = (
            allow_sliding_layers
            if allow_sliding_layers is not None
            else bool(get("attention", "allow_sliding_layers", False))
        )

        self._model = None
        self._tokenizer = None
        self._m_matrix: torch.Tensor | None = None
        self._mass_vector: torch.Tensor | None = None  # 1D mass vector (long contexts)
        self._original_sdpa = None

        # Opt-in attention recorder, off by default. When on, every decode
        # step recomputes the softmax in float32, which is only affordable for
        # a short probe.
        self._record_attention: bool = False
        self.recorded_attention: list[torch.Tensor] = []

        # How often the mass bias was applied or skipped; see mass_injection_stats().
        self.bias_applied_calls: int = 0
        self.bias_skipped_prefill_calls: int = 0
        self.bias_skipped_sliding_calls: int = 0
        self.prefill_last_row: bool = bool(prefill_last_row)
        self._prompt_len: int | None = None
        # Sdpa calls whose final prefill row received the bias: one per
        # attention layer per generate() when prefill_last_row is on, else 0.
        self.bias_applied_prefill_last_row_calls: int = 0

        # Token count and timing of the last generate(); None until the first call.
        self.last_generated_tokens: int | None = None
        self.last_prefill_ms: float | None = None
        self.last_decode_ms: float | None = None
        # Decode forwards of the last generate() = generated tokens - 1
        # (the first token comes out of the prefill forward).
        self.last_decode_forwards: int | None = None
        self._prefill_done = True

        # Set by load(): the checkpoint's declared architectures, the class
        # actually instantiated, and the loading report.
        self.checkpoint_architectures: list[str] = []
        self.loaded_class_name: str | None = None
        self.loading_info: dict[str, int] | None = None

    # ── Mass-injection counters ────────────────────────────────────────

    def _reset_mass_injection_stats(self) -> None:
        self.bias_applied_calls = 0
        self.bias_skipped_prefill_calls = 0
        self.bias_skipped_sliding_calls = 0
        self.bias_applied_prefill_last_row_calls = 0

    def mass_injection_stats(self) -> dict[str, int]:
        """How many sdpa calls applied or skipped the mass bias since the last set/clear.

        - bias_applied_calls:         calls that added the bias
        - bias_skipped_prefill_calls: prefill calls skipped because
                                      prefill_mass_scale is 0 (expected)
        - bias_skipped_sliding_calls: calls skipped on a truncated
                                      (sliding-window) cache; only with
                                      allow_sliding_layers=True

        ``bias_applied_prefill_last_row_calls`` is an attribute, not a key here,
        because verify_attention_math.py expects exactly these three keys.
        """
        return {
            "bias_applied_calls": self.bias_applied_calls,
            "bias_skipped_prefill_calls": self.bias_skipped_prefill_calls,
            "bias_skipped_sliding_calls": self.bias_skipped_sliding_calls,
        }

    @property
    def allow_sliding_layers(self) -> bool:
        return self._allow_sliding_layers

    def load(self) -> None:
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        # The loading class follows the checkpoint's declared architecture.
        # AutoModelForCausalLM is the default; it also loads the text part of
        # vision-language checkpoints whose model_type it knows (e.g. qwen3_5,
        # with no missing weights). AutoModelForImageTextToText is used when
        # AutoModelForCausalLM rejects the model_type, or when the checkpoint
        # is pre-quantized: its quantization ignore-list names module paths in
        # the full multimodal tree, which the text-only class does not have
        # (see prefers_native_multimodal_class).
        config = AutoConfig.from_pretrained(self._model_id)
        self.checkpoint_architectures = list(getattr(config, "architectures", None) or [])
        is_quantized = getattr(config, "quantization_config", None) is not None
        prefer_native_class = prefers_native_multimodal_class(self.checkpoint_architectures, is_quantized)
        print(f"[MassWeightedGemma] checkpoint architectures: {self.checkpoint_architectures} "
              f"(model_type={getattr(config, 'model_type', None)!r}, quantized={is_quantized}) -> "
              f"{'AutoModelForImageTextToText (quantized ignore-list needs the native tree)' if prefer_native_class else 'AutoModelForCausalLM'}")
        if self._quantization == "none":
            # bf16 without bitsandbytes (a pre-quantized checkpoint stays as it
            # is), so a measured effect comes from the mass bias and not from
            # extra quantization error.
            load_kwargs = dict(
                dtype=torch.bfloat16,
                device_map="auto",
                # The patch replaces F.scaled_dot_product_attention; "eager" or
                # "flash_attention_2" never call it, so the bias would be built
                # but never applied.
                attn_implementation="sdpa",
                output_loading_info=True,
            )
            if prefer_native_class:
                from transformers import AutoModelForImageTextToText

                # kv_cache_scheme (FP8 KV cache) is a vLLM feature: transformers
                # keeps the KV cache in bf16, and compressed-tensors 0.14 fails
                # on it for a nested text_config ("Cannot determine
                # num_attention_heads"). Drop it from the config we pass on.
                qc = getattr(config, "quantization_config", None)
                if isinstance(qc, dict) and qc.get("kv_cache_scheme") is not None:
                    qc = dict(qc)
                    qc.pop("kv_cache_scheme", None)
                    config.quantization_config = qc
                    print("[MassWeightedGemma] dropped quantization_config.kv_cache_scheme (KV stays bf16 in transformers)")
                    load_kwargs["config"] = config
                self._model, info = AutoModelForImageTextToText.from_pretrained(
                    self._model_id, **load_kwargs
                )
            else:
                try:
                    self._model, info = AutoModelForCausalLM.from_pretrained(
                        self._model_id, **load_kwargs
                    )
                except ValueError as exc:
                    # model_type not registered for AutoModelForCausalLM (some
                    # vision-language checkpoints); their text path is still a
                    # causal LM whose attention goes through SDPA.
                    from transformers import AutoModelForImageTextToText

                    print(f"[MassWeightedGemma] AutoModelForCausalLM refused ({exc}); "
                          "falling back to AutoModelForImageTextToText")
                    self._model, info = AutoModelForImageTextToText.from_pretrained(
                        self._model_id, **load_kwargs
                    )
        else:
            # bitsandbytes 4-bit on an unquantized checkpoint, for GPUs too
            # small for the bf16 weights. Weights stay 4-bit in memory and are
            # dequantized per matmul; see bnb_quant_config for the skip list.
            bnb_config = BitsAndBytesConfig(**bnb_quant_config(self._quantization))
            self._model, info = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                quantization_config=bnb_config,
                device_map="auto",
                dtype=torch.bfloat16,
                attn_implementation="sdpa",  # see the bf16 branch above
                output_loading_info=True,
            )
        self._record_loading_info(info)
        self._model.eval()
        self._patch_sdpa()
        print(f"[MassWeightedGemma] loaded: {self._model_id} as {self.loaded_class_name} "
              f"loading_info={self.loading_info}")

    def _record_loading_info(self, info: dict) -> None:
        """Store the loading report; raise on missing or mismatched weights.

        A model with weights left at their init values would run as a random
        model, so that is a hard error. Unexpected keys (e.g. a vision tower
        that the text-only class drops) are allowed and only counted.
        """
        missing = list(info.get("missing_keys", []) or [])
        unexpected = list(info.get("unexpected_keys", []) or [])
        mismatched = list(info.get("mismatched_keys", []) or [])
        self.loaded_class_name = type(self._model).__name__
        self.loading_info = {
            "missing": len(missing),
            "unexpected": len(unexpected),
            "mismatched": len(mismatched),
        }
        if missing or mismatched:
            raise RuntimeError(
                "%s loaded %s with %d missing and %d mismatched weight(s); "
                "refusing to run on partially initialised weights. missing[:10]=%r "
                "mismatched[:10]=%r"
                % (self.loaded_class_name, self._model_id, len(missing), len(mismatched),
                   missing[:10], mismatched[:10])
            )

    def set_m_matrix(self, M: torch.Tensor) -> None:
        self._m_matrix = M

    def clear_m_matrix(self) -> None:
        self._m_matrix = None

    def set_mass_vector(self, v: torch.Tensor) -> None:
        """Set the 1D mass vector, shape (seq_len,), and reset the counters."""
        self._mass_vector = v
        self._reset_mass_injection_stats()

    def clear_mass_vector(self) -> None:
        self._mass_vector = None
        self._reset_mass_injection_stats()

    @property
    def tokenizer(self):
        return self._tokenizer

    # ── Generation ─────────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        # With device_map="auto" the model may be split across devices; put
        # the inputs on the device of the first parameter (the embeddings).
        target_device = self._device
        try:
            target_device = next(self._model.parameters()).device
        except Exception:
            pass

        inputs = self._tokenizer(prompt, return_tensors="pt").to(target_device)
        input_ids = inputs["input_ids"]
        # The last prefill forward is the one whose seq_k equals the prompt
        # length (the only prefill forward when prefill is not chunked).
        self._prompt_len = int(input_ids.shape[1])
        if self.prefill_last_row and self._prefill_mass_scale > 0.0:
            raise ValueError(
                "prefill_last_row and prefill_mass_scale=%g are mutually exclusive: "
                "the first would add the full bias to the final prefill row on top of "
                "the scaled bias the second adds to every row" % self._prefill_mass_scale
            )

        from transformers import LogitsProcessorList

        # The timer's first call marks the end of prefill (after the last
        # prefill chunk), so patched_sdpa can tell a 1-token final prefill
        # chunk from a decode step.
        timer = make_first_token_timer(on_first_call=self._mark_prefill_done)
        gen_kwargs: dict = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": self._do_sample,
            "logits_processor": LogitsProcessorList([timer]),
        }
        if self._do_sample:
            gen_kwargs["temperature"] = self._temperature

        # Wall-clock split: prefill = start -> first logits call, decode = the rest.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        self._prefill_done = False
        try:
            with torch.no_grad():
                output_ids = self._model.generate(**inputs, **gen_kwargs)
        finally:
            self._prefill_done = True
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000.0

        new_ids = output_ids[0, input_ids.shape[1]:]
        self.last_generated_tokens = int(new_ids.shape[0])
        # The prefill forward(s) yield the first token; each further token is one decode forward.
        self.last_decode_forwards = max(self.last_generated_tokens - 1, 0)
        if timer.first_call_t is not None:
            self.last_prefill_ms = (timer.first_call_t - t0) * 1000.0
            self.last_decode_ms = total_ms - self.last_prefill_ms
        else:  # no decode step ran (max_new_tokens == 0): no split to report
            self.last_prefill_ms = None
            self.last_decode_ms = None
        return self._tokenizer.decode(new_ids, skip_special_tokens=True)

    def _mark_prefill_done(self) -> None:
        self._prefill_done = True

    # ── Core: the scaled_dot_product_attention patch ───────────────────

    def _patch_sdpa(self) -> None:
        """Replace torch's scaled_dot_product_attention with the mass-bias wrapper.

        SDPA adds ``attn_mask`` to the logits before the softmax, so merging
        ``w * M`` into the mask implements ``attn_scores += w * M``.

        The wrapper always calls ``_ORIGINAL_SDPA`` (captured at import), so it
        can never wrap itself. Patching again from the owning instance is a
        no-op; patching while another instance owns the patch raises.
        """
        global _PATCH_OWNER
        if _PATCH_OWNER is self:
            return
        if _PATCH_OWNER is not None:
            raise RuntimeError(
                "scaled_dot_product_attention is already patched by another %s "
                "instance; call restore_sdpa() on it before patching again"
                % type(_PATCH_OWNER).__name__
            )
        outer = self
        self._original_sdpa = _ORIGINAL_SDPA

        def patched_sdpa(
            query, key, value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            **kwargs,
        ):
            # Optional Q/K normalisation (_qk_norm_mode). Flattening outlier
            # ("massive") Q/K values (Sun et al. 2025, arXiv:2502.01563) makes
            # the pre-softmax logits more compact, so the additive bias has
            # more leverage.
            qk_mode = outer._qk_norm_mode
            if qk_mode != "off":
                d_head = query.shape[-1]
                scale_factor = d_head ** 0.5

                if qk_mode == "l2":
                    # Full L2 normalisation, rescaled by sqrt(d).
                    query = F.normalize(query.float(), p=2, dim=-1).to(query.dtype) * scale_factor
                    key = F.normalize(key.float(), p=2, dim=-1).to(key.dtype) * scale_factor

                elif qk_mode == "l2_soft":
                    # Partial normalisation: damps outliers but keeps most of the original vector.
                    alpha = outer._qk_norm_alpha
                    q_norm = F.normalize(query.float(), p=2, dim=-1).to(query.dtype) * scale_factor
                    k_norm = F.normalize(key.float(), p=2, dim=-1).to(key.dtype) * scale_factor
                    query = alpha * q_norm + (1.0 - alpha) * query
                    key = alpha * k_norm + (1.0 - alpha) * key

                elif qk_mode == "clip":
                    # Shrink only vectors whose norm exceeds threshold * sqrt(d);
                    # all others are left untouched.
                    max_norm = outer._qk_clip_threshold * scale_factor
                    q_norms = query.float().norm(dim=-1, keepdim=True).clamp(min=1e-9)
                    k_norms = key.float().norm(dim=-1, keepdim=True).clamp(min=1e-9)
                    q_scale = (max_norm / q_norms).clamp(max=1.0).to(query.dtype)
                    k_scale = (max_norm / k_norms).clamp(max=1.0).to(key.dtype)
                    query = query * q_scale
                    key = key * k_scale

            seq_q = query.shape[-2]
            seq_k = key.shape[-2]
            # Decode = the prefill forwards are over AND this is a one-row
            # query. Outside generate() ``_prefill_done`` is True (class
            # default), so this reduces to ``seq_q == 1``.
            is_decode = outer._prefill_done and seq_q == 1

            m_bias = build_mass_bias(
                seq_q,
                seq_k,
                m_matrix=outer._m_matrix,
                mass_vector=outer._mass_vector,
                mass_weight=outer._mass_weight,
                prefill_mass_scale=outer._prefill_mass_scale,
                dtype=query.dtype,
                device=query.device,
                strict_alignment=not outer._allow_sliding_layers,
                # getattr: verify_attention_math builds the instance via __new__ (no __init__)
                bias_cap=getattr(outer, "_bias_cap", None),
                phase_is_decode=is_decode,
            )

            if m_bias is not None:
                attn_mask, is_causal = combine_attn_mask(
                    attn_mask,
                    m_bias,
                    is_causal=is_causal,
                    seq_q=seq_q,
                    seq_k=seq_k,
                    dtype=query.dtype,
                    device=query.device,
                )
                outer.bias_applied_calls += 1
                # With a float mask AND enable_gqa=True torch may fall back to
                # the math kernel (float32 at long contexts); expand K/V here
                # and pass enable_gqa=False instead.
                key, value, kwargs = _expand_gqa_heads(query, key, value, kwargs)
            elif outer._m_matrix is None and outer._mass_vector is not None:
                # Count why the bias was skipped (same order of checks as in
                # build_mass_bias).
                if not is_decode and outer._prefill_mass_scale <= 0.0:
                    outer.bias_skipped_prefill_calls += 1
                elif seq_k < outer._mass_vector.shape[0]:
                    outer.bias_skipped_sliding_calls += 1

                # prefill_last_row: the last prefill forward (seq_k ==
                # prompt_len) gets the bias on its final query row only. The
                # other rows keep the plain prefill output (and the skip above
                # stays counted).
                if (
                    outer.prefill_last_row
                    and not outer._prefill_done
                    and seq_q > 1
                    and outer._prompt_len is not None
                    and seq_k == outer._prompt_len
                ):
                    return outer._prefill_last_row_sdpa(
                        query, key, value,
                        attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                        scale=scale, kwargs=kwargs,
                    )

            # Optional probability recorder. Runs on decode calls only (a
            # 1-token final prefill chunk is not decode), after the mass bias
            # has been merged into attn_mask, so it records exactly the
            # distribution the model uses. getattr: an instance built via
            # __new__ (no __init__) may lack the attribute.
            if getattr(outer, "_record_attention", False) and is_decode:
                outer._record_attention_row(
                    query, key, attn_mask, scale, bool(kwargs.get("enable_gqa", False))
                )

            return outer._original_sdpa(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                **kwargs,
            )

        # Install the wrapper on torch.nn.functional, where transformers looks
        # it up at call time.
        import torch.nn.functional as _F
        _F.scaled_dot_product_attention = patched_sdpa
        torch.nn.functional.scaled_dot_product_attention = patched_sdpa
        _PATCH_OWNER = self
        print("[MassWeightedGemma] patched scaled_dot_product_attention")

    def _prefill_last_row_sdpa(
        self, query, key, value, *, attn_mask, dropout_p, is_causal, scale, kwargs
    ):
        """Plain prefill output, with the final query row recomputed with ``w * mass`` added.

        The last row is treated as a decode row over the full prompt: full
        weight, bias cap applied, and ``is_causal=False`` because the last row
        may attend to every key; an existing mask contributes its last row.
        """
        seq_k = key.shape[-2]
        out = self._original_sdpa(
            query, key, value,
            attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
            **kwargs,
        )
        bias = build_mass_bias(
            1, seq_k,
            m_matrix=None,
            mass_vector=self._mass_vector,
            mass_weight=self._mass_weight,
            prefill_mass_scale=self._prefill_mass_scale,
            dtype=query.dtype,
            device=query.device,
            strict_alignment=not self._allow_sliding_layers,
            bias_cap=getattr(self, "_bias_cap", None),
            phase_is_decode=True,
        )
        if bias is None:  # sliding skip (allow_sliding_layers): row stays plain
            return out
        last_mask = None if attn_mask is None else attn_mask[..., -1:, :]
        row_mask, _ = combine_attn_mask(
            last_mask, bias, is_causal=False, seq_q=1, seq_k=seq_k,
            dtype=query.dtype, device=query.device,
        )
        key, value, kwargs = _expand_gqa_heads(query, key, value, kwargs)
        out[:, :, -1:, :] = self._original_sdpa(
            query[:, :, -1:, :], key, value,
            attn_mask=row_mask, dropout_p=dropout_p, is_causal=False, scale=scale,
            **kwargs,
        )
        self.bias_applied_prefill_last_row_calls += 1
        return out

    # ── Attention recorder (diagnostic probes only) ────────────────────

    def start_attention_recording(self) -> None:
        """Record the softmax row of every decode step from now on."""
        self.recorded_attention = []
        self._record_attention = True

    def stop_attention_recording(self) -> list[torch.Tensor]:
        self._record_attention = False
        return self.recorded_attention

    def _record_attention_row(
        self, query, key, attn_mask, scale, enable_gqa: bool
    ) -> None:
        """Append the attention probabilities of the last query row.

        Recomputes ``softmax(q @ k^T * scale + attn_mask)`` in float32 for the
        last query position and averages over batch and heads: one ``(seq_k,)``
        vector per sdpa call (per layer per decode step). HF's
        ``output_attentions=True`` cannot be used instead: it switches the
        model to eager attention, which bypasses the patch and so measures the
        unpatched model. Bool masks are converted as in ``combine_attn_mask``
        so they do not become 1.0/0.0.
        """
        seq_k = int(key.shape[-2])
        if seq_k > RECORD_MAX_KEYS:
            raise RuntimeError(
                "attention recording refused: seq_k=%d > RECORD_MAX_KEYS=%d (the F4 "
                "probe is <= 300 tokens; the recorder recomputes a float32 softmax row "
                "per layer and is not meant for the full context)" % (seq_k, RECORD_MAX_KEYS)
            )
        q = query[..., -1:, :].to(torch.float32)          # (B, Hq, 1, D)
        k = key.to(torch.float32)                          # (B, Hk, Sk, D)
        if k.shape[1] != q.shape[1] and q.shape[1] % k.shape[1] == 0:
            # grouped-query attention: each kv head serves several query heads
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        factor = (query.shape[-1] ** -0.5) if scale is None else float(scale)
        logits = torch.matmul(q, k.transpose(-2, -1)) * factor   # (B, Hq, 1, Sk)
        if attn_mask is not None:
            mask = attn_mask
            if mask.dtype == torch.bool:
                mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(
                    ~mask, torch.finfo(torch.float32).min
                )
            else:
                mask = mask.to(torch.float32)
            logits = logits + mask[..., -1:, :]
        probs = torch.softmax(logits, dim=-1)[..., -1, :]        # (B, Hq, Sk)
        self.recorded_attention.append(
            probs.mean(dim=(0, 1)).detach().to(torch.float32).cpu()
        )

    def restore_sdpa(self) -> None:
        """Put the original SDPA back (e.g. in tests).

        Only the owning instance restores; a call from any other instance is a
        no-op, so it cannot undo another instance's patch.
        """
        global _PATCH_OWNER
        if _PATCH_OWNER is not self:
            return
        import torch.nn.functional as _F
        _F.scaled_dot_product_attention = _ORIGINAL_SDPA
        torch.nn.functional.scaled_dot_product_attention = _ORIGINAL_SDPA
        _PATCH_OWNER = None

    # ── Compatibility check ────────────────────────────────────────────

    @property
    def attn_implementation(self) -> str | None:
        """The model's attention implementation ("sdpa", "eager", ...); the patch only works with "sdpa"."""
        if self._model is None:
            return None
        cfg = getattr(self._model, "config", None)
        if cfg is None:
            return None
        return getattr(cfg, "_attn_implementation", None)


# Model-agnostic alias; newer code imports MassWeightedLLM.
MassWeightedLLM = MassWeightedGemma


# ── Helpers ───────────────────────────────────────────────────────────

def _expand_gqa_heads(query, key, value, kwargs: dict) -> tuple:
    """If ``enable_gqa=True`` was requested, repeat the K/V heads up to the query head count.

    The heads are repeated in their own dtype and kwargs come back with
    ``enable_gqa=False``, so SDPA can use a fused kernel instead of the math
    fallback that promotes to float32. Without ``enable_gqa`` everything is
    returned unchanged.
    """
    if not kwargs.get("enable_gqa"):
        return key, value, kwargs
    groups = query.shape[1] // key.shape[1]
    if groups > 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    return key, value, {**kwargs, "enable_gqa": False}


def _make_causal_mask(
    seq_q: int, seq_k: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Float causal mask (-inf on future keys), used instead of is_causal=True.

    Bottom-right aligned: the last query row sees every key.
    """
    mask = torch.full((seq_q, seq_k), float("-inf"), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=seq_k - seq_q + 1)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_q, seq_k)


def build_mass_bias(
    seq_q: int,
    seq_k: int,
    *,
    m_matrix: torch.Tensor | None,
    mass_vector: torch.Tensor | None,
    mass_weight: float,
    prefill_mass_scale: float,
    dtype: torch.dtype,
    device: torch.device,
    strict_alignment: bool = True,
    bias_cap: float | None = None,
    phase_is_decode: bool | None = None,
) -> torch.Tensor | None:
    """Return the additive pre-softmax bias, or None when no bias applies.

    2D mode (``m_matrix`` set; wins if both are set): shape (1, 1, seq_q, seq_k),
        ``mass_weight * M[:seq_q, :seq_k]``, zero-padded where M is smaller.
    1D mode (``mass_vector``): shape (1, 1, 1, seq_k), broadcast over query rows,
        ``effective_w * mass_vector[:seq_k]``, zero-padded, where effective_w is
          - ``mass_weight`` on a decode step,
          - ``mass_weight * prefill_mass_scale`` on prefill when that scale > 0,
          - otherwise there is no bias (None).

    The bias is added to logits that are already scaled by 1/sqrt(d), so it
    must not be scaled by 1/sqrt(d) here.

    phase_is_decode: None -> decode iff ``seq_q == 1``; True -> decode; False
        -> prefill, even for a one-row final chunk of a chunked prefill.

    bias_cap: clamps the final ``w * mass`` (not just the mass) to at most this
        value; capping only the mass let w=3 turn a 3.0 cap into 9.0. None
        means no cap.

    Truncated cache (1D mode): the vector is indexed by absolute token
    position. A sliding-window layer keeps only the most recent keys
    (seq_k < len(mass_vector)) and SDPA is not told the offset, so the bias
    would land on the wrong tokens. Unless the call is an explicit prefill,
    this raises RuntimeError (``strict_alignment=True``, default) or returns
    None for that layer (``strict_alignment=False``). Mass injection therefore
    targets models whose layers all use full attention. In an explicit prefill
    (``phase_is_decode=False``) a shorter seq_k is a chunked prefill whose keys
    are positions 0..seq_k-1, so the vector is simply sliced.
    """
    if m_matrix is not None:
        # 2D matrix mode (short contexts)
        m_q = min(seq_q, m_matrix.shape[0])
        m_k = min(seq_k, m_matrix.shape[1])
        m_slice = mass_weight * m_matrix[:m_q, :m_k]

        # Zero-pad once the KV cache has grown past the size of M.
        if m_q < seq_q or m_k < seq_k:
            full = torch.zeros(seq_q, seq_k, dtype=torch.float32, device=m_matrix.device)
            full[:m_q, :m_k] = m_slice
            m_slice = full

        # (seq_q, seq_k) -> (1, 1, seq_q, seq_k)
        if bias_cap is not None:
            m_slice = m_slice.clamp(max=bias_cap)
        return m_slice.to(dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)

    if mass_vector is not None:
        # 1D mass-vector mode (long contexts, memory-light). Applied on decode
        # steps; on prefill only when prefill_mass_scale > 0, because the full
        # weight during prefill collapsed the model's representations.
        effective_w: float | None = None
        is_decode = (seq_q == 1) if phase_is_decode is None else bool(phase_is_decode)
        if is_decode:
            effective_w = mass_weight
        elif prefill_mass_scale > 0.0:
            effective_w = mass_weight * prefill_mass_scale

        if effective_w is None:
            return None

        # Truncated (sliding-window) cache check. An explicit prefill with a
        # shorter cache is a chunked prefill (keys are positions 0..seq_k-1),
        # so the vector is sliced; decode or unknown phase raises or skips.
        mass_len = mass_vector.shape[0]
        if seq_k < mass_len and phase_is_decode is not False:
            if strict_alignment:
                raise RuntimeError(
                    "mass injection requires full-attention layers: "
                    "seq_k=%d < mass_len=%d (sliding-window cache detected)"
                    % (seq_k, mass_len)
                )
            return None

        m_k = min(seq_k, mass_vector.shape[0])
        m_vec = effective_w * mass_vector[:m_k]

        if m_k < seq_k:
            full_vec = torch.zeros(seq_k, dtype=torch.float32, device=mass_vector.device)
            full_vec[:m_k] = m_vec
            m_vec = full_vec

        # (seq_k,) -> (1, 1, 1, seq_k), broadcast over heads and query rows
        if bias_cap is not None:
            m_vec = m_vec.clamp(max=bias_cap)
        return (
            m_vec.to(dtype=dtype, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )

    return None


def combine_attn_mask(
    attn_mask: torch.Tensor | None,
    m_bias: torch.Tensor,
    *,
    is_causal: bool,
    seq_q: int,
    seq_k: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, bool]:
    """Merge ``m_bias`` into ``attn_mask``; return ``(attn_mask, is_causal)``.

    - No mask: returns ``m_bias``. If ``is_causal=True`` the causal mask is
      added to it and ``is_causal`` comes back False, because SDPA does not
      accept a float mask together with ``is_causal=True``.
    - Bool mask (True = may attend; transformers' sdpa path passes these):
      converted like transformers' eager path,
      ``zeros.masked_fill(~mask, finfo(dtype).min)``, then added. A plain
      ``.to(dtype)`` would turn it into 1.0/0.0 and erase the causal/padding/
      sliding masking. finfo.min rather than -inf keeps fully masked rows from
      turning into NaN.
    - Float mask: ``attn_mask.to(dtype) + m_bias``.

    A shape mismatch raises RuntimeError with both shapes instead of silently
    running without the bias.
    """
    if attn_mask is None:
        # SDPA rejects a float mask together with is_causal=True, so fold the
        # causal mask into the bias.
        if is_causal:
            causal_mask = _make_causal_mask(seq_q, seq_k, dtype, device)
            m_bias = m_bias + causal_mask
            is_causal = False
        return m_bias, is_causal

    if attn_mask.dtype == torch.bool:
        # True = may attend, False = masked; same conversion as transformers' eager mask.
        float_mask = torch.zeros_like(attn_mask, dtype=dtype).masked_fill(
            ~attn_mask, torch.finfo(dtype).min
        )
    else:
        float_mask = attn_mask.to(dtype=dtype)

    try:
        return float_mask + m_bias, is_causal
    except RuntimeError as exc:
        raise RuntimeError(
            "mass bias could not be combined with attn_mask: "
            f"attn_mask.shape={tuple(attn_mask.shape)} dtype={attn_mask.dtype}, "
            f"m_bias.shape={tuple(m_bias.shape)}, seq_q={seq_q}, seq_k={seq_k}, "
            f"is_causal={is_causal}"
        ) from exc
