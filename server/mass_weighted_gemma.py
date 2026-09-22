"""
MassWeightedGemma (= MassWeightedLLM): 特許クレームの "attn_scores += w * M" を実装する。

【モデル非依存】
  AutoModelForCausalLM + AutoTokenizer を使用しているため、HuggingFace 上の
  任意の Causal LM に対応可能。検証済み:
    - google/gemma-3-4b-it (主実験)
    - THUDM/glm-4-9b-chat
    - Qwen/Qwen2.5-7B-Instruct
    - meta-llama/Llama-3.1-8B-Instruct

介入方法:
  torch.nn.functional.scaled_dot_product_attention を monkey-patch する。
  この関数は attn_mask（float tensor）を logits に加算してから softmax に渡す。
  つまり M を attn_mask として渡すことは "scores += w * M" と等価。

  transformers ライブラリは多くのアーキテクチャで sdpa バックエンドを使うため、
  モデル固有コードを一切変更せずに介入できる。
  `model.config._attn_implementation` が "sdpa" であれば動作保証。

使い方:
  model = MassWeightedGemma(model_id="THUDM/glm-4-9b-chat")
  model.load()
  M = m_matrix_builder.build(...)
  model.set_m_matrix(M)
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

# Astra round 2 (2026-09-18), item 2: the pristine torch kernel, captured ONCE
# at import time (before any patch), so a wrapper can never capture another
# wrapper and call itself.  ``_PATCH_OWNER`` is the instance whose closure is
# currently installed (None = unpatched); only the owner may restore.
_ORIGINAL_SDPA = F.scaled_dot_product_attention
_PATCH_OWNER: "MassWeightedGemma | None" = None

# Item 3: the recorder recomputes a float32 softmax row per layer.  It exists
# for the F4 share probe (<= 300 tokens); longer rows are refused, never
# silently recorded at 190k keys.
RECORD_MAX_KEYS = 4096


@lru_cache(maxsize=1)
def _first_token_timer_class():
    """``transformers`` is imported lazily (load() does the same) so importing
    this module stays torch-only for verify_attention_math.py."""
    from transformers import LogitsProcessor

    class FirstTokenTimer(LogitsProcessor):
        """Stores ``time.perf_counter()`` on its FIRST ``__call__``.

        ``generate()`` invokes the logits processors after the prefill forward
        and before the first token is sampled, so the first call marks the
        prefill/decode boundary (E1 wall-ms split).  Scores are returned
        untouched.
        """

        def __init__(self, on_first_call=None) -> None:
            self.first_call_t: float | None = None
            # A2: the owner's phase flag is flipped here, on the very first
            # logits call, i.e. after the LAST prefill forward (chunked or not).
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
    """A fresh ``FirstTokenTimer`` (a ``transformers.LogitsProcessor``).

    ``on_first_call`` (optional, no arguments) runs once, on the first call.
    """
    return _first_token_timer_class()(on_first_call)


BNB_SKIP_MODULES = ["in_proj_a", "in_proj_b", "lm_head"]


def bnb_quant_config(quant_type: str) -> dict:
    """Keyword arguments for ``transformers.BitsAndBytesConfig`` (4-bit reader).

    ``quant_type`` is a bitsandbytes 4-bit type ("nf4" or "fp4"). Compute in
    bf16 (the SDPA patch adds the bias in the query dtype); double
    quantization on; DeltaNet ``in_proj_a`` / ``in_proj_b`` and ``lm_head``
    are left unquantized (substring match on the module path, as
    transformers' ``replace_with_bnb_linear`` does), the same modules the
    RedHatAI INT4 recipe keeps in bf16.
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
    """A1 loading-class decision (see the comment in ``load()``).

    A pre-quantized ``*ForConditionalGeneration`` checkpoint's
    ``quantization_config.ignore`` list is written against the FULL
    multimodal module tree (``model.language_model.layers...``); the
    text-only ``AutoModelForCausalLM`` class drops the ``language_model.``
    segment (``model.layers...``), so the ignore rules silently stop
    matching and compressed-tensors quantizes modules the checkpoint left
    unquantized -- reproduced 2026-09-20 on RedHatAI/Qwen3.8-27B-INT4 (288
    missing weights). Prefer the checkpoint's own (native) class whenever
    both conditions hold; an unquantized checkpoint is unaffected.
    """
    return bool(is_quantized) and any(a.endswith("ForConditionalGeneration") for a in architectures)


class MassWeightedGemma:
    # B2: class-level defaults so an instance built without __init__
    # (verify_attention_math.py uses __new__) still answers the recorder probe.
    _record_attention: bool = False
    recorded_attention: list
    # A2: prefill/decode phase.  True outside generate(), so the direct closure
    # calls of verify_attention_math keep the ``seq_q == 1`` decode semantics;
    # generate() sets it False before model.generate and the FirstTokenTimer
    # sets it True after the last prefill forward.
    _prefill_done: bool = True
    # H15 option (b), OFF by default: during the LAST prefill forward add
    # w * mass to the final query row only (the row that yields the first
    # answer token).  ``_prompt_len`` is set by generate() and identifies that
    # forward (seq_k == prompt_len); None outside generate().
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
        # "none" = bf16, no BitsAndBytes (S2.2 reader path). Anything else is a
        # bnb_4bit_quant_type ("nf4" default) and keeps the historical 4-bit load.
        self._quantization: str = (
            quantization if quantization is not None else server_cfg.get("quantization", "nf4")
        )
        self._device: str = server_cfg.get("device", "cuda")
        self._max_new_tokens: int = max_new_tokens if max_new_tokens is not None else server_cfg.get("max_new_tokens", 512)
        self._temperature: float = temperature if temperature is not None else server_cfg.get("temperature", 0.7)
        self._do_sample: bool = do_sample if do_sample is not None else server_cfg.get("do_sample", True)
        self._mass_weight: float = get("attention", "mass_weight", 1.0)
        # §49.2 乖離③ への対応: prefill 時のマス加算スケール (デフォルト 0.0 = 現行動作維持)
        # 0.1 や 0.5 に上げると prefill 中も mass_vec を加算する (実験用)
        # 1.0 = 特許 §0082 完全準拠だが、過去に「全18層で表現崩壊」した経緯あり
        self._prefill_mass_scale: float = get("attention", "prefill_mass_scale", 0.0)
        # D-1: 実効バイアス (w × mass) の上限。None で無効。既定 3.0。
        _cap = get("attention", "bias_cap", 3.0)
        self._bias_cap: float | None = float(_cap) if _cap is not None else None

        # §53 Exp Q: QK-norm retrofit (Llama/Qwen 救済戦略)
        # "off"     = 現行動作 (Q/K 介入なし)
        # "l2"      = full L2 normalize + sqrt(head_dim) re-scale (Pattern E: Llama 破壊)
        # "l2_soft" = α-mix: query = α × L2-normalized + (1-α) × original
        # "clip"    = outlier のみ shrink。max_norm = threshold × sqrt(head_dim) で cap
        self._qk_norm_mode: str = "off"
        self._qk_norm_alpha: float = 0.5      # for "l2_soft"
        self._qk_clip_threshold: float = 2.0  # for "clip"

        # F2: sliding-window レイヤ (Gemma-3 等) では KV キャッシュが切り詰められ、
        # 1D マスベクトルの添字 (= 絶対位置) と key の添字がずれる。 既定では
        # 例外を投げて止める。 True にすると そのレイヤだけバイアスを飛ばす。
        self._allow_sliding_layers: bool = (
            allow_sliding_layers
            if allow_sliding_layers is not None
            else bool(get("attention", "allow_sliding_layers", False))
        )

        self._model = None
        self._tokenizer = None
        self._m_matrix: torch.Tensor | None = None
        self._mass_vector: torch.Tensor | None = None  # 1D mass vector（長コンテキスト用）
        self._original_sdpa = None

        # B2 (2026-09-07): opt-in attention recorder for phase_instrument.
        # OFF by default and never touched by the reader path: enabling it makes
        # patched_sdpa recompute the softmax in float32 on every DECODE step,
        # which is only affordable for a single 300-token probe.
        self._record_attention: bool = False
        self.recorded_attention: list[torch.Tensor] = []

        # F6: 無音スキップをなくすための計数器。 mass_injection_stats() で読める。
        self.bias_applied_calls: int = 0
        self.bias_skipped_prefill_calls: int = 0
        self.bias_skipped_sliding_calls: int = 0
        # H15 (b): sdpa calls whose FINAL prefill row received the bias.
        # Expected n_sdpa_layers per generate() when the switch is on, else 0.
        self.prefill_last_row: bool = bool(prefill_last_row)
        self._prompt_len: int | None = None
        self.bias_applied_prefill_last_row_calls: int = 0

        # E1: per-generate accounting (item 6). None until the first generate().
        self.last_generated_tokens: int | None = None
        self.last_prefill_ms: float | None = None
        self.last_decode_ms: float | None = None
        # A3: decode forwards of the last generate() = generated tokens - 1
        # (the first token comes out of the prefill forward).
        self.last_decode_forwards: int | None = None
        self._prefill_done = True

        # A1: what load() actually instantiated, and the loading report.
        self.checkpoint_architectures: list[str] = []
        self.loaded_class_name: str | None = None
        self.loading_info: dict[str, int] | None = None

    # ── マス注入の計数 ────────────────────────────────────────────────

    def _reset_mass_injection_stats(self) -> None:
        self.bias_applied_calls = 0
        self.bias_skipped_prefill_calls = 0
        self.bias_skipped_sliding_calls = 0
        self.bias_applied_prefill_last_row_calls = 0

    def mass_injection_stats(self) -> dict[str, int]:
        """マス注入が実際に何回効いたかを返す (F6: 無音スキップの可視化)。

        - bias_applied_calls:         バイアスを実際に加算した sdpa 呼び出し数
        - bias_skipped_prefill_calls: seq_q>1 かつ prefill_mass_scale=0 で
                                      飛ばした回数 (D-2 の想定動作。 異常ではない)
        - bias_skipped_sliding_calls: sliding-window キャッシュ検出により
                                      飛ばした回数 (allow_sliding_layers=True のときのみ)

        The H15 counter ``bias_applied_prefill_last_row_calls`` is an
        attribute, not a key here: verify_attention_math.py compares this dict
        against exactly these three keys.
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
        # A1: the class is chosen from the checkpoint's declared architecture.
        # A ``*ForConditionalGeneration`` vision-language checkpoint whose
        # model_type is registered for AutoModelForCausalLM (transformers 5.8.0
        # maps qwen3_5 -> Qwen3_5ForCausalLM) loads through the TEXT-ONLY class:
        # verified locally with 0 missing keys and identical weights -- for an
        # UNQUANTIZED checkpoint.  The ImageTextToText fallback is taken when
        # AutoModelForCausalLM raises ValueError (model_type not registered
        # for it), OR (2026-09-20, reproduced on RedHatAI/Qwen3.8-27B-INT4)
        # when the checkpoint is compressed-tensors quantized: its
        # ``quantization_config.ignore`` list names modules under the FULL
        # multimodal tree the checkpoint was calibrated against
        # (``model.language_model.layers.N.linear_attn.in_proj_a`` etc.).
        # Qwen3_5ForCausalLM's module tree drops the ``language_model.``
        # segment (``model.layers.N...``), so those ignore rules never match:
        # compressed-tensors quantizes modules the checkpoint left in bf16,
        # loading finds a "weight" key where it expects packed 4-bit
        # sub-tensors, and 288 weights come up missing.  Loading through the
        # checkpoint's own class keeps the module tree the ignore list was
        # written for.
        config = AutoConfig.from_pretrained(self._model_id)
        self.checkpoint_architectures = list(getattr(config, "architectures", None) or [])
        is_quantized = getattr(config, "quantization_config", None) is not None
        prefer_native_class = prefers_native_multimodal_class(self.checkpoint_architectures, is_quantized)
        print(f"[MassWeightedGemma] checkpoint architectures: {self.checkpoint_architectures} "
              f"(model_type={getattr(config, 'model_type', None)!r}, quantized={is_quantized}) -> "
              f"{'AutoModelForImageTextToText (quantized ignore-list needs the native tree)' if prefer_native_class else 'AutoModelForCausalLM'}")
        if self._quantization == "none":
            # bf16, no BitsAndBytes (S2.2: the reader must not be 4-bit unless
            # the checkpoint itself is pre-quantized, so the measured effect is
            # the mass term and not an extra quantization artefact).
            load_kwargs = dict(
                dtype=torch.bfloat16,
                device_map="auto",
                # M1: the patch replaces F.scaled_dot_product_attention. Any
                # other implementation ("eager", "flash_attention_2") never
                # calls it, so the mass bias would be built and never applied.
                attn_implementation="sdpa",
                output_loading_info=True,
            )
            if prefer_native_class:
                from transformers import AutoModelForImageTextToText

                # The checkpoint's kv_cache_scheme (FP8 KV) is a vLLM feature; the
                # transformers path keeps the KV cache in bf16 and compressed-tensors
                # 0.14 cannot even resolve num_attention_heads on the nested
                # text_config for it ("Cannot determine num_attention_heads").
                # Drop it from the config we hand to from_pretrained.
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
                    # Vision-language checkpoints whose model_type is NOT
                    # registered for AutoModelForCausalLM; their text-only
                    # path is still a causal LM whose attention goes through
                    # SDPA.
                    from transformers import AutoModelForImageTextToText

                    print(f"[MassWeightedGemma] AutoModelForCausalLM refused ({exc}); "
                          "falling back to AutoModelForImageTextToText")
                    self._model, info = AutoModelForImageTextToText.from_pretrained(
                        self._model_id, **load_kwargs
                    )
        else:
            # bitsandbytes 4-bit on an UNQUANTIZED checkpoint: weights stay 4-bit
            # in memory and are dequantized per matmul by the bnb kernels
            # (the 2026-09-20 fallback for a 46 GB card, DECISIONS H24). The
            # skip list mirrors the RedHat INT4 recipe: DeltaNet in_proj_a/b
            # and lm_head stay in bf16.
            bnb_config = BitsAndBytesConfig(**bnb_quant_config(self._quantization))
            self._model, info = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                quantization_config=bnb_config,
                device_map="auto",
                dtype=torch.bfloat16,
                attn_implementation="sdpa",  # M1, see above
                output_loading_info=True,
            )
        self._record_loading_info(info)
        self._model.eval()
        self._patch_sdpa()
        print(f"[MassWeightedGemma] loaded: {self._model_id} as {self.loaded_class_name} "
              f"loading_info={self.loading_info}")

    def _record_loading_info(self, info: dict) -> None:
        """A1: missing or mismatched weights are a hard error (a text-only class
        that silently left language-model weights at init values would run as
        a random reader).  Unexpected keys (the vision tower, dropped by the
        text-only class) are allowed and counted."""
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
        """1D mass vector をセットする（長コンテキスト用、shape: (seq_len,)）。"""
        self._mass_vector = v
        self._reset_mass_injection_stats()

    def clear_mass_vector(self) -> None:
        self._mass_vector = None
        self._reset_mass_injection_stats()

    @property
    def tokenizer(self):
        return self._tokenizer

    # ── 推論 ──────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        # device_map="auto" でモデルが split された場合に備え、
        # input は最初の埋め込み層の device に合わせる
        target_device = self._device
        try:
            target_device = next(self._model.parameters()).device
        except Exception:
            pass

        inputs = self._tokenizer(prompt, return_tensors="pt").to(target_device)
        input_ids = inputs["input_ids"]
        # H15 (b): the last prefill forward is the one whose seq_k equals the
        # prompt length (the only forward for an unchunked prefill).
        self._prompt_len = int(input_ids.shape[1])
        if self.prefill_last_row and self._prefill_mass_scale > 0.0:
            raise ValueError(
                "prefill_last_row and prefill_mass_scale=%g are mutually exclusive: "
                "the first would add the full bias to the final prefill row on top of "
                "the scaled bias the second adds to every row" % self._prefill_mass_scale
            )

        from transformers import LogitsProcessorList

        # A2: the timer's first call marks the end of prefill (after the LAST
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

        # E1 wall-ms split: total = prefill (start -> first logits call) + decode.
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
        # A3: 1 prefill forward (chunked: several) + (n - 1) decode forwards.
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

    # ── 特許コアロジック: scaled_dot_product_attention のパッチ ────────

    def _patch_sdpa(self) -> None:
        """
        torch.nn.functional.scaled_dot_product_attention を置き換える。

        attn_mask は logits に加算されてから softmax に渡される。
        つまり M を attn_mask として渡すことで
            attn_scores += w * M   （特許クレームそのもの）
        を実現する。

        Item 2 (ownership): the wrapper always closes over ``_ORIGINAL_SDPA``
        (captured at import), never over the current global, so patching
        twice cannot build a wrapper that calls itself.  A second call by the
        owning instance is a no-op; a call while ANOTHER instance owns the
        global patch raises.
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
            # §53 Exp Q: QK-norm retrofit (Llama/Qwen 救済戦略)
            # 効果: Q/K の "massive values" outliers (Sun et al. 2025: arXiv:2502.01563) を
            #       平準化し pre-softmax logits を compact にする → additive bias の leverage 増
            qk_mode = outer._qk_norm_mode
            if qk_mode != "off":
                d_head = query.shape[-1]
                scale_factor = d_head ** 0.5

                if qk_mode == "l2":
                    # 完全 L2 norm + sqrt(d) re-scale (§54 Pattern E: Llama 破壊)
                    query = F.normalize(query.float(), p=2, dim=-1).to(query.dtype) * scale_factor
                    key = F.normalize(key.float(), p=2, dim=-1).to(key.dtype) * scale_factor

                elif qk_mode == "l2_soft":
                    # α-mix: 部分 normalize で coherence 維持 + outlier 抑制
                    alpha = outer._qk_norm_alpha
                    q_norm = F.normalize(query.float(), p=2, dim=-1).to(query.dtype) * scale_factor
                    k_norm = F.normalize(key.float(), p=2, dim=-1).to(key.dtype) * scale_factor
                    query = alpha * q_norm + (1.0 - alpha) * query
                    key = alpha * k_norm + (1.0 - alpha) * key

                elif qk_mode == "clip":
                    # Outlier のみ shrink: max_norm = threshold × sqrt(d_head) で cap
                    # 正常な Q/K は無変更 → 最小侵襲 retrofit
                    max_norm = outer._qk_clip_threshold * scale_factor
                    q_norms = query.float().norm(dim=-1, keepdim=True).clamp(min=1e-9)
                    k_norms = key.float().norm(dim=-1, keepdim=True).clamp(min=1e-9)
                    q_scale = (max_norm / q_norms).clamp(max=1.0).to(query.dtype)
                    k_scale = (max_norm / k_norms).clamp(max=1.0).to(key.dtype)
                    query = query * q_scale
                    key = key * k_scale

            seq_q = query.shape[-2]
            seq_k = key.shape[-2]
            # A2: decode = the prefill forwards are over AND this is a 1-row
            # query.  Outside generate() ``_prefill_done`` is True (class
            # default), so this reduces to the historical ``seq_q == 1``.
            is_decode = outer._prefill_done and seq_q == 1

            # マス加算バイアスの構築 (2D M行列 / 1D マスベクトル) は
            # モジュールレベルの純関数 build_mass_bias に委譲する。
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
                # Item 1: with a float mask AND enable_gqa=True torch may fall
                # back to the math kernel (float32 promotion at 190k keys);
                # expand K/V here and hand the original enable_gqa=False.
                key, value, kwargs = _expand_gqa_heads(query, key, value, kwargs)
            elif outer._m_matrix is None and outer._mass_vector is not None:
                # F6: どちらの理由で飛ばしたのかを数える
                # (build_mass_bias 内の判定順序と同じ順で見る)
                if not is_decode and outer._prefill_mass_scale <= 0.0:
                    outer.bias_skipped_prefill_calls += 1
                elif seq_k < outer._mass_vector.shape[0]:
                    outer.bias_skipped_sliding_calls += 1

                # H15 (b): the LAST prefill forward (seq_k == prompt_len) gets
                # the bias on its final query row only.  The other rows keep
                # the plain prefill output (and the skip above stays counted).
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

            # B2: opt-in probability recorder. Runs on DECODE calls only
            # (by PHASE, item 3: a 1-token final prefill chunk is not decode),
            # AFTER the mass bias has been merged into attn_mask, so what it
            # records is exactly the distribution the model uses.
            # getattr: verify_attention_math builds the instance via
            # __new__ (no __init__), so the attribute may be absent.
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

        # torch.nn.functional と F の両方を置き換える
        import torch.nn.functional as _F
        _F.scaled_dot_product_attention = patched_sdpa
        torch.nn.functional.scaled_dot_product_attention = patched_sdpa
        _PATCH_OWNER = self
        print("[MassWeightedGemma] patched scaled_dot_product_attention")

    def _prefill_last_row_sdpa(
        self, query, key, value, *, attn_mask, dropout_p, is_causal, scale, kwargs
    ):
        """H15 (b): ordinary prefill output, then the FINAL query row recomputed
        with ``w * mass`` added (a decode row over the full prompt: full weight,
        effective-bias cap, ``is_causal=False`` because the last row may see
        every populated key; an existing mask contributes its last row)."""
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

    # ── B2: attention recorder (phase_instrument only) ────────────────

    def start_attention_recording(self) -> None:
        """Record the softmax row of every DECODE step from now on."""
        self.recorded_attention = []
        self._record_attention = True

    def stop_attention_recording(self) -> list[torch.Tensor]:
        self._record_attention = False
        return self.recorded_attention

    def _record_attention_row(
        self, query, key, attn_mask, scale, enable_gqa: bool
    ) -> None:
        """Append the attention probabilities of the LAST query row.

        Recomputes ``softmax(q @ k^T * scale + attn_mask)`` in float32 for the
        last query position and averages over batch and heads, giving one
        ``(seq_k,)`` probability vector per sdpa call (i.e. per layer per decode
        step).  This is the only honest way to observe the patched model: the
        HF ``output_attentions=True`` path silently switches the model to eager
        attention, which never calls ``F.scaled_dot_product_attention`` and
        therefore measures the UNPATCHED model.

        The mask is folded in with the same bool -> float convention as
        ``combine_attn_mask`` so a boolean causal/padding mask is not turned
        into 1.0/0.0.
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
        """テスト等でパッチを戻したい場合に使う。

        Item 2: only the owning instance restores; it puts back
        ``_ORIGINAL_SDPA`` (the import-time kernel) and clears ownership.  A
        non-owner call is a no-op so it cannot clobber another instance's patch.
        """
        global _PATCH_OWNER
        if _PATCH_OWNER is not self:
            return
        import torch.nn.functional as _F
        _F.scaled_dot_product_attention = _ORIGINAL_SDPA
        torch.nn.functional.scaled_dot_product_attention = _ORIGINAL_SDPA
        _PATCH_OWNER = None

    # ── 互換性確認 ────────────────────────────────────────────────────

    @property
    def attn_implementation(self) -> str | None:
        """モデルが使用している attention 実装名を返す（sdpa/eager/flash_attention_2 等）。
        sdpa であればマス注入パッチが効く。"""
        if self._model is None:
            return None
        cfg = getattr(self._model, "config", None)
        if cfg is None:
            return None
        return getattr(cfg, "_attn_implementation", None)


# 別名（リファクタリングを最小化、新しい実験から MassWeightedLLM として import 可能）
MassWeightedLLM = MassWeightedGemma


# ── ユーティリティ ────────────────────────────────────────────────────

def _expand_gqa_heads(query, key, value, kwargs: dict) -> tuple:
    """Item 1: when the caller asked for ``enable_gqa=True``, repeat the K/V
    heads to the query head count IN THEIR OWN DTYPE (bf16 on the pod) and
    return kwargs with ``enable_gqa=False``, so the original SDPA takes a
    fused kernel instead of the math fallback that promotes to float32.
    Without ``enable_gqa`` everything is returned untouched."""
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
    """is_causal=True の代わりに使うcausalマスク（-inf で未来をマスク）。"""
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
    """softmax 前に加算する additive bias を返す。適用対象がなければ None。

    phase_is_decode (A2, 2026-09-18): the caller's prefill/decode phase.
        None  -> historical heuristic ``seq_q == 1`` means decode.
        True  -> decode step (full weight; a cropped cache still raises).
        False -> prefill forward, even when seq_q == 1 (the 1-token final
                 chunk of a chunked prefill): prefill rule, and with
                 prefill_mass_scale > 0 and seq_k < mass_len (the cache is
                 still growing chunk by chunk) the vector is SLICED to the
                 first seq_k entries instead of raising "sliding".

    bias_cap (D-1, 2026-09-07 Fable review #1): 最終的な実効バイアス
    (w × mass) を bias_cap 以下に丸める。mass_vector.py の cap は mass 側の
    丸めであり、その後に mass_weight を掛けると w=3.0 で 9.0 (e^9 ≈ 8100 倍)
    になっていた。D-1 は「実効 additive bias ≤ ~3.0」なので、丸めは積に掛ける。
    None のときは丸めない (旧動作、テスト用)。

    2D モード (m_matrix): (1, 1, seq_q, seq_k) = mass_weight * M[:seq_q, :seq_k]。
        M が短い場合はゼロパディングする。
    1D モード (mass_vector): (1, 1, 1, seq_k) = effective_w * mass_vector[:seq_k]。
        同じくゼロパディングする。
        effective_w = mass_weight                        (seq_q == 1: デコード, D-2)
                    = mass_weight * prefill_mass_scale   (seq_q > 1 かつ scale > 0)
                    = なし → None を返す                  (それ以外)

    両方セットされている場合は 2D モードが優先される (現行動作)。
    バイアスは 1/sqrt(d) スケール後の logits に加算されるため、
    ここで 1/sqrt(d) を掛けてはならない (特許 §0082)。

    F2 — キャッシュ添字のずれ (strict_alignment):
        1D モードのマスベクトルは「キャッシュ位置」で添字付けされている。
        sliding-window 注意を使うレイヤ (Gemma-3 は全層のうち大半が
        sliding) では KV キャッシュが窓幅に切り詰められ、 seq_k が
        プロンプト長より短くなる。 このとき key j の絶対位置は
        j + kv_offset であり、 SDPA には kv_offset が渡ってこないため、
        mass_vector[j] は無関係なトークンに落ちる。
        そこで seq_k < mass_vector.shape[0] を「切り詰められたキャッシュ」
        の検出条件とし、
          strict_alignment=True  → RuntimeError を送出 (既定)
          strict_alignment=False → None を返してそのレイヤは飛ばす
        とする。

        対応対象は全層が full_attention の密なモデル (Qwen3.x 系) である。
        過去の Gemma-3 長コンテキストのマス注入実験は、 sliding レイヤで
        バイアスが無関係なトークンに乗っていたため影響を受けている。
    """
    if m_matrix is not None:
        # 2D M行列モード（短コンテキスト向け）
        m_q = min(seq_q, m_matrix.shape[0])
        m_k = min(seq_k, m_matrix.shape[1])
        m_slice = mass_weight * m_matrix[:m_q, :m_k]

        # KVキャッシュ成長でM行列サイズを超えた場合はゼロパディング
        if m_q < seq_q or m_k < seq_k:
            full = torch.zeros(seq_q, seq_k, dtype=torch.float32, device=m_matrix.device)
            full[:m_q, :m_k] = m_slice
            m_slice = full

        # (seq_q, seq_k) → (1, 1, seq_q, seq_k)
        if bias_cap is not None:
            m_slice = m_slice.clamp(max=bias_cap)
        return m_slice.to(dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)

    if mass_vector is not None:
        # 1D マスベクトルモード（長コンテキスト向け、メモリ効率的）
        # デフォルト: デコードステップ (seq_q==1) のみ適用。
        # 過去の経緯: プリフィル時に full mass_weight で加算すると全18層で
        # 表現が崩壊し、出力が繰り返し文字列になる現象を確認した。
        # §49.2 乖離③: prefill_mass_scale > 0.0 のとき prefill 中も
        # 部分スケールで適用する（特許 §0082 完全準拠への段階的アプローチ）。
        effective_w: float | None = None
        is_decode = (seq_q == 1) if phase_is_decode is None else bool(phase_is_decode)
        if is_decode:
            # デコードステップ: フルウェイト適用 (現行動作)
            effective_w = mass_weight
        elif prefill_mass_scale > 0.0:
            # プリフィル: 部分スケール適用 (実験的)
            effective_w = mass_weight * prefill_mass_scale

        if effective_w is None:
            return None

        # F2: 切り詰められた (sliding-window) キャッシュの検出
        # A2: an EXPLICIT prefill forward with a shorter cache is chunked
        # prefill (positions 0..seq_k-1 are exactly the cached ones), so the
        # vector is sliced; the raise stays for decode / unknown phase.
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

        # (seq_k,) → (1, 1, 1, seq_k) でブロードキャスト
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
    """m_bias を attn_mask に合成し、(attn_mask, is_causal) を返す。

    attn_mask が None かつ is_causal=True のとき:
        float attn_mask と is_causal=True は併用できないため、causal mask を
        m_bias に加算した上で is_causal=False にして返す。
    attn_mask がある場合:
        attn_mask.to(dtype) + m_bias を返す。

    F1 — bool マスクの取り扱い:
        transformers 5.8 の sdpa 経路 (masking_utils.sdpa_mask) は
        **torch.bool** の 4D マスク (True = 注意してよい) を渡してくる。
        これを .to(dtype) すると True→1.0 / False→0.0 になり、
        causal / padding / sliding のマスクが完全に消えてしまう
        (禁止されたキーに 0.0 のバイアスしか乗らず、 attention が漏れる)。
        そこで bool のときは transformers の eager 経路と同じ規約で
        float マスクへ変換してから加算する:
            zeros.masked_fill(~attn_mask, torch.finfo(dtype).min)
        -inf ではなく finfo.min を使うのは eager_mask と揃えるため
        (全キーが禁止された行で NaN にならない)。
        float マスクの経路は従来どおり変更なし。

    F6 — 無音スキップの廃止:
        shape 不一致は以前 except RuntimeError: pass で握り潰され、
        マス注入が効いていないまま実験が回っていた。 いまは shape を
        載せた RuntimeError を送出して落とす。
    """
    if attn_mask is None:
        # is_causal=True と float attn_mask は併用できないため
        # is_causal フラグを落として causal mask を m_bias に含める
        if is_causal:
            causal_mask = _make_causal_mask(seq_q, seq_k, dtype, device)
            m_bias = m_bias + causal_mask
            is_causal = False
        return m_bias, is_causal

    if attn_mask.dtype == torch.bool:
        # True = 注意してよい / False = 禁止。 eager_mask と同じ規約で float 化する。
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
