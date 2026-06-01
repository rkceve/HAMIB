"""
MassWeightedGPTOSS: wrapper for openai/gpt-oss-20b / gpt-oss-120b.

Differences from MassWeightedGemma:
  1. Already MXFP4-quantized natively, so bitsandbytes is not needed.
  2. _supports_sdpa = False, so the SDPA monkey-patch cannot inject mass.
     Instead, eager_attention_forward is monkey-patched directly.
  3. The Harmony chat template is required (a raw "User:" prompt produces
     repeated output).
  4. The attention output includes a "sink" column (mass is added while the
     sink is preserved).
  5. tokenizer: o200k_harmony (vocab ~200K)

Mass injection (eager version):
  GPT-OSS's eager_attention_forward:
      attn_weights = Q @ K^T * scaling
      attn_weights += attention_mask          # mass is added here
      combined = cat([attn_weights, sinks])
      probs = softmax(combined)
      scores = probs[..., :-1]                 # drop the sink

  Patched version:
      attn_weights += attention_mask + m_bias  # add on top of attn_mask
      (everything else is the same)

Chat template:
  - When a prompt is in "User: ...\nAssistant:" form (from HAMIBSession), it is
    automatically converted to [{"role":"user","content":...}] and the
    chat_template is applied.
  - When a context block (<CONTEXT>...) is present, it is separated out as the
    system prompt.
"""
from __future__ import annotations
import gc
import re
import torch

from server.mass_weighted_gemma import MassWeightedGemma


class MassWeightedGPTOSS(MassWeightedGemma):
    """Wrapper for openai/gpt-oss-* (eager attention patch + chat template)."""

    def __init__(
        self,
        config_path=None,
        *,
        model_id: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        do_sample: bool | None = None,
    ):
        if model_id is None:
            model_id = "openai/gpt-oss-20b"
        super().__init__(
            config_path=config_path,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        self._original_eager = None

    # ── Load ──────────────────────────────────────────────────────────

    def load(self) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        # GPT-OSS is natively MXFP4-quantized, so BitsAndBytesConfig is not needed.
        # torch_dtype="auto" applies MXFP4 via triton.
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            device_map="auto",
            torch_dtype="auto",
        )
        self._model.eval()

        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            try:
                self._model.generation_config.pad_token_id = self._tokenizer.eos_token_id
            except Exception:
                pass

        # SDPA is unavailable, so patch eager_attention_forward instead.
        self._patch_eager_attention()
        print(f"[MassWeightedGPTOSS] loaded: {self._model_id}")

    # ── eager attention monkey-patch ────────────────────────────────────

    def _patch_eager_attention(self) -> None:
        """Replace transformers.models.gpt_oss.modeling_gpt_oss.eager_attention_forward.

        Mass injection: add m_bias to attention_mask, then delegate to the
        original implementation. The sink-column handling
        (cat + softmax + drop sink) is left to the original implementation.
        """
        from transformers.models.gpt_oss import modeling_gpt_oss as mgo
        outer = self
        self._original_eager = mgo.eager_attention_forward

        def patched_eager(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling: float,
            dropout=0.0,
            **kwargs,
        ):
            # Add mass on top of attention_mask, then delegate to the original.
            seq_q = query.shape[-2]
            # attention_mask is passed in as (B, 1, seq_q, seq_k_orig), where
            # seq_k_orig is key.shape[-2] (before repeat_kv, not after), so size
            # m_bias to match key.shape[-2].
            seq_k_mask = key.shape[-2]

            m_bias = _build_m_bias(
                outer, seq_q, seq_k_mask, query.dtype, query.device
            )
            new_mask = attention_mask
            if m_bias is not None:
                if new_mask is None:
                    new_mask = m_bias
                else:
                    try:
                        new_mask = new_mask + m_bias.to(dtype=new_mask.dtype)
                    except RuntimeError:
                        pass

            return outer._original_eager(
                module, query, key, value, new_mask,
                scaling=scaling, dropout=dropout, **kwargs,
            )

        mgo.eager_attention_forward = patched_eager
        # It is also registered as "eager" in ALL_ATTENTION_FUNCTIONS, so
        # replace that entry too.
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            ALL_ATTENTION_FUNCTIONS.register("eager", patched_eager)
        except Exception:
            pass
        print("[MassWeightedGPTOSS] patched eager_attention_forward")

    def restore_sdpa(self) -> None:
        # For GPT-OSS, restore the eager implementation.
        if self._original_eager is not None:
            from transformers.models.gpt_oss import modeling_gpt_oss as mgo
            mgo.eager_attention_forward = self._original_eager

    # ── Inference (auto-applies chat template) ──────────────────────────

    # Regex that extracts "User: ...\nAssistant:" from the tail. The preceding
    # content (the system part) is captured separately, so here we only match
    # the text after "User:" non-greedily.
    _USER_TAIL = re.compile(
        r"(?P<head>[\s\S]*?)User:\s*(?P<u>[\s\S]*?)\n\s*Assistant:\s*$"
    )

    def _to_chat_template(self, prompt: str) -> str:
        """Convert a HAMIBSession "User:...\nAssistant:" prompt to Harmony format.

        - The preceding text (assistant instructions, <CONTEXT> block, etc.)
          becomes the system message.
        - A prompt that cannot be converted is returned unchanged.
        - The GPT-OSS Harmony chat template supports a reasoning_effort kwarg.
          The default "medium" makes the analysis channel long and consumes
          max_new_tokens, so it is lowered to "low".
        """
        try:
            m = self._USER_TAIL.search(prompt)
            if m is None:
                return prompt
            user_msg = m.group("u").strip()
            head = (m.group("head") or "").strip()
            messages = []
            if head:
                messages.append({"role": "system", "content": head})
            messages.append({"role": "user", "content": user_msg})
            # Try with reasoning_effort first, with a one-step fallback for
            # templates that do not support it.
            try:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    reasoning_effort="low",
                )
            except TypeError:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        except Exception:
            return prompt

    def generate(self, prompt: str) -> str:
        target_device = self._device
        try:
            target_device = next(self._model.parameters()).device
        except Exception:
            pass

        # Convert a HAMIBSession-style prompt to Harmony format.
        full_prompt = self._to_chat_template(prompt)
        inputs = self._tokenizer(full_prompt, return_tensors="pt").to(target_device)
        input_ids = inputs["input_ids"]

        # If a mass_vector was set externally, recompute the [PN{mass}] positions
        # against the actual token sequence produced by the chat_template.
        # (HAMIBSession computes positions on the pre-conversion prompt, so without
        #  this the positions would shift and mass would apply to the wrong tokens.)
        if self._mass_vector is not None and full_prompt != prompt:
            from server.cd_parser import find_pn_positions
            pn = find_pn_positions(input_ids[0].tolist(), self._tokenizer)
            if pn:
                seq_len = input_ids.shape[1]
                new_vec = torch.zeros(
                    seq_len, dtype=torch.float32, device=input_ids.device,
                )
                for pos, mass in pn:
                    if 0 <= pos < seq_len:
                        new_vec[pos] += mass
                self._mass_vector = new_vec
            else:
                self._mass_vector = None

        gen_kwargs: dict = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": self._do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if self._do_sample:
            gen_kwargs["temperature"] = self._temperature

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_ids = output_ids[0, input_ids.shape[1]:]
        result = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        result = _strip_harmony_channels(result)

        del inputs, input_ids, output_ids, new_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return result


# ── Helpers ──────────────────────────────────────────────────────────────


def _build_m_bias(
    outer: MassWeightedGPTOSS,
    seq_q: int,
    seq_k: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """Turn the 1D mass vec / 2D M matrix into a bias tensor to add to attention_mask.

    Returns shape (1,1,seq_q,seq_k) or (1,1,1,seq_k), or None.
    """
    M = outer._m_matrix
    mass_vec = outer._mass_vector

    if M is not None:
        # Unused in the default path: outer._m_matrix is never set
        # (no caller of set_m_matrix() exists). 2D M-matrix mode is kept
        # for short-context experiments only.
        m_q = min(seq_q, M.shape[0])
        m_k = min(seq_k, M.shape[1])
        m_slice = outer._mass_weight * M[:m_q, :m_k]
        if m_q < seq_q or m_k < seq_k:
            full = torch.zeros(seq_q, seq_k, dtype=torch.float32, device=M.device)
            full[:m_q, :m_k] = m_slice
            m_slice = full
        return m_slice.to(dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)

    if mass_vec is not None:
        # decode-only guard (applying mass during prefill can collapse output)
        effective_w: float | None = None
        if seq_q == 1:
            effective_w = outer._mass_weight
        elif outer._prefill_mass_scale > 0.0:
            effective_w = outer._mass_weight * outer._prefill_mass_scale
        if effective_w is None:
            return None

        m_k = min(seq_k, mass_vec.shape[0])
        m_vec = effective_w * mass_vec[:m_k]
        if m_k < seq_k:
            full_vec = torch.zeros(seq_k, dtype=torch.float32, device=mass_vec.device)
            full_vec[:m_k] = m_vec
            m_vec = full_vec
        return m_vec.to(dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    return None


# Harmony channel markers (after skip_special_tokens) often appear on one line,
# e.g. "assistantfinal..." or "assistantanalysis...assistantfinal...".
# Drop the line-start anchor so they are also matched mid-line.
_CHANNEL_FINAL_PATTERN = re.compile(
    r"(?:assistantfinal|<\|channel\|>\s*final\s*<\|message\|>)\s*"
    r"(?P<body>[\s\S]+?)"
    r"(?=(?:<\|return\|>|<\|end\|>|assistantanalysis|$))",
)


def _strip_harmony_channels(text: str) -> str:
    """Extract only the final-channel response from GPT-OSS Harmony output.

    Example formats (after skip_special_tokens=True):
        "assistantanalysis<thinking text>assistantfinal<answer>"
        "assistantfinal<answer>"
        "<answer>" (no final-channel tag, straight output)
    """
    m = _CHANNEL_FINAL_PATTERN.search(text)
    if m:
        return m.group("body").strip()
    # If there is no final marker, strip the leading analysis section
    # ("assistantanalysis<...>") from the front.
    cleaned = re.sub(r"^(?:assistantanalysis|analysis)[\s\S]*", "", text)
    return cleaned.strip() or text.strip()
