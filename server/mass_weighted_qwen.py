"""
MassWeightedQwen: mass injection for Qwen-family models.

Differences from MassWeightedGemma:
1. Applies the Qwen 2.5 chat template via `tokenizer.apply_chat_template`.
2. Limits `generation_config.max_length` (the default 128K pre-allocates the KV
   cache and causes VRAM shortage and slowdowns), bounding it by input length +
   max_new_tokens.
3. Clears the KV cache explicitly between trials.
4. Qwen-specific pad_token / eos_token settings.

Shared with MassWeightedGemma:
- scaled_dot_product_attention monkey-patch
- 1D mass vector injection (seq_q==1 guard)
- 2D M matrix injection

Usage:
  from server.mass_weighted_qwen import MassWeightedQwen
  m = MassWeightedQwen(model_id="Qwen/Qwen2.5-3B-Instruct")
  m.load()
  m.set_mass_vector(vec)
  out = m.chat([{"role": "user", "content": "hello"}])  # via the chat template
  # or
  out = m.generate("raw prompt")  # raw text (bypasses the chat template)
"""
from __future__ import annotations
import gc
from pathlib import Path
import torch

from server.mass_weighted_gemma import MassWeightedGemma


class MassWeightedQwen(MassWeightedGemma):
    """
    MassWeightedLLM for Qwen-family models (Qwen 2.5 / Qwen 3, ...).
    Inherits the parent's SDPA patch and adds the Qwen chat template and
    generation config.
    """

    def __init__(
        self,
        config_path=None,
        *,
        model_id: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        do_sample: bool | None = None,
    ):
        # Default model_id: Qwen 2.5-1.5B
        if model_id is None:
            model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        super().__init__(
            config_path=config_path,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

    def load(self) -> None:
        super().load()
        # Qwen-specific generation_config fix: the default max_length=131072 (128K)
        # pre-allocates the KV cache, which on 6GB of VRAM runs out of memory or
        # slows down badly, so it is capped.
        gen_cfg = self._model.generation_config
        if hasattr(gen_cfg, "max_length"):
            # generate() overrides max_length per call with input_len + max_new_tokens;
            # 16K here is only a safe upper limit.
            gen_cfg.max_length = 16384
        # Use eos as pad_token_id (Qwen's pad_token may be None)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            gen_cfg.pad_token_id = self._tokenizer.eos_token_id
        print(f"[MassWeightedQwen] generation_config.max_length capped at {gen_cfg.max_length}")

    def chat(self, messages: list[dict]) -> str:
        """
        Generate with the Qwen chat template applied.
        messages use the OpenAI format: [{"role": "user|assistant|system", "content": "..."}, ...]
        """
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.generate(prompt)

    def generate(self, prompt: str) -> str:
        """
        Override of the parent generate(): sets max_length from the input length,
        so no VRAM is wasted on a pre-allocated KV cache.
        """
        target_device = self._device
        try:
            target_device = next(self._model.parameters()).device
        except Exception:
            pass

        inputs = self._tokenizer(prompt, return_tensors="pt").to(target_device)
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]

        # Qwen-specific: max_length = input length + generation length + margin
        dynamic_max_length = input_len + self._max_new_tokens + 16

        gen_kwargs: dict = {
            "max_new_tokens": self._max_new_tokens,
            "max_length": dynamic_max_length,
            "do_sample": self._do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if self._do_sample:
            gen_kwargs["temperature"] = self._temperature

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_ids = output_ids[0, input_ids.shape[1]:]
        result = self._tokenizer.decode(new_ids, skip_special_tokens=True)

        # Qwen-specific: free the KV cache and other intermediates between trials
        del inputs, input_ids, output_ids, new_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return result
