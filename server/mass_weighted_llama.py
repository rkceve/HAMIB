"""
MassWeightedLlama: mass-injection implementation for Llama models (Llama 3.x, Llama 3.2).

Differences from MassWeightedGemma:
1. Applies the Llama 3 chat template via `tokenizer.apply_chat_template`.
2. Limits `generation_config.max_length` dynamically instead of 128K
   (Llama 3.2 defaults to max_length=131072, which OOMs on 6GB VRAM).
3. Explicitly clears the KV cache between trials.
4. Llama-specific pad_token setup.

Usage:
  from server.mass_weighted_llama import MassWeightedLlama
  m = MassWeightedLlama(model_id="unsloth/Llama-3.2-1B-Instruct")
  m.load()
  out = m.generate("raw prompt")
"""
from __future__ import annotations
import gc
import torch

from server.mass_weighted_gemma import MassWeightedGemma


class MassWeightedLlama(MassWeightedGemma):
    """
    MassWeightedLLM for Llama models (Llama 3.x, Llama 3.2, etc.).
    Inherits the parent's sdpa patch while applying Llama-specific chat
    template and generation config.
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
        if model_id is None:
            model_id = "unsloth/Llama-3.2-1B-Instruct"
        super().__init__(
            config_path=config_path,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

    def load(self) -> None:
        super().load()
        gen_cfg = self._model.generation_config
        # Llama 3.2 defaults to max_length=131072, which OOMs on a 6GB GPU.
        # Cap it at 16K as a safe upper bound.
        if hasattr(gen_cfg, "max_length"):
            gen_cfg.max_length = 16384
        # Llama may have pad_token set to None.
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            gen_cfg.pad_token_id = self._tokenizer.eos_token_id
        print(f"[MassWeightedLlama] generation_config.max_length capped at {gen_cfg.max_length}")

    def chat(self, messages: list[dict]) -> str:
        """Generate by applying the Llama 3 chat template."""
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.generate(prompt)

    def generate(self, prompt: str) -> str:
        """Set max_length dynamically based on input length and clear VRAM between trials."""
        target_device = self._device
        try:
            target_device = next(self._model.parameters()).device
        except Exception:
            pass

        inputs = self._tokenizer(prompt, return_tensors="pt").to(target_device)
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]
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

        del inputs, input_ids, output_ids, new_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return result
