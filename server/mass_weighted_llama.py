"""
MassWeightedLlama: Llama 系モデル (Llama 3.x, Llama 3.2) 専用のマスインジェクション実装。

【MassWeightedGemma との違い】
1. Llama 3 の chat template を `tokenizer.apply_chat_template` で適用
2. `generation_config.max_length` を 128K → 動的に制限
   (Llama 3.2 の max_length=131072 がデフォルト → 6GB VRAM では OOM)
3. trial 間の KV cache 明示クリア
4. Llama 特有の pad_token 設定

使い方:
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
    Llama 系モデル (Llama 3.x, Llama 3.2 等) 専用の MassWeightedLLM。
    親クラスの sdpa パッチを継承しつつ、Llama 固有の chat template と
    generation config を適用する。
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
        # Llama 3.2 のデフォルト max_length=131072 → 6GB GPU では OOM
        # 安全な上限として 16K に制限
        if hasattr(gen_cfg, "max_length"):
            gen_cfg.max_length = 16384
        # Llama は pad_token が None の場合あり
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            gen_cfg.pad_token_id = self._tokenizer.eos_token_id
        print(f"[MassWeightedLlama] generation_config.max_length capped at {gen_cfg.max_length}")

    def chat(self, messages: list[dict]) -> str:
        """Llama 3 の chat template を適用して生成。"""
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.generate(prompt)

    def generate(self, prompt: str) -> str:
        """max_length を入力長に応じて動的設定し、trial 間の VRAM クリアを実施。"""
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
