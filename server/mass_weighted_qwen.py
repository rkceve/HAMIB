"""
MassWeightedQwen: Qwen 系モデル専用のマスインジェクション実装。

【MassWeightedGemma との違い】
1. Qwen 2.5 の chat template を `tokenizer.apply_chat_template` で適用
2. `generation_config.max_length` を制限 (デフォルトの 128K だと KV cache を事前確保して
   VRAM 不足/速度低下を引き起こすため、入力長 + max_new_tokens に制約)
3. trial 間の KV cache 明示クリア
4. Qwen 特有の pad_token / eos_token 設定

【共通する部分（MassWeightedGemma 由来）】
- scaled_dot_product_attention の monkey-patch
- 1D mass vector 注入 (seq_q==1 ガード)
- 2D M 行列注入

使い方:
  from server.mass_weighted_qwen import MassWeightedQwen
  m = MassWeightedQwen(model_id="Qwen/Qwen2.5-3B-Instruct")
  m.load()
  m.set_mass_vector(vec)
  out = m.chat([{"role": "user", "content": "hello"}])  # chat template で送信
  # または
  out = m.generate("raw prompt")  # raw text 送信（bypass chat template）
"""
from __future__ import annotations
import gc
from pathlib import Path
import torch

from server.mass_weighted_gemma import MassWeightedGemma


class MassWeightedQwen(MassWeightedGemma):
    """
    Qwen 系モデル (Qwen 2.5 / Qwen 3 等) 専用の MassWeightedLLM。
    親クラスの sdpa パッチを継承しつつ、Qwen 固有の chat template と
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
        # デフォルト model_id を Qwen 2.5-1.5B に
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
        # Qwen 固有の generation_config 修正:
        # デフォルト max_length=131072 (128K) は KV cache を事前確保して
        # 6GB VRAM では VRAM 不足/極端な速度低下を引き起こすため制限
        gen_cfg = self._model.generation_config
        if hasattr(gen_cfg, "max_length"):
            # max_length は実行時に input_len + max_new_tokens で動的に上書きされる
            # ここでは安全な上限として 16K に設定
            gen_cfg.max_length = 16384
        # pad_token_id を eos に揃える (Qwen は pad_token が None の場合あり)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            gen_cfg.pad_token_id = self._tokenizer.eos_token_id
        print(f"[MassWeightedQwen] generation_config.max_length capped at {gen_cfg.max_length}")

    def chat(self, messages: list[dict]) -> str:
        """
        Qwen の chat template を適用して生成。
        messages は OpenAI 互換形式: [{"role": "user|assistant|system", "content": "..."}, ...]
        """
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.generate(prompt)

    def generate(self, prompt: str) -> str:
        """
        親クラスの generate を上書き: max_length を入力長に応じて動的設定する。
        これによって KV cache 事前確保による VRAM 浪費を防ぐ。
        """
        target_device = self._device
        try:
            target_device = next(self._model.parameters()).device
        except Exception:
            pass

        inputs = self._tokenizer(prompt, return_tensors="pt").to(target_device)
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]

        # Qwen 固有: max_length を動的に調整 (入力長 + 生成長 + マージン)
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

        # Qwen 固有: trial 間で KV cache 等の中間メモリを明示クリア
        del inputs, input_ids, output_ids, new_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return result
