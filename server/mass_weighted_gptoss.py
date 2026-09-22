"""
MassWeightedGPTOSS: openai/gpt-oss-20b / gpt-oss-120b 用ラッパー

【MassWeightedGemma との違い】
  1. native MXFP4 量子化済みなので bitsandbytes 不要
  2. _supports_sdpa = False → SDPA monkey-patch では mass injection 不可
     → eager_attention_forward を直接 monkey-patch する
  3. Harmony chat template が必須 (raw "User:" だと繰り返し出力)
  4. attention output に "sink" 列が含まれる (sink を保持しつつ mass を加算)
  5. tokenizer: o200k_harmony (vocab ~200K)

【mass injection 仕組み (eager 版)】
  GPT-OSS の eager_attention_forward:
      attn_weights = Q @ K^T * scaling
      attn_weights += attention_mask          ← ここで mass を足し込む
      combined = cat([attn_weights, sinks])
      probs = softmax(combined)
      scores = probs[..., :-1]                 ← sink を落とす

  パッチ版:
      attn_weights += attention_mask + m_bias  ← attn_mask に上乗せ
      (以下同じ)

【chat template 適用】
  - prompt が "User: ...\nAssistant:" 形式 (CMSSession 由来) のときは
    自動的に [{"role":"user","content":...}] に変換して chat_template を適用。
  - context_block (<CONTEXT>...) が含まれる場合は system prompt として分離。
"""
from __future__ import annotations
from pathlib import Path
import gc
import re
import torch
import torch.nn.functional as F

from server.mass_weighted_gemma import MassWeightedGemma


class MassWeightedGPTOSS(MassWeightedGemma):
    """openai/gpt-oss-* 用 wrapper (eager attention patch + chat template)"""

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

    # ── ロード ────────────────────────────────────────────────────────

    def load(self) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        # GPT-OSS は native MXFP4 量子化済み → BitsAndBytesConfig 不要
        # torch_dtype="auto" で MXFP4 が triton 経由で適用される
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

        # SDPA は使えないので、 eager_attention_forward を patch する
        self._patch_eager_attention()
        print(f"[MassWeightedGPTOSS] loaded: {self._model_id}")

    # ── eager attention monkey-patch ────────────────────────────────────

    def _patch_eager_attention(self) -> None:
        """transformers.models.gpt_oss.modeling_gpt_oss.eager_attention_forward を差し替える。

        マス注入: attention_mask に m_bias を加算してから親実装に委譲する。
        sink 列の処理 (cat + softmax + drop sink) はオリジナル実装に任せる。
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
            # mass を attention_mask に上乗せして親実装に委譲
            seq_q = query.shape[-2]
            # GPT-OSS の eager は K に repeat_kv をかけてから Q @ K^T を行うが、
            # この時点での key shape は (B, n_kv_heads, seq_k, d_head) なので
            # seq_k = key.shape[-2] が正しい長さ
            seq_k = key.shape[-2] * module.num_key_value_groups \
                if hasattr(module, "num_key_value_groups") else key.shape[-2]
            # ↑ ただし attention_mask は (B,1,seq_q,seq_k_orig) 形式で渡されてくる。
            #   seq_k_orig は key.shape[-2] そのまま (repeat_kv 後 ではなく前)。
            #   なので m_bias のサイズも key.shape[-2] に合わせる。
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
        # ALL_ATTENTION_FUNCTIONS にも eager として登録されているので
        # そちらも差し替える
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            ALL_ATTENTION_FUNCTIONS.register("eager", patched_eager)
        except Exception:
            pass
        print("[MassWeightedGPTOSS] patched eager_attention_forward")

    def restore_sdpa(self) -> None:
        # GPT-OSS では eager を戻す
        if self._original_eager is not None:
            from transformers.models.gpt_oss import modeling_gpt_oss as mgo
            mgo.eager_attention_forward = self._original_eager

    # ── 推論 (chat template 自動適用) ────────────────────────────────────

    # "User: ...\nAssistant:" を末尾から抽出する正規表現。
    # 直前までの内容 (system 部分) は別途取り出すので、 ここでは User: 以降だけ
    # 非貪欲マッチで掴む。
    _USER_TAIL = re.compile(
        r"(?P<head>[\s\S]*?)User:\s*(?P<u>[\s\S]*?)\n\s*Assistant:\s*$"
    )

    def _to_chat_template(self, prompt: str) -> str:
        """CMSSession 由来の "User:...\nAssistant:" prompt を Harmony 形式へ変換。

        - 直前のテキスト (アシスタント説明 + <CONTEXT> ブロック等) は system message へ。
        - 変換できない prompt は raw prompt をそのまま返す。
        - GPT-OSS の Harmony chat template は reasoning_effort kwarg をサポート。
          デフォルトの "medium" だと analysis channel が長くなり max_new_tokens を
          食いつぶすため、 "low" に下げる。
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
            # まず reasoning_effort 付きで試す。 サポートされていないテンプレ用に
            # 1 段フォールバック。
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

        # CMSSession 形式の prompt は Harmony 形式に変換
        full_prompt = self._to_chat_template(prompt)
        inputs = self._tokenizer(full_prompt, return_tensors="pt").to(target_device)
        input_ids = inputs["input_ids"]

        # mass_vector が外部からセットされている場合、 chat_template 適用後の
        # 実際のトークン列に対して [PN{mass}] 位置を再計算する。
        # (CMSSession は変換前の prompt で位置を計算しているため、 そのままでは
        #  位置がずれて mass が間違ったトークンに適用される)
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


# ── ヘルパ ───────────────────────────────────────────────────────────────


def _build_m_bias(
    outer: MassWeightedGPTOSS,
    seq_q: int,
    seq_k: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """1D mass vec / 2D M matrix を attention_mask 加算用 bias テンソルにする。

    Returns shape (1,1,seq_q,seq_k) or (1,1,1,seq_k)、 または None。
    """
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
        return m_slice.to(dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)

    if mass_vec is not None:
        # decode-only ガード (prefill では崩壊する経緯あり)
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


# Harmony channel マーカー (skip_special_tokens 後): "assistantfinal..." または
# "assistantanalysis...assistantfinal..." のように 1 行に並ぶことが多い。
# 行頭制約を外して中央でも捕まえる。
_CHANNEL_FINAL_PATTERN = re.compile(
    r"(?:assistantfinal|<\|channel\|>\s*final\s*<\|message\|>)\s*"
    r"(?P<body>[\s\S]+?)"
    r"(?=(?:<\|return\|>|<\|end\|>|assistantanalysis|$))",
)


def _strip_harmony_channels(text: str) -> str:
    """GPT-OSS Harmony 出力から最終 (final channel) 応答だけ抽出する。

    フォーマット例 (skip_special_tokens=True 後):
        "assistantanalysis<thinking text>assistantfinal<answer>"
        "assistantfinal<answer>"
        "<answer>" (final channel タグなし、 ストレート出力)
    """
    m = _CHANNEL_FINAL_PATTERN.search(text)
    if m:
        return m.group("body").strip()
    # final マーカーが無い場合、 先頭の analysis セクションだけ除去
    # "assistantanalysis<...>" を頭から削る
    cleaned = re.sub(r"^(?:assistantanalysis|analysis)[\s\S]*", "", text)
    return cleaned.strip() or text.strip()
