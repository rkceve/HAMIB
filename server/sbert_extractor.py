"""
SBERT cosine + regex hybrid extractor (cms_session._llm_extract_fn の代替)

【§74 Phase A 実装】
  §72 で recall 失敗 (T3 で 0 spans) を診断: SBERT threshold が会話調短文に
  対し too strict。修正:
    A-0: threshold 緩和 (0.4 → 0.2)
    A-0: regex で NAME + CODE pair を一次抽出 (確実な fact 検出)
    A-1: NMS で overlapping spans を 1 件に集約
    A-3: NAME + CODE 共起 → 同一 chunk 内 pair として 1 dict にまとめる
         (parent_hint を NAME に設定して satellite 維持)

【インターフェイス】
  callable: (text: str) -> list[dict]
    dict 構造: {"text": str, "level": "sun"|"planet"|"satellite", "parent_hint": str}

【Phase A 出力ポリシー】
  - regex で fact pair (NAME + CODE) を検出した場合:
    * 1 dict per fact、level=sun、text="NAME → CODE" 形式
  - regex で pair が見つからない場合は SBERT span のみ:
    * NMS + 部分一致除去後、level=satellite、parent_hint=""
"""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Callable

# POSIX/Windows 両対応: backslash literal は Linux で expanduser されないため
# os.path.join で組み立てて HOME を確実に解決する。
os.environ.setdefault(
    "HF_HOME",
    os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
)


DEFAULT_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_QUERIES = [
    "識別コードと対応値のペア",
    "コード ABC の値は XYZ",
    "alphanumeric identifier code",
    "data value",
    "重要な事実情報",
    "プロジェクト名とコード",  # 会話調用に追加
    "project code identifier",
]

# A-0: fact pair 抽出用 regex
#   日本語混在対応のため \b は使わず明示的に前後の非英字境界をマッチ
#   NAME: 大文字始まり + 必ず小文字を含む (Alpha/Beta/Gamma 等を識別、CRANE は除外)
#   CODE: 英大文字 2+ + ハイフン + 数字 (CRANE-1, ABC-100 等、全 caps)
NAME_PATTERN = re.compile(r'(?:^|[^A-Za-z])([A-Z][a-z]+[A-Za-z]*)(?![A-Za-z])')
CODE_PATTERN = re.compile(r'(?:^|[^A-Za-z0-9-])([A-Z]{2,}-\d+)(?![A-Za-z0-9])')
# 一般的すぎる words は除外
_NAME_STOPLIST = {
    "User", "Assistant", "System", "Hi", "Hello", "Yes", "No", "JSON",
    "ID", "Code", "Name", "Project", "BERT", "SBERT", "LLM", "API",
    "Phase", "Turn", "Total", "RESP", "USER", "TEST", "DEBUG",
}


def _find_fact_pairs(text: str, name_window: int = 80) -> list[dict]:
    """
    NAME と CODE が proximity 内に共起する pair を fact として抽出。

    name_window: NAME と CODE の最大文字距離 (80 char = 同一文/節想定)
    """
    if not text:
        return []
    names = []
    for m in NAME_PATTERN.finditer(text):
        n = m.group(1)
        if n in _NAME_STOPLIST:
            continue
        # 大文字 100% は除外 (ALPHA 等の identifier-only も避ける — でも Phase A では拾いたい)
        # → 残す方針: 後段 dedupe で対応
        names.append((m.start(), n))
    codes = [(m.start(), m.group(1)) for m in CODE_PATTERN.finditer(text)]
    if not names or not codes:
        return []

    facts: list[dict] = []
    used_codes = set()
    for code_pos, code in codes:
        if code in used_codes:
            continue
        # 最近接 name を探す (CODE より前 or 直後を許容)
        nearest = None
        nearest_dist = name_window + 1
        for name_pos, name in names:
            dist = abs(code_pos - name_pos)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = name
        if nearest and nearest_dist <= name_window:
            # §80.9 (2026-05-16) fix: 旧 "{name} の対応コード: {code}" 形式は
            # all-MiniLM-L6-v2 で Mu/Nu/Tau/Chi の短 NAME 間 cosine sim が
            # 0.92+ に達して GraphMerger で誤 merge していた (L2 で 3 miss、 L3 でも同様)。
            # "{name}: {code}" 形式は max sim 0.78 で安全、 LLM 可読性も維持。
            facts.append({
                "name": nearest,
                "code": code,
                "text": f"{nearest}: {code}",
                "level": "sun",
                "parent_hint": "",
            })
            used_codes.add(code)
    return facts


class SBERTExtractor:
    """sliding-window cosine + regex hybrid extractor"""

    def __init__(self,
                 model_id: str = DEFAULT_MODEL_ID,
                 window_size: int = 30,
                 step: int = 5,
                 threshold: float = 0.2,  # Phase A: 0.4 → 0.2
                 queries: list[str] | None = None,
                 default_level: str = "satellite",
                 enable_regex_hybrid: bool = True,
                 name_window: int = 80,
                 max_sbert_spans_per_call: int = 3,
                 regex_only: bool = False):
        # §78.9 Gemma 3n 修正: SBERT span が過剰抽出 (1 turn で 5-10 spans) され、
        # turn 累積で cd=99 (cms_g3n_g3n cd=34 の 3 倍) → mass vector で 99 個
        # の attention bias が積み上がり Gemma 3n の per_layer_input 構造で
        # output 完全破綻 (cms_g3n_sbert recall 0/10)。
        # → 1 turn あたり SBERT span 抽出を top-3 max_score で打ち切り、
        #   cd_max を §77 と同等域 (30-40 程度) に抑える。
        # regex_facts は確実な fact 抽出なので上限外 (1 turn で 1-2 件)
        self.model_id = model_id
        self.window_size = window_size
        self.step = step
        self.threshold = threshold
        self.queries = queries or DEFAULT_QUERIES
        self.default_level = default_level
        self.enable_regex_hybrid = enable_regex_hybrid
        self.name_window = name_window
        self.max_sbert_spans_per_call = max_sbert_spans_per_call
        # §80.9 L3 root-cause fix:
        # regex_only=True → SBERT sliding window を完全に無効化。
        # NAME+CODE regex pair のみ SUN ノードとして抽出する。
        # chitchat / 部分一致 span / "了解しました" 等の promotion → SUN 昇格 chain を断ち、
        # CD を厳密に fact 数のみに保つ (L3 で cd_max=50 を実現)。
        self.regex_only = regex_only
        self._model = None
        self._query_embs = None

    def setup(self):
        # regex_only モードでは SBERT model はロード不要 (sliding window 不使用)
        if self.regex_only:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_id)
            self._query_embs = self._model.encode(
                self.queries, convert_to_tensor=True, normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as e:
            import warnings
            warnings.warn(
                f"SBERTExtractor: SBERT ロード失敗 ({e})。"
                f"regex_only=True にフォールバック。CD 構築は NAME+CODE pair のみになります。",
                RuntimeWarning, stacklevel=2,
            )
            self.regex_only = True

    def extract(self, text: str) -> list[dict]:
        if not self.regex_only and self._model is None:
            self.setup()
        if not text or not text.strip():
            return []

        # ===== A-0: regex で fact pair 検出 =====
        regex_facts = []
        if self.enable_regex_hybrid:
            pairs = _find_fact_pairs(text, name_window=self.name_window)
            for p in pairs:
                regex_facts.append({
                    "text": p["text"],
                    "level": "sun",
                    "parent_hint": "",
                })

        # ===== regex_only モード: SBERT sliding window をスキップ =====
        if self.regex_only:
            return regex_facts

        # ===== SBERT span 検出 =====
        candidates = []
        for i in range(0, max(1, len(text) - self.window_size + 1), self.step):
            span = text[i:i + self.window_size].strip()
            if span:
                candidates.append(span)

        sbert_spans: list[str] = []
        if candidates:
            import torch
            cand_embs = self._model.encode(
                candidates, convert_to_tensor=True, normalize_embeddings=True,
                batch_size=64, show_progress_bar=False,
            )
            scores = cand_embs @ self._query_embs.T
            max_scores, _ = scores.max(dim=1)
            # §78.9: top-K cap で過剰抽出を抑制 (Gemma 3n 用)
            # 旧: threshold 以上の全 span を採用 → 大量抽出
            # 新: threshold 以上 AND top-K max_score のみ採用
            mask = max_scores >= self.threshold
            if mask.any():
                # threshold pass した中で max_scores 上位 K を選択
                indices = mask.nonzero(as_tuple=True)[0]
                selected_scores = max_scores[indices]
                K = min(self.max_sbert_spans_per_call, len(indices))
                top_k_idx_in_selected = selected_scores.topk(K).indices
                sel = indices[top_k_idx_in_selected].cpu().tolist()
                sbert_spans = [candidates[i] for i in sel]
                sbert_spans = _dedupe_substrings(sbert_spans)

        # ===== A-1: 統合 — regex fact があれば SBERT は補助情報のみ =====
        out: list[dict] = []
        seen_texts: set[str] = set()

        # regex facts 優先 (確実な fact)
        for f in regex_facts:
            if f["text"] not in seen_texts:
                seen_texts.add(f["text"])
                out.append(f)

        # SBERT spans を補助で追加 (regex fact text と重複する場合は skip)
        # §80.9 fix: text format が "{name}: {code}" に変わったため ": " で split。
        # _find_fact_pairs が返す dict の name/code は regex_facts 変換後に消えるので
        # text から再 split して name/code を復元する。
        for s in sbert_spans:
            if any(p_name in s and p_code in s
                   for p_name, p_code in [(f["text"].split(": ")[0],
                                           f["text"].split(": ")[-1])
                                          for f in regex_facts]):
                continue
            if s in seen_texts:
                continue
            seen_texts.add(s)
            out.append({
                "text": s[:160].strip(),
                "level": self.default_level,
                "parent_hint": "",
            })

        return out

    def __call__(self, text: str) -> list[dict]:
        return self.extract(text)


def _dedupe_substrings(spans: list[str]) -> list[str]:
    """部分文字列を含む span を 1 件にまとめる (長い順 priority)"""
    out: list[str] = []
    spans_sorted = sorted(spans, key=len, reverse=True)
    for s in spans_sorted:
        if not s:
            continue
        if any(s in existing or existing in s for existing in out):
            continue
        out.append(s)
    return out


def make_extractor_fn(
    model_id: str = DEFAULT_MODEL_ID,
    window_size: int = 30,
    step: int = 5,
    threshold: float = 0.2,
    enable_regex_hybrid: bool = True,
    max_sbert_spans_per_call: int = 3,
    regex_only: bool = False,
) -> Callable[[str], list[dict]]:
    """CMSSession に渡せる callable を返す (Phase A 版)。

    §80.9 (2026-05-16): regex_only=True で sliding window を完全無効化し、
    NAME+CODE pair のみで CD 構築する。 L3 (50 fact) で cd_max=50 に
    抑制でき、 GPT-OSS の coherence 崩壊 (cd>150) を回避する。
    """
    ext = SBERTExtractor(
        model_id=model_id, window_size=window_size,
        step=step, threshold=threshold,
        enable_regex_hybrid=enable_regex_hybrid,
        max_sbert_spans_per_call=max_sbert_spans_per_call,
        regex_only=regex_only,
    )
    ext.setup()
    return ext
