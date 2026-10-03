"""
SBERT cosine + regex hybrid extractor (replacement for cms_session._llm_extract_fn).

Phase A changes, made after recall failed (0 spans on T3) because the SBERT
threshold was too strict for short conversational sentences:
  A-0: threshold relaxed (0.4 -> 0.2)
  A-0: regex extracts NAME + CODE pairs first (reliable fact detection)
  A-1: overlapping spans are collapsed into one
  A-3: a NAME + CODE co-occurrence becomes a single dict

Interface:
  callable: (text: str) -> list[dict]
    dict: {"text": str, "level": "sun"|"planet"|"satellite", "parent_hint": str}

Output policy:
  - When regex finds fact pairs (NAME + CODE):
    * one dict per fact, level=sun, text="NAME: CODE"
  - Otherwise SBERT spans only:
    * after deduplication of overlapping spans, level=satellite, parent_hint=""
"""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Callable

# Works on POSIX and Windows: a backslash path literal is not expanded on Linux,
# so the path is built with os.path.join to resolve HOME reliably.
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
    "プロジェクト名とコード",  # added for conversational text
    "project code identifier",
]

# A-0: regexes for fact-pair extraction
#   No \b, because the text mixes in Japanese; non-letter boundaries are matched explicitly.
#   NAME: capitalized word containing lowercase (Alpha/Beta/Gamma; excludes CRANE)
#   CODE: 2+ capitals + hyphen + digits (CRANE-1, ABC-100; all caps)
NAME_PATTERN = re.compile(r'(?:^|[^A-Za-z])([A-Z][a-z]+[A-Za-z]*)(?![A-Za-z])')
CODE_PATTERN = re.compile(r'(?:^|[^A-Za-z0-9-])([A-Z]{2,}-\d+)(?![A-Za-z0-9])')
# Words too common to be names
_NAME_STOPLIST = {
    "User", "Assistant", "System", "Hi", "Hello", "Yes", "No", "JSON",
    "ID", "Code", "Name", "Project", "BERT", "SBERT", "LLM", "API",
    "Phase", "Turn", "Total", "RESP", "USER", "TEST", "DEBUG",
}


def _find_fact_pairs(text: str, name_window: int = 80) -> list[dict]:
    """
    Extract NAME/CODE pairs that occur close together as facts.

    name_window: max character distance between NAME and CODE (80 chars ~ one sentence/clause)
    """
    if not text:
        return []
    names = []
    for m in NAME_PATTERN.finditer(text):
        n = m.group(1)
        if n in _NAME_STOPLIST:
            continue
        # Excluding all-caps identifiers (e.g. ALPHA) was considered, but Phase A
        # wants them, so they are kept and left to the later dedupe.
        names.append((m.start(), n))
    codes = [(m.start(), m.group(1)) for m in CODE_PATTERN.finditer(text)]
    if not names or not codes:
        return []

    facts: list[dict] = []
    used_codes = set()
    for code_pos, code in codes:
        if code in used_codes:
            continue
        # Find the nearest NAME (before or right after the CODE)
        nearest = None
        nearest_dist = name_window + 1
        for name_pos, name in names:
            dist = abs(code_pos - name_pos)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = name
        if nearest and nearest_dist <= name_window:
            # The text is "{name}: {code}". The older format (name, a Japanese
            # "corresponding code" phrase, code) gave short names such as
            # Mu/Nu/Tau/Chi a cosine similarity of 0.92+ under all-MiniLM-L6-v2,
            # so GraphMerger merged distinct facts (3 misses on L2, same on L3).
            # "{name}: {code}" peaks at 0.78 and stays readable for the LLM.
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
        # Span cap for Gemma 3n: SBERT over-extracted (5-10 spans per turn), the CD
        # grew to 99 nodes (3x the 34 of cms_g3n_g3n), and 99 stacked attention
        # biases broke Gemma 3n's per_layer_input output completely
        # (cms_g3n_sbert recall 0/10). SBERT spans are therefore cut to the top 3
        # by max_score per call, keeping cd_max around 30-40. regex_facts are
        # reliable and not capped (1-2 per turn).
        self.model_id = model_id
        self.window_size = window_size
        self.step = step
        self.threshold = threshold
        self.queries = queries or DEFAULT_QUERIES
        self.default_level = default_level
        self.enable_regex_hybrid = enable_regex_hybrid
        self.name_window = name_window
        self.max_sbert_spans_per_call = max_sbert_spans_per_call
        # regex_only=True disables the SBERT sliding window entirely and extracts
        # only NAME+CODE regex pairs as SUN nodes. This stops chitchat, partial
        # spans and acknowledgements ("understood") from being promoted to SUN,
        # so the CD holds exactly one node per fact (cd_max=50 on L3).
        self.regex_only = regex_only
        self._model = None
        self._query_embs = None

    def setup(self):
        # regex_only mode needs no SBERT model (no sliding window)
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

        # ===== A-0: regex fact-pair detection =====
        regex_facts = []
        if self.enable_regex_hybrid:
            pairs = _find_fact_pairs(text, name_window=self.name_window)
            for p in pairs:
                regex_facts.append({
                    "text": p["text"],
                    "level": "sun",
                    "parent_hint": "",
                })

        # ===== regex_only mode: skip the SBERT sliding window =====
        if self.regex_only:
            return regex_facts

        # ===== SBERT span detection =====
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
            # Top-K cap against over-extraction (for Gemma 3n): keep spans that pass
            # the threshold AND are in the top K by max_score (previously every span
            # above the threshold was kept).
            mask = max_scores >= self.threshold
            if mask.any():
                # Among spans above the threshold, take the top K by max_scores
                indices = mask.nonzero(as_tuple=True)[0]
                selected_scores = max_scores[indices]
                K = min(self.max_sbert_spans_per_call, len(indices))
                top_k_idx_in_selected = selected_scores.topk(K).indices
                sel = indices[top_k_idx_in_selected].cpu().tolist()
                sbert_spans = [candidates[i] for i in sel]
                sbert_spans = _dedupe_substrings(sbert_spans)

        # ===== A-1: merge; when regex facts exist, SBERT spans are only supplementary =====
        out: list[dict] = []
        seen_texts: set[str] = set()

        # Regex facts first (reliable facts)
        for f in regex_facts:
            if f["text"] not in seen_texts:
                seen_texts.add(f["text"])
                out.append(f)

        # Add SBERT spans as supplements, skipping any that repeat a regex fact.
        # The name/code keys are dropped when regex_facts is built, so they are
        # recovered by splitting the "{name}: {code}" text on ": ".
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
    """Collapse spans that contain one another into one (longest first)."""
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
    """Return a callable that can be passed to CMSSession (Phase A version).

    regex_only=True disables the sliding window and builds the CD from NAME+CODE
    pairs only. That keeps cd_max at 50 on L3 (50 facts) and avoids the GPT-OSS
    coherence collapse seen above cd=150.
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
