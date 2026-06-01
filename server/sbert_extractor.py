"""
SBERT cosine + regex hybrid extractor (an alternative to hamib_session._llm_extract_fn).

Behavior:
  - Relax the SBERT threshold so short conversational sentences are not missed.
  - Use regex to first extract NAME + CODE pairs (reliable fact detection).
  - Collapse overlapping spans into one via NMS.
  - When NAME and CODE co-occur, combine them into a single dict as a within-chunk pair.

Interface:
  callable: (text: str) -> list[dict]
    dict structure: {"text": str, "level": "sun"|"planet"|"satellite", "parent_hint": str}

Output policy:
  - When regex detects a fact pair (NAME + CODE):
    * one dict per fact, level=sun, text in "NAME -> CODE" form
  - When regex finds no pair, emit SBERT spans only:
    * after NMS and substring removal, level=satellite, parent_hint=""
"""
from __future__ import annotations
import os
import re
from typing import Callable

# POSIX/Windows compatible: a backslash literal is not expanduser-ed on Linux,
# so build the path with os.path.join to resolve HOME reliably.
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

# regex for extracting fact pairs
#   To handle mixed Japanese text, avoid \b and explicitly match non-alpha boundaries.
#   NAME: starts uppercase and must contain a lowercase letter (matches Alpha/Beta/Gamma, excludes CRANE)
#   CODE: 2+ uppercase letters + hyphen + digits (CRANE-1, ABC-100, etc., all caps)
NAME_PATTERN = re.compile(r'(?:^|[^A-Za-z])([A-Z][a-z]+[A-Za-z]*)(?![A-Za-z])')
CODE_PATTERN = re.compile(r'(?:^|[^A-Za-z0-9-])([A-Z]{2,}-\d+)(?![A-Za-z0-9])')
# Exclude words that are too common
_NAME_STOPLIST = {
    "User", "Assistant", "System", "Hi", "Hello", "Yes", "No", "JSON",
    "ID", "Code", "Name", "Project", "BERT", "SBERT", "LLM", "API",
    "Phase", "Turn", "Total", "RESP", "USER", "TEST", "DEBUG",
}


def _find_fact_pairs(text: str, name_window: int = 80) -> list[dict]:
    """
    Extract pairs where NAME and CODE co-occur within proximity as facts.

    name_window: maximum character distance between NAME and CODE
                 (80 chars assumes the same sentence/clause)
    """
    if not text:
        return []
    names = []
    for m in NAME_PATTERN.finditer(text):
        n = m.group(1)
        if n in _NAME_STOPLIST:
            continue
        # Keep all-caps tokens (e.g. ALPHA); dedupe handles them later
        names.append((m.start(), n))
    codes = [(m.start(), m.group(1)) for m in CODE_PATTERN.finditer(text)]
    if not names or not codes:
        return []

    facts: list[dict] = []
    used_codes = set()
    for code_pos, code in codes:
        if code in used_codes:
            continue
        # Find the nearest name (allowing before or just after the CODE)
        nearest = None
        nearest_dist = name_window + 1
        for name_pos, name in names:
            dist = abs(code_pos - name_pos)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = name
        if nearest and nearest_dist <= name_window:
            # Use the "{name}: {code}" text format. It keeps short-NAME cosine
            # similarity low enough to avoid GraphMerger mis-merges while staying
            # readable to the LLM.
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
        # Cap SBERT span extraction per turn at the top-3 max_score to keep the
        # accumulated CD size bounded. Without the cap, over-extraction (5-10 spans
        # per turn) piles up too many attention biases in the mass vector and breaks
        # the model output. regex_facts are reliable facts and are exempt from the cap.
        self.model_id = model_id
        self.window_size = window_size
        self.step = step
        self.threshold = threshold
        self.queries = queries or DEFAULT_QUERIES
        self.default_level = default_level
        self.enable_regex_hybrid = enable_regex_hybrid
        self.name_window = name_window
        self.max_sbert_spans_per_call = max_sbert_spans_per_call
        # regex_only=True fully disables the SBERT sliding window and extracts only
        # NAME+CODE regex pairs as SUN nodes. This breaks the promotion chain from
        # chitchat / partial-match spans / acknowledgements into SUN nodes and keeps
        # the CD strictly to the number of facts.
        self.regex_only = regex_only
        self._model = None
        self._query_embs = None

    def setup(self):
        # In regex_only mode the SBERT model need not be loaded (no sliding window)
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
                f"SBERTExtractor: failed to load SBERT ({e}). "
                f"Falling back to regex_only=True; CD construction will use NAME+CODE pairs only.",
                RuntimeWarning, stacklevel=2,
            )
            self.regex_only = True

    def extract(self, text: str) -> list[dict]:
        if not self.regex_only and self._model is None:
            self.setup()
        if not text or not text.strip():
            return []

        # ===== Detect fact pairs with regex =====
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
            cand_embs = self._model.encode(
                candidates, convert_to_tensor=True, normalize_embeddings=True,
                batch_size=64, show_progress_bar=False,
            )
            scores = cand_embs @ self._query_embs.T
            max_scores, _ = scores.max(dim=1)
            # Apply a top-K cap to suppress over-extraction: keep only spans that
            # are above threshold AND among the top-K by max_score.
            mask = max_scores >= self.threshold
            if mask.any():
                # Among the spans that pass the threshold, select the top-K by max_scores
                indices = mask.nonzero(as_tuple=True)[0]
                selected_scores = max_scores[indices]
                K = min(self.max_sbert_spans_per_call, len(indices))
                top_k_idx_in_selected = selected_scores.topk(K).indices
                sel = indices[top_k_idx_in_selected].cpu().tolist()
                sbert_spans = [candidates[i] for i in sel]
                sbert_spans = _dedupe_substrings(sbert_spans)

        # ===== Integrate: when regex facts exist, SBERT spans are supplementary only =====
        out: list[dict] = []
        seen_texts: set[str] = set()

        # Prioritize regex facts (reliable facts)
        for f in regex_facts:
            if f["text"] not in seen_texts:
                seen_texts.add(f["text"])
                out.append(f)

        # Add SBERT spans as supplements (skip those overlapping a regex fact text).
        # The text format is "{name}: {code}", so split on ": " to recover the
        # name/code that are dropped during the regex_facts conversion.
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
    """Collapse spans that contain a substring of another into one (longest first)."""
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
    """Return a callable that can be passed to HAMIBSession.

    With regex_only=True, the sliding window is fully disabled and the CD is built
    from NAME+CODE pairs only, keeping the CD size bounded and avoiding the model's
    coherence breakdown that occurs with large CDs.
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
