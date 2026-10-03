"""
CMSSession: self-contained session manager for the patent's first embodiment.

Runs management unit 1 (ManagementUnit), evaluation unit 2 (EvaluationUnit) and
Gemma inference in one process. No FastAPI server is needed; Gemma runs directly
on the local GPU/CPU.

One turn:
  1. Serialize the current CD into a <CONTEXT> block.
  2. Find the [PN{mass}] token positions and build a 1D mass vector.
  3. Run Gemma with mass injection to get the assistant response.
  4. Management unit 1:
     a. TextChunker splits the turn into chunks.
     b. Gemma (_llm_extract_fn) extracts node candidates.
     c. NodeClassifier -> GraphBuilder build a provisional CD.
     d. GraphMerger merges the provisional CD into the current CD (patent §0036-§0041).
  5. Evaluation unit 2 (every eval_interval turns):
     a. EvalGraphBuilder builds an evaluation CD from the recent turns.
     b. Scorer scores both CDs (absence of contradictions + information density).
     c. If the evaluation CD wins, Replacer replaces the current CD (patent §0063-§0072).

Usage:
  from store.cd_store import CDStore
  from server.cms_session import CMSSession

  store = CDStore(persist_path=Path("data/cd_store.json"))
  session = CMSSession(store)
  response = session.chat("Hello")
"""
from __future__ import annotations
import json
import sys
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from store.cd_store import CDStore
from models.correlation_diagram import CorrelationDiagram
from management.text_chunker import TextChunker
from management.node_classifier import NodeClassifier
from management.graph_builder import GraphBuilder
from management.graph_merger import GraphMerger
from evaluation.eval_graph_builder import EvalGraphBuilder
from evaluation.replacer import Replacer
from communication.cd_serializer import CDSerializer
from server.mass_weighted_gemma import MassWeightedGemma
from server.cd_parser import (
    find_pn_positions,
    find_pn_positions_with_level,
    inherit_satellite_mass,
    find_pn_spans,
    filter_positions_by_level,
    levels_from_context_block,
    positions_for_levels,
    extract_nodes_prompt,
)
from server.mass_vector import positions_to_mass_vector
from utils.config import get
from utils.similarity import set_llm_similarity_fn


class CMSSession:
    def __init__(
        self,
        store: CDStore,
        model_id: str | None = None,
        use_mass: bool = True,
        _preloaded_gemma: MassWeightedGemma | None = None,
        extract_model_id: str | None = None,
        _preloaded_extract_gemma: MassWeightedGemma | None = None,
        skip_mgmt_on_query: bool = True,
        max_extract_tokens: int = 1500,
        use_llm_similarity: bool = False,
        speculative_update: bool = False,
        disable_eval2: bool = True,  # True by default: the Scorer deviates from the spec, so evaluation unit 2 stays off until it is fixed.
        scorer=None,
        extractor_fn=None,  # Optional external extractor (e.g. SBERT), Callable[[str], list[dict]].
        prompt_template: str | None = None,  # Overrides the prompt used by chat().
                                              # Placeholders: {context_block} {user_text}.
                                              # None keeps the original Japanese template.
    ):
        """
        Args:
            store:                    CDStore instance.
            model_id:                 Inference model id. None uses config.server.model_id.
            use_mass:                 False skips mass injection (ablation runs).
            _preloaded_gemma:         For experiments: an already loaded inference model.
            extract_model_id:         Model id for management unit 1 / evaluation unit 2
                                      (defaults to sharing the inference model).
            _preloaded_extract_gemma: For experiments: an already loaded extraction model.
            skip_mgmt_on_query:       If True, skip management unit 1 on query turns
                                      (keeps questions from polluting the CD).
            max_extract_tokens:       Skip management unit 1 when the turn is longer than this.
            use_llm_similarity:       If True, use LLM-based node similarity (patent §0042).
                                      The default is the faster embedding similarity
                                      (all-MiniLM-L6-v2).
            speculative_update:       If True, run management unit 1 / evaluation unit 2 in a
                                      background thread (speculative update, patent
                                      §0076-§0077), so the CD is updated while waiting for
                                      the next user input and responses stay fast.
            disable_eval2:            If True, disable evaluation unit 2 (consistency
                                      maintenance) entirely. Used for the with/without
                                      comparison of evaluation unit 2.
        """
        self._store = store
        self._model_id = model_id
        self._use_mass = use_mass
        self._gemma: MassWeightedGemma | None = _preloaded_gemma
        self._extract_model_id = extract_model_id
        self._extract_gemma: MassWeightedGemma | None = _preloaded_extract_gemma
        self._skip_mgmt_on_query = skip_mgmt_on_query
        self._max_extract_tokens = max_extract_tokens
        self._use_llm_similarity = use_llm_similarity
        self._speculative_update = speculative_update
        self._disable_eval2 = disable_eval2
        self._extractor_fn = extractor_fn  # external extractor (e.g. SBERT)
        # Replaceable prompt template for the English benchmarks (LongMemEval / LoCoMo):
        # the Japanese template's "do not repeat the reference information" instruction
        # broke the substring-match scoring.
        self._prompt_template = prompt_template

        # Background thread and locks for speculative updates
        self._bg_thread: threading.Thread | None = None
        self._bg_lock = threading.Lock()
        self._cd_lock = threading.Lock()  # makes CDStore access thread-safe

        if use_llm_similarity:
            # Patent §0042: use the LLM for node similarity
            set_llm_similarity_fn(self._llm_similarity_fn)
        else:
            set_llm_similarity_fn(None)

        # Experiment metric, updated after each chat() call
        self.last_context_tokens: int = 0

        # Management unit 1
        self._chunker = TextChunker()
        self._classifier = NodeClassifier()
        self._builder = GraphBuilder()
        self._merger = GraphMerger()

        # Evaluation unit 2
        self._eval_builder = EvalGraphBuilder()
        # The scorer is pluggable (None means the default v1 Scorer)
        self._replacer = Replacer(store, scorer=scorer)

        # CD serializer
        self._serializer = CDSerializer()

        self._eval_interval: int = get("evaluation", "eval_interval_rounds", 5)
        self._turn: int = 0
        self._recent_turns: list[tuple[str, str]] = []

    # ── Lazy Gemma loading ────────────────────────────────────────────

    def load(self) -> None:
        """Load Gemma explicitly (calling this before the first chat() warms it up)."""
        if self._gemma is None:
            kwargs = {"model_id": self._model_id} if self._model_id else {}
            self._gemma = MassWeightedGemma(**kwargs)
            self._gemma.load()
        if self._extract_model_id is not None and self._extract_gemma is None:
            self._extract_gemma = MassWeightedGemma(model_id=self._extract_model_id)
            self._extract_gemma.load()

    def _ensure_gemma(self) -> None:
        if self._gemma is None:
            self.load()
        elif self._extract_model_id is not None and self._extract_gemma is None:
            self._extract_gemma = MassWeightedGemma(model_id=self._extract_model_id)
            self._extract_gemma.load()

    def _get_extract_gemma(self) -> MassWeightedGemma:
        """Return the extraction model, or the inference model when none is set."""
        if self._extract_gemma is not None:
            return self._extract_gemma
        return self._gemma

    # ── Main entry point ──────────────────────────────────────────────

    def chat(self, user_text: str) -> str:
        """
        Run one CMS turn and return the assistant response.

        Order: wait for background work -> inference -> management unit 1 and
        evaluation unit 2 (synchronous or speculative).

        Per-phase timings are recorded in self.last_timing:
          keys: prompt_build_ms, gen_ms, mgmt_ms, eval2_ms, total_ms,
                skipped_query, skipped_long, eval2_triggered
        """
        import time as _time  # avoid a name clash
        self.last_timing = {
            "prompt_build_ms": 0.0, "gen_ms": 0.0,
            "mgmt_ms": 0.0, "eval2_ms": 0.0, "total_ms": 0.0,
            "skipped_query": False, "skipped_long": False,
            "eval2_triggered": False,
        }
        _t_start = _time.perf_counter()

        self._ensure_gemma()
        self._turn += 1

        # With speculative updates, wait for the previous turn's background work
        if self._speculative_update:
            self._wait_background()

        # Step 1-3: CD → context block, prompt, mass vector
        with self._cd_lock:
            cd = self._store.get_current()
            context_block = self._serializer.to_context_block(cd) if cd.suns else ""

        if context_block:
            if self._prompt_template is not None:
                prompt = self._prompt_template.format(
                    context_block=context_block, user_text=user_text,
                )
            else:
                prompt = (
                    "あなたは会話アシスタントです。以下の参考情報（相関図データ）を踏まえて、"
                    "ユーザーの質問にのみ回答してください。"
                    "参考情報の内容をそのまま繰り返してはいけません。簡潔に答えてください。\n\n"
                    f"{context_block}\n\n"
                    f"User: {user_text}\nAssistant:"
                )
        else:
            prompt = f"User: {user_text}\nAssistant:"

        tokenizer = self._gemma.tokenizer
        prompt_ids = tokenizer.encode(prompt)
        self.last_context_tokens = len(prompt_ids)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mass_vec = self._build_mass_vector(prompt_ids, device, context_block)

        _t_after_prompt = _time.perf_counter()
        self.last_timing["prompt_build_ms"] = (_t_after_prompt - _t_start) * 1000

        # Step 4: Gemma inference
        if self._use_mass and mass_vec is not None:
            self._gemma.set_mass_vector(mass_vec)
        response = self._gemma.generate(prompt)
        self._gemma.clear_mass_vector()
        self._gemma.clear_m_matrix()

        _t_after_gen = _time.perf_counter()
        self.last_timing["gen_ms"] = (_t_after_gen - _t_after_prompt) * 1000

        # Step 5-6: management unit 1 + evaluation unit 2
        skip_query = self._skip_mgmt_on_query and self._is_query_turn(user_text)
        skip_long = False
        if not skip_query:
            full_turn_text = f"User: {user_text}\nAssistant: {response}"
            n_tok = len(self._gemma.tokenizer.encode(full_turn_text, add_special_tokens=False))
            if n_tok > self._max_extract_tokens:
                skip_long = True
        self.last_timing["skipped_query"] = skip_query
        self.last_timing["skipped_long"] = skip_long

        if self._speculative_update and not (skip_query or skip_long):
            # Speculative update in the background: not timed (zero from the user's view)
            self._launch_background_update(user_text, response)
        else:
            # Synchronous: time each phase
            _t_mgmt_start = _time.perf_counter()
            if not (skip_query or skip_long):
                self._update_cd(cd, user_text, response)
            _t_mgmt_end = _time.perf_counter()
            self.last_timing["mgmt_ms"] = (_t_mgmt_end - _t_mgmt_start) * 1000

            # Evaluation unit 2 (checks eval_interval internally, then runs or skips)
            _t_eval_start = _time.perf_counter()
            should_run_eval2 = (
                (not self._disable_eval2)
                and (self._turn % self._eval_interval == 0)
            )
            self._post_turn_evaluation(user_text, response)
            _t_eval_end = _time.perf_counter()
            self.last_timing["eval2_ms"] = (_t_eval_end - _t_eval_start) * 1000
            self.last_timing["eval2_triggered"] = should_run_eval2

        self.last_timing["total_ms"] = (_time.perf_counter() - _t_start) * 1000
        return response

    # ── Speculative update (patent §0076-§0077) ───────────────────────

    def _launch_background_update(self, user_text: str, response: str) -> None:
        """
        Run management unit 1 and evaluation unit 2 in a background thread.
        Waits for the previous turn's background work first.
        """
        self._wait_background()

        def _bg_task():
            try:
                with self._cd_lock:
                    cd = self._store.get_current()
                self._update_cd(cd, user_text, response)
                self._post_turn_evaluation(user_text, response)
            except Exception as e:
                print(f"[投機的更新] バックグラウンド処理でエラー: {e}")

        thread = threading.Thread(target=_bg_task, daemon=True)
        with self._bg_lock:
            self._bg_thread = thread
        thread.start()

    def _wait_background(self, timeout: float | None = None) -> None:
        """Wait for the background work, if any, to finish."""
        with self._bg_lock:
            t = self._bg_thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)

    def wait_pending_updates(self, timeout: float | None = None) -> None:
        """
        Public API to wait for pending speculative updates, so the CD is up to
        date when an experiment collects its summary.
        """
        self._wait_background(timeout=timeout)

    def _post_turn_evaluation(self, user_text: str, response: str) -> None:
        """Post-turn evaluation unit 2 logic (called from both the sync and speculative paths).

        With disable_eval2=True only recent_turns is maintained; evaluation unit 2 is not called.
        """
        with self._cd_lock:
            self._recent_turns.append((user_text, response))
            if len(self._recent_turns) > self._eval_interval:
                self._recent_turns.pop(0)
            should_run = (not self._disable_eval2) and (self._turn % self._eval_interval == 0)
        if should_run:
            self._run_evaluation()

    # ── Query-turn detection ──────────────────────────────────────────

    _QUERY_PHRASES = (
        "を一語で答えてください",
        "を答えてください",
        "を教えてください",
        "を答えなさい",
        "は何ですか",
        "は何でしょうか",
        "を教えて",
        "を答えて",
        "を一語で",
    )

    def _is_query_turn(self, user_text: str) -> bool:
        """
        Return True if the user text is a query (question) turn.

        A query turn introduces no new facts, so running management unit 1 on it
        would only pollute the CD. Heuristic: short text that looks like a question.

        The bare word for "please" is deliberately not a cue: fact-stating
        sentences ("please record this reliably") contain it too. Only the
        question-specific compound phrases in _QUERY_PHRASES are used.
        """
        text = user_text.strip()
        if len(text) > 300:
            return False
        if text.endswith("？") or text.endswith("?"):
            return True
        return any(phrase in text for phrase in self._QUERY_PHRASES)

    # ── Management unit 1: CD update ──────────────────────────────────

    def _update_cd(
        self, cd: CorrelationDiagram, user_text: str, assistant_text: str
    ) -> None:
        """
        Build a provisional CD from this turn and merge it into the current CD
        (patent §0036-§0041).

        The provisional CD starts empty and accumulates as chunks are processed.
        GraphMerger then merges it into the current CD, summing the mass of similar
        nodes; GraphMerger.merge also recomputes mass (§0062) and coordinates (§0030).
        """
        provisional = CorrelationDiagram()
        chunks = self._chunker.chunk_turn(user_text, assistant_text, self._turn)
        for chunk in chunks:
            # §94.2 fix: pass builder so classifier applies each item's
            # proposal incrementally. Without this, a satellite item whose
            # parent_hint refers to an entity declared earlier in the SAME
            # chunk cannot find that entity in provisional (the entity's
            # NEW_SUN proposal hasn't been applied yet) and falls through
            # the _build_proposals fallback all the way to NEW_SUN promotion.
            # Net effect: every fact-bearing sentence becomes a top-level sun
            # and speaker→fact attribution is lost in the CD.
            self._classifier.classify(
                chunk, provisional, self._llm_extract_fn,
                builder=self._builder,
            )

        # Merge the provisional CD into the current CD (locked: CDStore is not thread-safe)
        with self._cd_lock:
            self._merger.merge(cd, provisional)
            self._store.set_current(cd)

    # ── Evaluation unit 2: CD scoring and replacement ─────────────────

    def _run_evaluation(self) -> None:
        """
        Build an evaluation CD from the last eval_interval turns and replace the
        current CD if it scores at least `margin` higher (patent §0063-§0072).
        """
        with self._cd_lock:
            base_cd = self._store.get_current()
            start_turn = self._turn - len(self._recent_turns) + 1
            recent_turns_copy = list(self._recent_turns)
        eval_cd = self._eval_builder.build(
            recent_turns_copy, base_cd, self._llm_extract_fn, start_turn
        )
        with self._cd_lock:
            self._store.set_eval(eval_cd)
            result = self._replacer.evaluate_and_replace()
        # Exposed so benchmarks can observe replacement counts and score_diff
        self.last_eval_result = result
        print(
            f"[評価部2] turn={self._turn}: "
            f"replaced={result['replaced']}, "
            f"score_diff={result.get('score_diff', 0):.4f}"
        )

    # ── LLM concept extraction (management unit 1 / evaluation unit 2) ───

    def _llm_extract_fn(self, text: str) -> list[dict]:
        """
        Extract node candidates from text.

        Uses the external extractor (e.g. SBERT) when extractor_fn is set,
        otherwise the original Gemma + JSON prompt path.

        On success returns the parsed JSON, e.g.
          [{"text": "The value for ALPHA is CRANE-1", "level": "sun", "parent_hint": ""}]
        On failure falls back to the whole text as one satellite node.
        """
        # External extractor path
        if self._extractor_fn is not None:
            try:
                nodes = self._extractor_fn(text)
                if isinstance(nodes, list):
                    return [n for n in nodes if isinstance(n, dict) and "text" in n]
            except Exception:
                pass
            return [{"text": text[:80].strip(), "level": "satellite", "parent_hint": ""}]

        # Original Gemma path
        prompt = extract_nodes_prompt(text)
        gemma = self._get_extract_gemma()
        gemma.clear_mass_vector()
        gemma.clear_m_matrix()
        try:
            raw = gemma.generate(prompt)
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start != -1 and end > start:
                nodes = json.loads(raw[start:end])
                if isinstance(nodes, list):
                    return [n for n in nodes if isinstance(n, dict) and "text" in n]
        except Exception:
            pass
        # Fallback: treat the whole text as one satellite node
        return [{"text": text[:80].strip(), "level": "satellite", "parent_hint": ""}]

    # ── LLM similarity (§0042) ────────────────────────────────────────

    def _llm_similarity_fn(self, text_a: str, text_b: str) -> float:
        """
        Patent §0042: one-to-one node similarity scored by the LLM.
        Returns a value from 0.0 (unrelated) to 1.0 (identical).

        Uses a zero-shot prompt. It is called often, so cache the results in a
        real deployment.
        """
        gemma = self._get_extract_gemma()
        prompt = (
            "次の2つの概念の意味的類似度を 0.0（無関係）〜 1.0（同一概念）の数値で評価してください。\n"
            "数値のみを出力してください（説明文は不要）。\n"
            f"概念A: {text_a}\n"
            f"概念B: {text_b}\n"
            "類似度:"
        )
        gemma.clear_mass_vector()
        gemma.clear_m_matrix()
        try:
            raw = gemma.generate(prompt).strip()
            # Take the first decimal or integer in the output
            import re as _re
            m = _re.search(r"[01](?:\.\d+)?|\.\d+", raw)
            if m:
                return max(0.0, min(1.0, float(m.group(0))))
        except Exception:
            pass
        return 0.0

    # ── Mass vector construction ──────────────────────────────────────

    def _build_mass_vector(
        self, prompt_ids: list[int], device: str, context_block: str = ""
    ) -> torch.Tensor | None:
        """
        Return a 1D vector with mass placed at the concept-text tokens that follow
        each [PN{mass}] marker, or None (no injection) when there is no [PN] marker.

        Experiment L: a 2D M matrix (applied in prefill and decode) collapsed to 0%,
        while the 1D vector (decode only, seq_q == 1 guard) kept 80%.

        The values are assembled by server/mass_vector.positions_to_mass_vector,
        the single place for the cap min(cap, mass*scale) and for resolving two
        markers at the same position with max instead of a sum.

        When inject_levels restricts the levels, each node's level is read from the
        line indentation of the context_block actually embedded in the prompt, not
        from token whitespace (SentencePiece drops leading spaces on decode). Only
        when no context_block is given does it fall back to
        find_pn_positions_with_level.
        """
        tokenizer = self._gemma.tokenizer
        # Patent ¶0079: when inject_levels is set, inject only at [PN] markers of those levels
        inject_levels = get("attention", "inject_levels", None)
        if inject_levels:
            if isinstance(inject_levels, str):  # guard for a single level written as a bare string
                inject_levels = [inject_levels]
            levels = set(inject_levels)
            if context_block:
                node_levels = levels_from_context_block(context_block)
                spans = find_pn_spans(prompt_ids, tokenizer)
                if get("attention", "satellite_mass_mode", "fixed") == "inherit":
                    spans = inherit_satellite_mass(spans, node_levels)
                pn_positions = positions_for_levels(spans, node_levels, levels)
            else:
                pn_positions = filter_positions_by_level(
                    find_pn_positions_with_level(prompt_ids, tokenizer),
                    levels,
                )
        elif context_block and get("attention", "satellite_mass_mode", "fixed") == "inherit":
            node_levels = levels_from_context_block(context_block)
            spans = inherit_satellite_mass(find_pn_spans(prompt_ids, tokenizer), node_levels)
            pn_positions = positions_for_levels(spans, node_levels, None)
        else:
            pn_positions = find_pn_positions(prompt_ids, tokenizer)

        return positions_to_mass_vector(
            pn_positions,
            len(prompt_ids),
            cap=float(get("attention", "mass_cap", 3.0)),
            scale=float(get("attention", "mass_scale", 1.0)),
            device=device,
        )

    # ── State accessors ───────────────────────────────────────────────

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def store(self) -> CDStore:
        return self._store
