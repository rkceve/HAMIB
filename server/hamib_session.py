"""
HAMIBSession: self-contained session management class.

Integrates ManagementUnit, EvaluationUnit, and Gemma inference into a single
process. No FastAPI server is required; Gemma runs directly on the local
GPU/CPU.

Flow (one turn):
  1. Serialize the current CD into a <CONTEXT> block.
  2. Detect [PN{mass}] token positions and build a 1D mass vector.
  3. Run Gemma inference (with mass injection) to produce the assistant reply.
  4. ManagementUnit:
     a. Chunk the turn with TextChunker.
     b. Extract node candidates with Gemma (_llm_extract_fn).
     c. Build a provisional CD via NodeClassifier and GraphBuilder.
     d. Merge the provisional CD into the current CD with GraphMerger.
  5. EvaluationUnit (every eval_interval turns):
     a. Build an evaluation CD from the recent turns with EvalGraphBuilder.
     b. Score both CDs with Scorer (absence of contradictions + information density).
     c. If the evaluation CD scores higher, replace the current CD via Replacer.

Usage:
  from store.cd_store import CDStore
  from server.hamib_session import HAMIBSession

  store = CDStore(persist_path=Path("data/cd_store.json"))
  session = HAMIBSession(store)
  response = session.chat("こんにちは")
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
from server.cd_parser import find_pn_positions, extract_nodes_prompt
from utils.config import get
from utils.similarity import set_llm_similarity_fn


class HAMIBSession:
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
        disable_eval2: bool = True,  # When True, fully disables the EvaluationUnit (eval2).
        scorer=None,
        extractor_fn=None,  # Optional external extractor (e.g. SBERT). Expects Callable[[str], list[dict]].
        prompt_template: str | None = None,  # Overrides the prompt used in chat().
                                              # placeholders: {context_block} {user_text}.
                                              # When None, the default prompt template is used.
    ):
        """
        Args:
            store:                    CDStore instance.
            model_id:                 Inference model ID. When None, config.server.model_id is used.
            use_mass:                 When False, skips mass injection (for ablation experiments).
            _preloaded_gemma:         For experiments; pass an already-loaded inference model.
            extract_model_id:         Model ID for the ManagementUnit and EvaluationUnit
                                      (shares the inference model when omitted).
            _preloaded_extract_gemma: For experiments; pass an already-loaded extraction model.
            skip_mgmt_on_query:       When True, skips the ManagementUnit on query turns
                                      (prevents CD pollution).
            max_extract_tokens:       Skips the ManagementUnit when text length exceeds this value.
            use_llm_similarity:       When True, uses LLM-based node similarity judgment.
                                      Defaults to fast embedding-based (all-MiniLM-L6-v2) similarity.
            speculative_update:       When True, runs the ManagementUnit and EvaluationUnit
                                      asynchronously on a background thread (speculative update).
                                      Advancing CD updates while waiting for user input keeps the
                                      perceived response latency low.
            disable_eval2:            When True, fully disables the EvaluationUnit (consistency
                                      maintenance). Used for controlled with/without comparison.
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
        self._extractor_fn = extractor_fn  # External extractor (e.g. SBERT)
        # Allows swapping the prompt template (e.g. for English benchmarks).
        self._prompt_template = prompt_template

        # Background thread and locks for speculative update
        self._bg_thread: threading.Thread | None = None
        self._bg_lock = threading.Lock()
        self._cd_lock = threading.Lock()  # Makes CDStore operations thread-safe

        if use_llm_similarity:
            # Switch node similarity judgment to the LLM-based implementation
            set_llm_similarity_fn(self._llm_similarity_fn)
        else:
            set_llm_similarity_fn(None)

        # Metrics for experiments (updated after each chat() call)
        self.last_context_tokens: int = 0

        # ManagementUnit
        self._chunker = TextChunker()
        self._classifier = NodeClassifier()
        self._builder = GraphBuilder()
        self._merger = GraphMerger()

        # EvaluationUnit
        self._eval_builder = EvalGraphBuilder()
        # The scorer is swappable (defaults to the v1 Scorer when None)
        self._replacer = Replacer(store, scorer=scorer)

        # CD serializer
        self._serializer = CDSerializer()

        self._eval_interval: int = get("evaluation", "eval_interval_rounds", 5)
        self._turn: int = 0
        self._recent_turns: list[tuple[str, str]] = []

    # ── Gemma lazy loading ──────────────────────────────────────────────

    def load(self) -> None:
        """Explicitly load Gemma (call before the first chat to warm up)."""
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
        """Return the extraction model, falling back to the inference model when unset."""
        if self._extract_gemma is not None:
            return self._extract_gemma
        return self._gemma

    # ── Main entry point ─────────────────────────────────────────

    def chat(self, user_text: str) -> str:
        """
        Run one turn of HAMIB inference and return the assistant reply.

        Internal flow:
          wait for background work -> inference -> (sync or speculative)
          ManagementUnit and EvaluationUnit

        Instrumentation: self.last_timing dict records the elapsed time of each phase.
          keys: prompt_build_ms, gen_ms, mgmt_ms, eval2_ms, total_ms,
                skipped_query, skipped_long, eval2_triggered
        """
        import time as _time  # avoid name collision
        self.last_timing = {
            "prompt_build_ms": 0.0, "gen_ms": 0.0,
            "mgmt_ms": 0.0, "eval2_ms": 0.0, "total_ms": 0.0,
            "skipped_query": False, "skipped_long": False,
            "eval2_triggered": False,
        }
        _t_start = _time.perf_counter()

        self._ensure_gemma()
        self._turn += 1

        # When speculative update is enabled, wait for the previous turn's background work to finish
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
                # English translation of the Japanese prompt below (reference
                # only — the Japanese original is the live prompt; translating
                # it would shift the LLM's output. See README "A note on
                # language"):
                #   You are a conversation assistant. Based on the reference
                #   information (correlation diagram data) below, answer ONLY
                #   the user's question. Do not repeat the reference
                #   information verbatim. Answer concisely.
                #
                #   {context_block}
                #
                #   User: {user_text}
                #   Assistant:
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
        mass_vec = self._build_mass_vector(prompt_ids, device)

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

        # Step 5-6: ManagementUnit + EvaluationUnit
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
            # Speculative update runs in the background; it is not measured here and counts as 0 in perceived latency
            self._launch_background_update(user_text, response)
        else:
            # Synchronous execution; measure each phase
            _t_mgmt_start = _time.perf_counter()
            if not (skip_query or skip_long):
                self._update_cd(cd, user_text, response)
            _t_mgmt_end = _time.perf_counter()
            self.last_timing["mgmt_ms"] = (_t_mgmt_end - _t_mgmt_start) * 1000

            # EvaluationUnit (internally decides eval_interval and runs or skips)
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

    # ── Speculative update ───────────────────────────────

    def _launch_background_update(self, user_text: str, response: str) -> None:
        """
        Run the ManagementUnit and EvaluationUnit asynchronously on a background thread.
        Waits for the previous turn's background work to finish before launching.
        """
        self._wait_background()

        def _bg_task():
            try:
                with self._cd_lock:
                    cd = self._store.get_current()
                self._update_cd(cd, user_text, response)
                self._post_turn_evaluation(user_text, response)
            except Exception as e:
                print(f"[speculative-update] background task error: {e}")

        thread = threading.Thread(target=_bg_task, daemon=True)
        with self._bg_lock:
            self._bg_thread = thread
        thread.start()

    def _wait_background(self, timeout: float | None = None) -> None:
        """Wait for background work to finish if it is running."""
        with self._bg_lock:
            t = self._bg_thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)

    def wait_pending_updates(self, timeout: float | None = None) -> None:
        """
        Public API to wait for speculative updates to finish from outside.
        Used to guarantee the CD is up to date when aggregating experiment summaries.
        """
        self._wait_background(timeout=timeout)

    def _post_turn_evaluation(self, user_text: str, response: str) -> None:
        """Post-turn EvaluationUnit logic (called from both the sync and speculative paths).

        When disable_eval2=True, only manages recent_turns and does not run the EvaluationUnit.
        """
        with self._cd_lock:
            self._recent_turns.append((user_text, response))
            if len(self._recent_turns) > self._eval_interval:
                self._recent_turns.pop(0)
            should_run = (not self._disable_eval2) and (self._turn % self._eval_interval == 0)
        if should_run:
            self._run_evaluation()

    # ── Query-turn detection ──────────────────────────────────────────────

    # English glosses for the Japanese substring patterns below (reference
    # only — these are used as literal substring matches, and the Japanese
    # strings ARE the active heuristic). They detect question-style
    # utterances so the ManagementUnit is skipped on query turns:
    #   "を一語で答えてください" -> "answer ... in one word, please"
    #   "を答えてください"        -> "please answer ..."
    #   "を教えてください"        -> "please tell me ..."
    #   "を答えなさい"            -> "answer ..." (imperative)
    #   "は何ですか"              -> "what is ...?"
    #   "は何でしょうか"          -> "what is ...?" (polite)
    #   "を教えて"                -> "tell me ..." (casual)
    #   "を答えて"                -> "answer ..." (casual)
    #   "を一語で"                -> "... in one word"
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
        Determine whether the user text is a query (question) turn.

        Query turns introduce no new facts, so running the ManagementUnit would only
        pollute the CD; such turns are skipped. Heuristic: True when the text is short
        and has question-like characteristics.

        Note: the bare word "ください" is not included in the heuristic. Fact-introducing
        sentences such as "確実に記録してください" also contain "ください" and would be
        misdetected. Only question-specific compound phrases such as "を答えてください"
        and "を一語で答えてください" are used.
        """
        text = user_text.strip()
        if len(text) > 300:
            return False
        if text.endswith("？") or text.endswith("?"):
            return True
        return any(phrase in text for phrase in self._QUERY_PHRASES)

    # ── ManagementUnit: CD update ────────────────────────────────────────────────

    def _update_cd(
        self, cd: CorrelationDiagram, user_text: str, assistant_text: str
    ) -> None:
        """
        Build a provisional correlation diagram from this turn's conversation and
        merge it into the current CD.

        The provisional CD starts empty at the beginning of the turn and accumulates
        as chunks are processed. Afterwards GraphMerger integrates it into the current
        CD, summing the mass of similar nodes. The merge recomputes mass and coordinates
        (inside GraphMerger.merge).
        """
        provisional = CorrelationDiagram()
        chunks = self._chunker.chunk_turn(user_text, assistant_text, self._turn)
        for chunk in chunks:
            # Pass builder so classifier applies each item's
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

        # Merge the provisional CD into the current CD (locked because CDStore is not thread-safe)
        with self._cd_lock:
            self._merger.merge(cd, provisional)
            self._store.set_current(cd)

    # ── EvaluationUnit: CD evaluation and replacement ─────────────────────────────────────────

    def _run_evaluation(self) -> None:
        """
        Build an evaluation CD from the most recent eval_interval turns and replace
        the current CD if its score exceeds the current CD by at least the margin.
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
        # Exposed so benchmarks can observe the replacement count and score_diff
        self.last_eval_result = result
        print(
            f"[eval-unit] turn={self._turn}: "
            f"replaced={result['replaced']}, "
            f"score_diff={result.get('score_diff', 0):.4f}"
        )

    # ── LLM concept extraction (shared by ManagementUnit and EvaluationUnit) ────────────────────────

    def _llm_extract_fn(self, text: str) -> list[dict]:
        """
        Extract node candidates from text.

        When extractor_fn is set, uses the external extractor (e.g. SBERT);
        otherwise uses the Gemma + JSON prompt path.

        On success: returns the parsed JSON result
          e.g. [{"text": "「ALPHA」の対応値は「CRANE-1」", "level": "sun", "parent_hint": ""}]
        On failure: falls back to returning the whole text as a satellite node
        """
        # External extractor injection path
        if self._extractor_fn is not None:
            try:
                nodes = self._extractor_fn(text)
                if isinstance(nodes, list):
                    return [n for n in nodes if isinstance(n, dict) and "text" in n]
            except Exception:
                pass
            return [{"text": text[:80].strip(), "level": "satellite", "parent_hint": ""}]

        # Existing Gemma path
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
        # Fallback: treat the whole text as a satellite node
        return [{"text": text[:80].strip(), "level": "satellite", "parent_hint": ""}]

    # ── LLM similarity judgment ──────────────────────────────────────────

    def _llm_similarity_fn(self, text_a: str, text_b: str) -> float:
        """
        Pairwise node similarity judgment using the LLM.
        Returns a value from 0.0 (dissimilar) to 1.0 (identical).

        Implemented with zero-shot prompting. Because it is called frequently,
        caching the results is recommended in real deployments.
        """
        gemma = self._get_extract_gemma()
        # English translation of the Japanese prompt below (reference only —
        # the Japanese original is the live prompt; translating it would shift
        # similarity scores. See README "A note on language"):
        #   Rate the semantic similarity of the two concepts below as a number
        #   between 0.0 (unrelated) and 1.0 (identical concept).
        #   Output only the number (no explanation).
        #   Concept A: {text_a}
        #   Concept B: {text_b}
        #   Similarity:
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
            # Extract the number (the first decimal or integer that appears)
            import re as _re
            m = _re.search(r"[01](?:\.\d+)?|\.\d+", raw)
            if m:
                return max(0.0, min(1.0, float(m.group(0))))
        except Exception:
            pass
        return 0.0

    # ── Mass vector construction ────────────────────────────────────────────────

    def _build_mass_vector(
        self, prompt_ids: list[int], device: str
    ) -> torch.Tensor | None:
        """
        Return a 1D vector with mass added at the concept-text token positions
        immediately after each [PN{mass}] token. Returns None when there is no
        [PN] pattern (no mass injection).

        Uses the 1D vector form (decode-only, guarded by seq_q==1) rather than a
        2D M matrix.
        """
        pn_positions = find_pn_positions(prompt_ids, self._gemma.tokenizer)
        if not pn_positions:
            return None
        seq_len = len(prompt_ids)
        vec = torch.zeros(seq_len, dtype=torch.float32, device=device)
        for pos, mass in pn_positions:
            if 0 <= pos < seq_len:
                vec[pos] += mass
        return vec

    # ── State accessors ──────────────────────────────────────────────────

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def store(self) -> CDStore:
        return self._store
