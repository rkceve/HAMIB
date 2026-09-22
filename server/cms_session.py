"""
CMSSession: 特許第1実施形態の自己完結型セッション管理クラス。

管理部1（ManagementUnit）・評価部2（EvaluationUnit）・Gemma推論を1プロセスに統合。
FastAPIサーバー起動不要。GemmaをローカルGPU/CPUで直接使用する。

フロー（1ターン）:
  1. 現行CDをシリアライズ → <CONTEXT>ブロック生成
  2. [PN{mass}]トークン位置を検出 → 1Dマスベクトル構築
  3. Gemma推論（マス注入あり）→ アシスタント応答
  4. 管理部1:
     a. TextChunker でターンをチャンク化
     b. Gemma (_llm_extract_fn) でノード候補抽出
     c. NodeClassifier → GraphBuilder で仮CDを構築
     d. GraphMerger で仮CDを現行CDにマージ（特許§0036〜§0041）
  5. 評価部2（eval_interval毎）:
     a. EvalGraphBuilder で直近ターンから評価CDを構築
     b. Scorer で両CDを採点（矛盾の欠如 + 情報の濃縮度）
     c. 評価CDが優秀なら Replacer で現行CDを置換（特許§0063〜§0072）

使い方:
  from store.cd_store import CDStore
  from server.cms_session import CMSSession

  store = CDStore(persist_path=Path("data/cd_store.json"))
  session = CMSSession(store)
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
        disable_eval2: bool = True,  # §60 (2026-05-10): default 反転 — Scorer 設計乖離で eval2 が機能していないため (cf. §49.2/§52/§60)。Scorer 修正後に False に戻す。
        scorer=None,
        extractor_fn=None,  # §71 (2026-05-12): 外部 extractor 注入用 (SBERT 等)。Callable[[str], list[dict]] を期待。
        prompt_template: str | None = None,  # §82 (2026-05-17): chat() で使う prompt の上書き。
                                              # placeholders: {context_block} {user_text}。
                                              # None なら従来の日本語 template を使用。
    ):
        """
        Args:
            store:                    CDStore インスタンス
            model_id:                 推論用モデルID。None の場合は config.server.model_id を使用。
            use_mass:                 False にするとマス注入をスキップ（アブレーション実験用）。
            _preloaded_gemma:         実験用。ロード済み推論モデルを渡す。
            extract_model_id:         管理部1・評価部2 用モデルID（省略時は推論モデルと共用）。
            _preloaded_extract_gemma: 実験用。ロード済み抽出モデルを渡す。
            skip_mgmt_on_query:       True の場合、クエリターンで管理部1をスキップ（CD汚染防止）。
            max_extract_tokens:       テキスト長がこの値を超えると管理部1をスキップ。
            use_llm_similarity:       True の場合、特許§0042 準拠で LLM ベース類似度判定を使用。
                                      デフォルトは埋め込み（all-MiniLM-L6-v2）ベースで高速。
            speculative_update:       True の場合、特許§0076-§0077 準拠で管理部1・評価部2 を
                                      バックグラウンドスレッドで非同期実行する（投機的更新）。
                                      ユーザー入力待機中に CD 更新を進めることで体感応答速度を維持。
            disable_eval2:            True の場合、評価部2（整合性維持処理）を完全に無効化する。
                                      実験Z（評価部2 効果検証）で「あり/なし」の対照実験に使用する。
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
        self._extractor_fn = extractor_fn  # §71 external extractor (SBERT 等)
        # §82: 英語ベンチ (LongMemEval / LoCoMo) 用に prompt template を差し替え可能に。
        # 「参考情報をそのまま繰り返してはいけません」が substring 一致を壊していた問題対応。
        self._prompt_template = prompt_template

        # 投機的更新用のバックグラウンドスレッド・ロック
        self._bg_thread: threading.Thread | None = None
        self._bg_lock = threading.Lock()
        self._cd_lock = threading.Lock()  # CDStore 操作のスレッドセーフ化

        if use_llm_similarity:
            # 特許§0042: ノード類似度判定を LLM ベースに切り替える
            set_llm_similarity_fn(self._llm_similarity_fn)
        else:
            set_llm_similarity_fn(None)

        # 実験用メトリクス（chat() 呼び出し後に更新される）
        self.last_context_tokens: int = 0

        # 管理部1
        self._chunker = TextChunker()
        self._classifier = NodeClassifier()
        self._builder = GraphBuilder()
        self._merger = GraphMerger()

        # 評価部2
        self._eval_builder = EvalGraphBuilder()
        # §49: scorer を差し替え可能にした (None なら v1 Scorer がデフォルト)
        self._replacer = Replacer(store, scorer=scorer)

        # CD シリアライザ
        self._serializer = CDSerializer()

        self._eval_interval: int = get("evaluation", "eval_interval_rounds", 5)
        self._turn: int = 0
        self._recent_turns: list[tuple[str, str]] = []

    # ── Gemma 遅延ロード ──────────────────────────────────────────────

    def load(self) -> None:
        """明示的にGemmaをロードする（初回chat前に呼ぶとウォームアップできる）。"""
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
        """抽出専用モデルを返す。未設定の場合は推論モデルと共用。"""
        if self._extract_gemma is not None:
            return self._extract_gemma
        return self._gemma

    # ── メインエントリポイント ─────────────────────────────────────────

    def chat(self, user_text: str) -> str:
        """
        1ターン分のCMS推論を実行してアシスタント応答を返す。

        内部フロー:
          バックグラウンド処理待機 → 推論 → (同期 or 投機的) 管理部1・評価部2

        §61 P0-1 計装: self.last_timing dict で各 phase の所要時間を記録する。
          keys: prompt_build_ms, gen_ms, mgmt_ms, eval2_ms, total_ms,
                skipped_query, skipped_long, eval2_triggered
        """
        import time as _time  # 名前衝突回避
        self.last_timing = {
            "prompt_build_ms": 0.0, "gen_ms": 0.0,
            "mgmt_ms": 0.0, "eval2_ms": 0.0, "total_ms": 0.0,
            "skipped_query": False, "skipped_long": False,
            "eval2_triggered": False,
        }
        _t_start = _time.perf_counter()

        self._ensure_gemma()
        self._turn += 1

        # 投機的更新が有効な場合、前ターンのバックグラウンド処理が完了するまで待機
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

        # Step 4: Gemma推論
        if self._use_mass and mass_vec is not None:
            self._gemma.set_mass_vector(mass_vec)
        response = self._gemma.generate(prompt)
        self._gemma.clear_mass_vector()
        self._gemma.clear_m_matrix()

        _t_after_gen = _time.perf_counter()
        self.last_timing["gen_ms"] = (_t_after_gen - _t_after_prompt) * 1000

        # Step 5-6: 管理部1 + 評価部2
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
            # 投機的更新 (background) — timing は計測不能、UX 上は 0
            self._launch_background_update(user_text, response)
        else:
            # 同期実行 — 各 phase 計測
            _t_mgmt_start = _time.perf_counter()
            if not (skip_query or skip_long):
                self._update_cd(cd, user_text, response)
            _t_mgmt_end = _time.perf_counter()
            self.last_timing["mgmt_ms"] = (_t_mgmt_end - _t_mgmt_start) * 1000

            # 評価部2 (内部で eval_interval 判定 + 実行 or skip)
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

    # ── 投機的更新（特許§0076-§0077） ───────────────────────────────

    def _launch_background_update(self, user_text: str, response: str) -> None:
        """
        管理部1・評価部2 をバックグラウンドスレッドで非同期実行する。
        前ターンのバックグラウンド処理が完了していない場合は待機してから起動。
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
        """バックグラウンド処理が走っている場合、完了を待つ。"""
        with self._bg_lock:
            t = self._bg_thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)

    def wait_pending_updates(self, timeout: float | None = None) -> None:
        """
        外部から投機的更新の完了を待つ公開API。
        実験のサマリー集計時に CD が最新状態であることを保証するために使用する。
        """
        self._wait_background(timeout=timeout)

    def _post_turn_evaluation(self, user_text: str, response: str) -> None:
        """ターン終了後の評価部2 実行ロジック（同期/投機的のいずれからも呼ばれる）。

        disable_eval2=True の場合は recent_turns の管理だけ行い、評価部2 は呼ばない。
        """
        with self._cd_lock:
            self._recent_turns.append((user_text, response))
            if len(self._recent_turns) > self._eval_interval:
                self._recent_turns.pop(0)
            should_run = (not self._disable_eval2) and (self._turn % self._eval_interval == 0)
        if should_run:
            self._run_evaluation()

    # ── クエリターン判定 ──────────────────────────────────────────────

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
        ユーザーテキストがクエリ（質問）ターンかどうかを判定する。

        クエリターンでは新たな事実が導入されず、管理部1を実行してもCDを汚染するだけなので
        スキップする。ヒューリスティック: 短くて疑問文の特徴を持つ場合にTrue。

        注意: 「ください」単体はヒューリスティックに含めない。
        「確実に記録してください」のような事実導入文も「ください」を含むため誤検出される。
        「を答えてください」「を一語で答えてください」等の質問特有の複合句のみ使用する。
        """
        text = user_text.strip()
        if len(text) > 300:
            return False
        if text.endswith("？") or text.endswith("?"):
            return True
        return any(phrase in text for phrase in self._QUERY_PHRASES)

    # ── 管理部1: CD更新 ────────────────────────────────────────────────

    def _update_cd(
        self, cd: CorrelationDiagram, user_text: str, assistant_text: str
    ) -> None:
        """
        今ターンの会話から仮の相関図を構築し、現行CDにマージする（特許§0036〜§0041）。

        仮CDはターン開始時に空で初期化され、チャンクを処理するにつれて蓄積される。
        処理後に GraphMerger で現行CDと統合し、類似ノードは mass を合算する。
        マージ後は §0062 質量再計算 + §0030 座標再計算が走る (GraphMerger.merge 内)。
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

        # 仮CDを現行CDにマージ（CDStoreはスレッドセーフではないためロック）
        with self._cd_lock:
            self._merger.merge(cd, provisional)
            self._store.set_current(cd)

    # ── 評価部2: CD評価・置換 ─────────────────────────────────────────

    def _run_evaluation(self) -> None:
        """
        直近 eval_interval ターンから評価CDを構築し、
        スコアが現行CDを margin 以上上回れば置換する（特許§0063〜§0072）。
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
        # §49: ベンチマークで replaced 回数 / score_diff を観測できるように公開
        self.last_eval_result = result
        print(
            f"[評価部2] turn={self._turn}: "
            f"replaced={result['replaced']}, "
            f"score_diff={result.get('score_diff', 0):.4f}"
        )

    # ── LLM 概念抽出（管理部1 / 評価部2 共通）────────────────────────

    def _llm_extract_fn(self, text: str) -> list[dict]:
        """
        テキストからノード候補を抽出する。

        §71: extractor_fn が指定されていれば外部抽出器 (SBERT 等) を使用、
              指定なければ Gemma + JSON prompt の従来パスを使用。

        成功時: JSON パース結果を返す
          例: [{"text": "「ALPHA」の対応値は「CRANE-1」", "level": "sun", "parent_hint": ""}]
        失敗時: フォールバックとして全文を satellite ノードとして返す
        """
        # §71: 外部 extractor 注入経路
        if self._extractor_fn is not None:
            try:
                nodes = self._extractor_fn(text)
                if isinstance(nodes, list):
                    return [n for n in nodes if isinstance(n, dict) and "text" in n]
            except Exception:
                pass
            return [{"text": text[:80].strip(), "level": "satellite", "parent_hint": ""}]

        # 既存 Gemma パス
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
        # フォールバック: テキスト全体を satellite ノードとして扱う
        return [{"text": text[:80].strip(), "level": "satellite", "parent_hint": ""}]

    # ── LLM 類似度判定 (§0042) ──────────────────────────────────────────

    def _llm_similarity_fn(self, text_a: str, text_b: str) -> float:
        """
        特許§0042 準拠: LLM を用いた1対1のノード類似度判定。
        0.0 (非類似) 〜 1.0 (完全一致) の数値を返す。

        本実装はゼロショットプロンプティングで実装する。
        高頻度に呼ばれるため、本番運用では結果のキャッシュを推奨する。
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
            # 数値を抽出（最初に現れる小数または整数）
            import re as _re
            m = _re.search(r"[01](?:\.\d+)?|\.\d+", raw)
            if m:
                return max(0.0, min(1.0, float(m.group(0))))
        except Exception:
            pass
        return 0.0

    # ── マスベクトル構築 ────────────────────────────────────────────────

    def _build_mass_vector(
        self, prompt_ids: list[int], device: str, context_block: str = ""
    ) -> torch.Tensor | None:
        """
        [PN{mass}] トークン直後の概念テキストトークン位置に mass を置いた
        1D ベクトルを返す。[PN] パターンがなければ None（マス注入なし）。

        実験Lで確認: 2D M行列（prefill+decode適用）は0%に崩壊。
        1D ベクトル（decode専用, seq_q==1 ガード）は80%を維持。

        F3: 値の組み立ては server/mass_vector.positions_to_mass_vector に
        委譲する（D-3 の上限 min(cap, mass*scale) と、同一位置の衝突を
        加算ではなく max で解決する規則がそこに一本化されている）。

        F7: inject_levels でレベルを絞る場合、レベルはトークン列の空白では
        なく、プロンプトへ実際に埋め込んだ context_block の行頭インデントから
        求める（SentencePiece は decode で語頭空白を落とすため）。
        context_block が渡らなかった場合のみ、従来の
        find_pn_positions_with_level に退避する。
        """
        tokenizer = self._gemma.tokenizer
        # WO-6 (¶0079): inject_levels が指定された場合はそのレベルの [PN] だけに注入する
        inject_levels = get("attention", "inject_levels", None)
        if inject_levels:
            if isinstance(inject_levels, str):  # 単一レベルを文字列で書かれた場合の保険
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

    # ── 状態アクセサ ──────────────────────────────────────────────────

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def store(self) -> CDStore:
        return self._store
