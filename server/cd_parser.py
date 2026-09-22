"""
CDParser: クライアントから受け取った node_list と context_block を解析し、
サーバー内部で使用できる形式に変換する。

また、テキストから概念ノード候補を抽出する /extract_nodes エンドポイントの
ロジックも担当する（Gemma を使って概念抽出）。
"""
from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class ParsedNode:
    node_id: str
    text: str
    level: str
    mass: float
    token_repr: str


def parse_node_list(node_list: list[dict]) -> list[ParsedNode]:
    return [
        ParsedNode(
            node_id=n["node_id"],
            text=n["text"],
            level=n["level"],
            mass=float(n["mass"]),
            token_repr=n["token_repr"],
        )
        for n in node_list
    ]


# ── [PN{mass}] トークン検出 ────────────────────────────────────────────
_PN_PATTERN = re.compile(r"\[PN([\d.]+)\]")
# Spec-faithful level markers (CDSerializer(level_markers=True)):
#   [SN] sun / [PN{mass}] planet / [RN] satellite. Only planets carry a mass.
_MARKER_PATTERN = re.compile(r"\[(SN|RN|PN([\d.]+))\]")
_MARKER_STARTS = ("[PN", "[SN", "[RN")
_MARKER_LEVEL = {"S": "sun", "P": "planet", "R": "satellite"}


def _marker_start(text: str, level_markers: bool) -> int:
    """Index of the first marker start in text, or -1."""
    if not level_markers:
        return text.find("[PN")
    hits = [text.find(m) for m in _MARKER_STARTS]
    hits = [h for h in hits if h != -1]
    return min(hits) if hits else -1

# 概念テキスト外で保持するデコード済み文字数の上限。
# _PN_PATTERN が一致しうる最大長 (~12 文字) より十分に大きければよい。
# これを入れないと decoded_so_far が入力全体まで伸び、走査が O(n^2) になる。
_SCAN_TAIL_CHARS = 32

# CDSerializer は "  " * indent + "[PN{mass}] text" 形式で出力する
# (communication/cd_serializer.py の _node_line)。indent 0/1/2 が
# sun/planet/satellite に対応する。
LEVEL_BY_INDENT = {0: "sun", 1: "planet", 2: "satellite"}


def _level_from_leading_spaces(n_spaces: int) -> str:
    """行頭の空白数からレベル名を求める。0-1→sun, 2-3→planet, >=4→satellite。"""
    return LEVEL_BY_INDENT[min(n_spaces // 2, 2)]


def _scan_pn_spans(
    token_ids: list[int], tokenizer, *, level_markers: bool = False
) -> list[tuple[float, str, list[int]]]:
    """[PN{mass}] マーカーごとに (mass, level, 概念トークン位置リスト) を返す走査本体。

    find_pn_positions / find_pn_positions_with_level / find_pn_spans は
    すべてこの 1 つの走査から導出される。走査を共有しているので

        [(p, m) for p, m, _ in find_pn_positions_with_level(ids, tok)]
        == find_pn_positions(ids, tok)

    は構造的に必ず成り立つ。

    level は「その [PN が現れた行の行頭空白数」から求める。SentencePiece 系
    トークナイザでは decode([id]) が先頭の空白を落とすため信頼できない
    (F7)。レベルで絞りたい場合は levels_from_context_block +
    positions_for_levels を使うこと。
    """
    # F8 (perf, deliberately NOT changed): this decodes ONE token at a time.
    # Batching the decode would be faster, but the scan is a character-level
    # state machine over the decoded stream -- a marker can straddle two tokens
    # and BPE can fuse "]" with the following text -- so a batched decode would
    # have to reproduce the same per-token boundaries anyway. Correctness first:
    # the scan runs once per question, not per generated token.
    spans: list[tuple[float, str, list[int]]] = []
    decoded_so_far = ""
    line_text = ""  # 直近の '\n' 以降にデコードされたテキスト
    in_concept_mass: float | None = None
    in_concept_level: str = LEVEL_BY_INDENT[0]
    current: list[int] = []

    def _close() -> None:
        nonlocal in_concept_mass, current
        if in_concept_mass is not None:
            spans.append((in_concept_mass, in_concept_level, current))
        in_concept_mass = None
        current = []

    for i, tid in enumerate(token_ids):
        token_decoded = tokenizer.decode([tid], skip_special_tokens=False)

        # 行バッファの更新 (レベル判定用。既存の走査状態には影響しない)
        line_text += token_decoded
        if "\n" in line_text:
            line_text = line_text[line_text.rindex("\n") + 1:]

        if in_concept_mass is not None:
            # 概念テキスト中: 改行 or 次のマーカーを見つけたら終端
            nxt = _marker_start(token_decoded, level_markers)
            if "\n" in token_decoded or nxt != -1:
                if "\n" in token_decoded:
                    # 2026-09-18: the closing token may itself carry concept
                    # text BEFORE its newline (e.g. " beta\n", "。\n"); it is
                    # part of the span and must receive the mass.
                    if token_decoded[:token_decoded.index("\n")].strip():
                        current.append(i)
                    tail = token_decoded[token_decoded.rindex("\n") + 1:]
                else:
                    tail = token_decoded[nxt:]
                _close()
                decoded_so_far = tail
            else:
                current.append(i)
        else:
            # F5: 概念テキスト外でのみ、直近 _SCAN_TAIL_CHARS 文字だけを残す。
            # 現トークン自体は必ず丸ごと残るので、マーカーの取りこぼしはない。
            decoded_so_far = decoded_so_far[-_SCAN_TAIL_CHARS:] + token_decoded
            m = (_MARKER_PATTERN if level_markers else _PN_PATTERN).search(decoded_so_far)
            if m:
                if level_markers:
                    # [SN]/[RN] carry no mass (0.0); [PN{mass}] carries the planet mass.
                    in_concept_mass = float(m.group(2)) if m.group(2) else 0.0
                    in_concept_level = _MARKER_LEVEL[m.group(1)[0]]
                else:
                    in_concept_mass = float(m.group(1))
                    # この [PN が現れた行の行頭空白数からレベルを決める
                    head = line_text[:line_text.rindex("[PN")] if "[PN" in line_text else line_text
                    in_concept_level = _level_from_leading_spaces(len(head) - len(head.lstrip(" ")))
                current = []
                residual = decoded_so_far[m.end():]
                decoded_so_far = residual
                # F5: BPE が "]" と後続テキストを 1 トークンに融合した場合
                # (例: "] AAA")、このトークン i 自体が概念テキストを含む。
                # 以前は else 分岐で i を一切追加しなかったため、
                # そのノードの位置がまるごと落ちていた。
                nxt_r = _marker_start(residual, level_markers)
                if "\n" in residual:
                    concept_part, tail = residual.split("\n", 1)
                    terminated = True
                elif nxt_r != -1:
                    concept_part, tail = residual[:nxt_r], residual[nxt_r:]
                    terminated = True
                else:
                    concept_part, tail = residual, residual
                    terminated = False
                if concept_part.strip():
                    current.append(i)
                if terminated:
                    # 概念が同一トークン内で終わっている場合は即終端する
                    # (以前はここを見ておらず mass が次の行へ漏れていた)
                    decoded_so_far = tail
                    _close()

    _close()
    return spans


def find_marker_spans(
    token_ids: list[int], tokenizer
) -> list[tuple[str, float, list[int]]]:
    """Spec-faithful format scan: one entry per [SN]/[PN{mass}]/[RN] marker, in
    prompt order, as (level, mass, concept_token_positions). Suns and
    satellites have mass 0.0 (spec 0079: mass is defined for planets only).
    The level comes from the marker itself, so it does not depend on the
    tokenizer preserving indentation (robust for SentencePiece models).
    """
    return [(level, mass, pos) for mass, level, pos in _scan_pn_spans(
        token_ids, tokenizer, level_markers=True
    )]


def marker_positions(
    spans: list[tuple[str, float, list[int]]],
    *,
    inject_levels: set[str] | None = None,
    satellite_inherit: bool = False,
) -> list[tuple[int, float]]:
    """Turn find_marker_spans output into (position, mass) pairs for the mass
    vector. inject_levels=None -> {"planet"} (spec default). With
    satellite_inherit=True the satellites following a planet take that
    planet's mass (experiment switch, spec-neutral: the spec leaves satellite
    mass undefined). Suns never receive mass in this format.
    """
    levels = inject_levels if inject_levels else {"planet"}
    out: list[tuple[int, float]] = []
    planet_mass = 0.0
    for level, mass, positions in spans:
        if level == "planet":
            planet_mass = mass
            eff = mass
        elif level == "satellite":
            eff = planet_mass if satellite_inherit else mass
        else:
            # sun (or anything else): carries no mass, and it CLOSES the
            # previous planet's inheritance scope (2026-09-07 review). Without
            # the reset a satellite that follows a sun but precedes that sun's
            # first planet would inherit the previous sun's last planet mass,
            # putting the bias on an unrelated fact.
            planet_mass = 0.0
            eff = 0.0
        if level not in levels and not (level == "satellite" and satellite_inherit):
            continue
        if eff <= 0.0:
            continue
        out.extend((p, eff) for p in positions)
    return out


def find_pn_positions(token_ids: list[int], tokenizer) -> list[tuple[int, float]]:
    """
    tokenized された input_ids 上で [PN{mass}] 直後の概念テキスト全体のトークン位置と
    mass を返す。

    §83 (2026-05-18) 修正: 以前は ] の次のトークン (i+1) 1 個のみに mass を付与して
    いたが、 BPE で "CRANE-164" が ["C","RANE","-164"] のように分割されると先頭の "C"
    にしか bias がかからず、 特許 §0080「概念を引力の座標とする」意図と乖離していた。
    修正後は概念テキスト末尾 (改行 or 次の [PN) まで全トークンを mass 付きで返す。

    F5 (2026-09-06) 修正: BPE が "]" と後続テキストを 1 トークンに融合すると
    ("] AAA")、そのノードの位置が 1 つも返らずノードごと落ちていた。
    融合トークンも概念トークンとして返すようになった。

    Returns: list of (token_position, mass)
    """
    return [
        (pos, mass)
        for mass, _level, positions in _scan_pn_spans(token_ids, tokenizer)
        for pos in positions
    ]


def find_pn_spans(token_ids: list[int], tokenizer) -> list[tuple[float, list[int]]]:
    """[PN] の出現ごとに (mass, 概念トークン位置リスト) を **プロンプト順で** 返す。

    F7: レベル判定を空白 (インデント) に頼らないための入口。
    CDSerializer.to_context_block は sun→planet→satellite の深さ優先順で
    行を出力し、予算付きシリアライザ (to_context_block_budgeted) も
    残ったノードを同じ走査順で出力する。したがって

        find_pn_spans(...)[k]   と   levels_from_context_block(block)[k]

    は同じノードを指す。positions_for_levels がこの対応でレベル絞り込みを行う。

    Returns: list of (mass, [token_position, ...])
    """
    return [
        (mass, positions)
        for mass, _level, positions in _scan_pn_spans(token_ids, tokenizer)
    ]


def levels_from_context_block(block: str) -> list[str]:
    """<CONTEXT> ブロック文字列から、ノードのレベル列を出力順で取り出す。

    CDSerializer._node_line が "  " * indent + "[PN{mass}] text" を出力するので、
    行頭の空白数を 2 で割ればレベルが決まる (0→sun, 1→planet, 2 以上→satellite)。
    ラッパー行 (<CONTEXT> / </CONTEXT>) と [PN を含まない行は無視する。

    トークナイザを通さない文字列から求めるため、SentencePiece の
    「先頭空白が decode で消える」問題 (F7) の影響を受けない。
    """
    levels: list[str] = []
    for line in block.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("<"):
            continue
        if "[PN" not in line:
            continue
        head = line[:line.index("[PN")]
        levels.append(_level_from_leading_spaces(len(head) - len(head.lstrip(" "))))
    return levels


def inherit_satellite_mass(
    spans: list[tuple[float, list[int]]],
    node_levels: list[str],
) -> list[tuple[float, list[int]]]:
    """衛星ノードの質量を、直前の惑星ノードの質量に置き換えた spans を返す。

    2026-09-07 Fable review #2: 基準解 CD では正解の 185/189 が衛星行にあるのに、
    衛星の質量は 0.1 固定 (config graph.default_satellite_mass) で、答えを持つ行に
    最小のバイアスが掛かっていた。明細書 (¶0079) は質量を惑星にのみ定義し衛星の
    質量を定めていないので、「惑星の質量を配下の衛星に継承させる」注入モードを
    実験セルとして用意する (attention.satellite_mass_mode = "inherit")。
    テキスト側 ([PN] の数値) は変えない — 読み手が見る文面を変えないため。

    spans と node_levels の件数不一致は ValueError (無音で飛ばさない)。
    惑星より前に現れる衛星 (壊れた直列化) は元の質量のまま。
    """
    if len(spans) != len(node_levels):
        raise ValueError(
            f"span/level count mismatch: {len(spans)} [PN] spans "
            f"but {len(node_levels)} node levels"
        )
    out: list[tuple[float, list[int]]] = []
    planet_mass: float | None = None
    for (mass, positions), level in zip(spans, node_levels):
        if level == "planet":
            planet_mass = mass
            out.append((mass, positions))
        elif level == "satellite" and planet_mass is not None:
            out.append((planet_mass, positions))
        else:
            out.append((mass, positions))
    return out


def positions_for_levels(
    spans: list[tuple[float, list[int]]],
    node_levels: list[str],
    levels: set[str] | None,
) -> list[tuple[int, float]]:
    """find_pn_spans の結果を node_levels で絞り込み、list[(pos, mass)] にする。

    node_levels は「プロンプトへ実際に直列化された CD のノードレベル列」で、
    levels_from_context_block から得るのが本筋 (F7)。

    levels が None または空のときは全レベルを残す (現行動作)。
    spans と node_levels の件数が食い違う場合は、無音で飛ばさず ValueError。
    """
    if len(spans) != len(node_levels):
        raise ValueError(
            f"span/level count mismatch: {len(spans)} [PN] spans "
            f"but {len(node_levels)} node levels"
        )
    out: list[tuple[int, float]] = []
    for (mass, positions), level in zip(spans, node_levels):
        if levels and level not in levels:
            continue
        out.extend((pos, mass) for pos in positions)
    return out


# ── レベル付き [PN] 位置検出 (WO-6 / ¶0079) ────────────────────────────

def find_pn_positions_with_level(
    token_ids: list[int], tokenizer
) -> list[tuple[int, float, str]]:
    """
    find_pn_positions と同一の走査に、行のインデントから求めたレベルを付与して返す。

    **SentencePiece 系トークナイザでは信頼できない**: decode([id]) が語頭の
    空白を落とすため、行頭インデントが復元できずレベルが崩れる (F7)。
    レベル絞り込みには levels_from_context_block + positions_for_levels を
    使うこと。この関数は文字単位トークナイザや後方互換のために残してある。

    レベルは「その [PN が現れた行の行頭の空白数」で決まる
    (デコード済みストリームの最後の '\\n' 以降の先頭空白を数える)。
    0-1 → "sun", 2-3 → "planet", >=4 → "satellite"。

    同一のトークン列に対して
        [(p, m) for p, m, _ in find_pn_positions_with_level(ids, tok)]
        == find_pn_positions(ids, tok)
    が必ず成り立つ (走査ロジックが同一であるため)。

    Returns: list of (token_position, mass, level)
    """
    return [
        (pos, mass, level)
        for mass, level, positions in _scan_pn_spans(token_ids, tokenizer)
        for pos in positions
    ]


def filter_positions_by_level(
    positions: list[tuple[int, float, str]], levels: set[str] | None
) -> list[tuple[int, float]]:
    """
    find_pn_positions_with_level の結果を levels で絞り込み、
    MMatrixBuilder / マスベクトル用の list[(pos, mass)] に落とす。

    levels が None（または空）のときは全レベルを残す（現行動作）。
    """
    if not levels:
        return [(pos, mass) for pos, mass, _ in positions]
    return [(pos, mass) for pos, mass, level in positions if level in levels]


def extract_nodes_prompt(text: str) -> str:
    """
    特許§0039-§0040 準拠の概念抽出プロンプト。

    各チャンクを以下の3項目で100点満点でスコアリングし、
    最高得点の項目に対応するレベルにノードを分類する:
      - 包括性 (sun対応): 全体トピックの要約・表題となり得るか
      - 独立性 (planet対応): 既存の話題とは異なる新しい事実や議論の柱か
      - 詳細度 (satellite対応): 数値・固有名詞・具体的手順など補足情報か

    重要: IDとその対応値・属性が含まれる場合は、関係全体を1ノードに記述する。
    例 「ALPHA」→「CRANE-1」 は1つの sun ノードとして扱う。
    これにより CDSerializer が生成する [PN{mass}] トークンがモデルに正しく解釈される。
    """
    return (
        "以下のテキストから重要な情報を抽出し、JSON形式で返してください。\n"
        "【3項目スコアリング】各情報を以下の3項目で100点満点で評価し、"
        "最高得点の項目に対応する level に分類してください:\n"
        "  - 包括性 (comprehensiveness): 全体トピックの要約・表題となり得るか → sun\n"
        "  - 独立性 (independence): 新しい事実や議論の柱か → planet\n"
        "  - 詳細度 (detail): 数値・固有名詞・具体的手順か → satellite\n"
        "【ルール】ID・コード・名前とその対応値がある場合は、"
        "関係全体を1つのノードに記述すること。\n"
        "  良い例: {\"text\": \"「ALPHA」の対応値は「CRANE-1」\", \"level\": \"sun\", "
        "\"score_comprehensiveness\": 90, \"score_independence\": 60, "
        "\"score_detail\": 30, \"parent_hint\": \"\"}\n"
        "  悪い例: {\"text\": \"ALPHA\", ...} と {\"text\": \"CRANE-1\", ...} を別々に返す\n"
        "フォーマット: [{\"text\": \"...\", \"level\": \"sun|planet|satellite\", "
        "\"score_comprehensiveness\": 0-100, \"score_independence\": 0-100, "
        "\"score_detail\": 0-100, \"parent_hint\": \"親概念名（なければ空文字）\"}]\n\n"
        f"テキスト:\n{text}\n\n"
        "JSON:"
    )
