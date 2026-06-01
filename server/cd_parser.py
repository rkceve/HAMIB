"""
CDParser: parses the node_list and context_block received from the client and
converts them into a form usable inside the server.

Also provides the logic for the /extract_nodes endpoint, which extracts concept
node candidates from text using Gemma.
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


# ── [PN{mass}] token detection ────────────────────────────────────────────
_PN_PATTERN = re.compile(r"\[PN([\d.]+)\]")


def find_pn_positions(token_ids: list[int], tokenizer) -> list[tuple[int, float]]:
    """
    Return the token positions and mass for the entire concept text immediately
    after each [PN{mass}] marker in the tokenized input_ids.

    Applies the mass to every token of the concept text up to its end (a newline
    or the next [PN]). Applying mass to only the single token after "]" would, when
    BPE splits e.g. "CRANE-164" into ["C","RANE","-164"], bias only the leading "C".

    Returns: list of (token_position, mass)
    """
    positions: list[tuple[int, float]] = []
    decoded_so_far = ""
    in_concept_mass: float | None = None
    for i, tid in enumerate(token_ids):
        token_decoded = tokenizer.decode([tid], skip_special_tokens=False)
        if in_concept_mass is not None:
            # Inside the concept text: terminate at a newline or the next [PN
            if "\n" in token_decoded or "[PN" in token_decoded:
                in_concept_mass = None
                if "\n" in token_decoded:
                    tail = token_decoded[token_decoded.rindex("\n") + 1:]
                else:
                    tail = token_decoded[token_decoded.index("[PN"):]
                decoded_so_far = tail
            else:
                positions.append((i, in_concept_mass))
        else:
            decoded_so_far += token_decoded
            m = _PN_PATTERN.search(decoded_so_far)
            if m:
                in_concept_mass = float(m.group(1))
                decoded_so_far = decoded_so_far[m.end():]
    return positions


def extract_nodes_prompt(text: str) -> str:
    """
    Concept extraction prompt.

    Scores each chunk on the following three criteria out of 100 points each and
    classifies the node at the level corresponding to the highest-scoring criterion:
      - Comprehensiveness (-> sun): could it summarize or title the overall topic?
      - Independence (-> planet): is it a new fact or a distinct point of discussion?
      - Detail (-> satellite): is it supplementary info such as a number, proper noun,
        or concrete procedure?

    Important: when an ID and its corresponding value/attribute are present, describe
    the whole relationship in a single node (e.g. treat "「ALPHA」→「CRANE-1」" as one
    sun node). This ensures the [PN{mass}] tokens produced by CDSerializer are
    interpreted correctly by the model.
    """
    # English translation of the Japanese prompt below (reference only — the
    # Japanese original is the live prompt; translating it would shift the
    # LLM's output distribution and break the recorded experiment numbers,
    # so the original is kept verbatim. See README "A note on language"):
    #
    #   Extract important information from the text below and return it as JSON.
    #   [3-axis scoring] Score each piece of information out of 100 on three
    #   axes, and classify it at the level corresponding to the highest score:
    #     - Comprehensiveness: could it summarize / title the overall topic? -> sun
    #     - Independence:      is it a new fact or a distinct point of discussion? -> planet
    #     - Detail:            is it a number, proper noun, or concrete procedure? -> satellite
    #   [Rule] When an ID/code/name has a corresponding value, describe the
    #   whole relationship in a single node.
    #     Good example: {"text": "ALPHA's corresponding value is CRANE-1",
    #                    "level": "sun", "score_comprehensiveness": 90,
    #                    "score_independence": 60, "score_detail": 30,
    #                    "parent_hint": ""}
    #     Bad example:  returning {"text": "ALPHA"} and {"text": "CRANE-1"} separately
    #   Format: [{"text": "...", "level": "sun|planet|satellite",
    #             "score_comprehensiveness": 0-100, "score_independence": 0-100,
    #             "score_detail": 0-100, "parent_hint": "<parent concept, '' if none>"}]
    #
    #   Text:
    #   {text}
    #
    #   JSON:
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
