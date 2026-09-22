"""Prompt constants and answer defaults for the manager harness.

Design: HARNESS_DESIGN.md Stream B, B4.

Rules followed by every prompt below:
  * ENGLISH instructions.  The design assumed Japanese conversations, but the
    benchmark corpus (benchmark/longchat/restaurant_chat_v2.json) is 100%
    English: 832k characters, zero ``。``, 6498 ``.``.  The prompts therefore
    instruct in English and require the EXTRACTED STATEMENTS to be written in
    the SOURCE's language, so a Japanese conversation still yields Japanese
    nodes (H3a).
  * ONE question per prompt.
  * The source text appears VERBATIM, fenced by ``<<<`` / ``>>>`` so a rule-based
    test judge can recover it (see ``backends.make_driver_fake_judge``).
  * The answer format is stated LAST, as a single sentence.
  * Braces are avoided entirely so ``str.format`` needs no escaping.
"""

from __future__ import annotations

# -- question kinds (used as call-counter keys and as part of the cache key) --
K_BOUNDARY = "boundary"
K_EXTRACT = "extract"
K_SUPPORTED = "supported"
K_COMPREHENSIVE = "comprehensive"
K_INDEPENDENT = "independent"
K_DETAIL = "detail"
K_BELONGS = "belongs"
K_SAME = "same"
# Spec-faithful mode (SPEC_FAITHFUL_DESIGN.md S1.2 step 2): ONE call per chunk
# that returns the node text AND the three 0-100 axis scores as one JSON
# object.  It replaces the extract + supported + 3x yes/no chain.
K_NODE = "node"

ALL_KINDS: tuple[str, ...] = (
    K_BOUNDARY,
    K_EXTRACT,
    K_SUPPORTED,
    K_COMPREHENSIVE,
    K_INDEPENDENT,
    K_DETAIL,
    K_BELONGS,
    K_SAME,
    K_NODE,
)

# Fence used around every verbatim source span.
FENCE_OPEN = "<<<"
FENCE_CLOSE = ">>>"

# Answer-format lines, stated last in every prompt.
YES_NO_FORMAT = "Answer with exactly one word: yes or no."
JSON_FORMAT = "Return only a JSON array of strings."

# -- step 1: chunk boundary (0038) -------------------------------------------
Q_BOUNDARY = """A and B are two consecutive fragments of one conversation.
A:
<<<
{a}
>>>
B:
<<<
{b}
>>>
Does the topic change from A to B? Answer with exactly one word: yes or no."""

# -- step 2: statement extraction (0030 / D-4) -------------------------------
Q_EXTRACT = """Extract the standalone factual statements contained in the source text below.
Source:
<<<
{text}
>>>
Each statement must be at most {max_chars} characters long, must stand on its own
without pronouns whose referent is outside the statement, and must be written in
the same language as the source text.
If the source states no facts, return an empty array.
Return only a JSON array of strings."""

# -- step 3: faithfulness (D-4) ----------------------------------------------
Q_SUPPORTED = """Statement:
<<<
{statement}
>>>
Source:
<<<
{text}
>>>
Is the statement supported by the source text alone?
Answer with exactly one word: yes or no."""

# -- step 4: three classification axes (0039-0040) ---------------------------
# H6: the two topic-level axes see the discussion the statement came from and the
# topics already recorded in the correlation diagram; Q_DETAIL is statement-only
# because "does it carry numbers / proper nouns / concrete steps" needs no context.
Q_COMPREHENSIVE = """Statement:
<<<
{statement}
>>>
Discussion the statement was taken from:
<<<
{context}
>>>
Current topics:
{topics}
Could the statement serve as the title or the summary of that whole discussion?
Answer with exactly one word: yes or no."""

Q_INDEPENDENT = """Statement:
<<<
{statement}
>>>
Discussion the statement was taken from:
<<<
{context}
>>>
Current topics:
{topics}
Does the statement introduce a new fact or a new pillar of the discussion, rather
than a supporting detail of something already stated?
Answer with exactly one word: yes or no."""

Q_DETAIL = """Statement:
<<<
{statement}
>>>
Does the statement supplement one specific matter with a number, a proper noun or
a concrete step? Answer with exactly one word: yes or no."""

# -- step 5: provisional linking (0041) --------------------------------------
# NOTE: {a} is the STATEMENT and {b} the TOPIC, because SimilarityJudge always
# formats a=query, b=candidate and the query here is the statement being placed.
Q_BELONGS = """Statement:
<<<
{a}
>>>
Topic:
<<<
{b}
>>>
Does the statement belong under this topic?
Answer with exactly one word: yes or no."""

# -- step 6 / similarity judge: 1-to-1 sameness (0042) -----------------------
Q_SAME = """Statement A:
<<<
{a}
>>>
Statement B:
<<<
{b}
>>>
Do A and B state the same matter?
Answer with exactly one word: yes or no."""

# Prompt template used by SimilarityJudge for each pairwise question kind.
PAIRWISE_PROMPTS: dict[str, str] = {
    K_SAME: Q_SAME,
    K_BELONGS: Q_BELONGS,
}

# Text used for the "Current topics:" list when the base diagram is still empty.
NO_TOPICS = "none"

# -- spec mode: one node per chunk (0030, 0039-0040) -------------------------
# ONE call per chunk.  The fragment appears VERBATIM inside the fence, the three
# axes are described in the 0039 wording, and the answer format (the JSON keys)
# is stated LAST.  {max_chars} is the node-text cap of SpecConfig.max_node_chars.
Q_NODE = """Summarize the following conversation fragment as ONE self-contained sentence of at most {max_chars} characters, in the same language as the fragment.
Fragment:
<<<
{text}
>>>
Then score that sentence from 0 to 100 on three axes:
comprehensiveness - could it serve as the title or the summary of the whole topic;
independence - does it state a new fact or a new pillar of the discussion, distinct from details;
detail - does it supplement one specific matter with numbers, proper nouns or concrete steps.
Return only a JSON object with the keys summary, comprehensiveness, independence, detail.
Example: {{"summary": "...", "comprehensiveness": 40, "independence": 70, "detail": 20}}"""

# The example line is the LAST line of Q_NODE and restates the exact shape (M8,
# 2026-09-07): a model that ignores a prose format line usually still copies a
# concrete example.  The braces are DOUBLED in the template because Q_NODE goes
# through str.format; the RENDERED prompt carries single braces, so this is the
# only prompt in this file that contains a brace after formatting.
Q_NODE_EXAMPLE_LINE = (
    'Example: {"summary": "...", "comprehensiveness": 40, '
    '"independence": 70, "detail": 20}'
)

# Length of the fallback node text when Q_NODE never returns a valid object.
# Mirrors manager.FALLBACK_STATEMENT_CHARS / build_cd_offline's text[:80].
FALLBACK_NODE_CHARS = 80

# The scores a fallback node carries: all zero -> satellite (0040 fallback).
DEFAULT_NODE_SCORES: dict[str, int] = {
    "comprehensiveness": 0,
    "independence": 0,
    "detail": 0,
}

# -- retry suffixes ----------------------------------------------------------
RETRY_YES_NO_SUFFIX = "Reply with the single word yes or the single word no, nothing else."
RETRY_JSON_SUFFIX = "Reply with a JSON array of strings only, nothing else."
# Q_NODE returns an OBJECT, so the array wording of RETRY_JSON_SUFFIX would
# instruct the model to reply in the wrong shape on the retry.
RETRY_JSON_OBJECT_SUFFIX = "Reply with a JSON object only, nothing else."

# -- per-question defaults when the model never gives a parsable answer -------
DEFAULT_ANSWERS: dict[str, bool] = {
    # Merging is irreversible and a wrong merge destroys a fact, while a spurious
    # extra node only costs tokens -> an unparsable sameness/belonging answer
    # must NOT merge or attach.
    K_SAME: False,
    K_BELONGS: False,
    # D-4: an unverifiable statement is dropped rather than recorded.
    K_SUPPORTED: False,
    # All three axes "no" means satellite (0040 fallback), which is also the
    # majority level in the oracle CD (1 sun / 32 planets / 207 satellites).
    K_COMPREHENSIVE: False,
    K_INDEPENDENT: False,
    K_DETAIL: False,
    # 0038: keeping the split preserves information; a wrong merge of two
    # chunks can hide the second topic entirely.
    K_BOUNDARY: True,
    # K_NODE is never asked as a yes/no question (it returns a JSON object and
    # is handled by SpecManager, not by JudgeRunner.ask_yes_no).  The entry only
    # keeps DEFAULT_ANSWERS total over ALL_KINDS; see DEFAULT_NODE_SCORES for
    # the real Q_NODE fallback.
    K_NODE: False,
}
