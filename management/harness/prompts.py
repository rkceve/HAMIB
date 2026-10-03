"""Prompt texts and answer defaults for the manager harness.

Every prompt follows the same rules:
  * Instructions are in English because the benchmark conversations are
    English, but extracted text must be written in the source's language, so
    a Japanese conversation still produces Japanese nodes.
  * One question per prompt.
  * Source text is quoted verbatim between ``<<<`` and ``>>>`` lines, so a
    rule-based test judge can read it back (``backends.first_fenced_span``).
  * The answer format is the last sentence.
  * No braces, so ``str.format`` needs no escaping (Q_NODE's example line is
    the one exception).
"""

from __future__ import annotations

# Question kinds: keys of the call counters and part of the answer-cache key.
K_BOUNDARY = "boundary"
K_EXTRACT = "extract"
K_SUPPORTED = "supported"
K_COMPREHENSIVE = "comprehensive"
K_INDEPENDENT = "independent"
K_DETAIL = "detail"
K_BELONGS = "belongs"
K_SAME = "same"
# SpecManager's single question per chunk: it returns the node text and the
# three 0-100 axis scores as one JSON object, replacing the extract,
# supported and three yes/no axis questions.
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

# Fence around every verbatim source span.
FENCE_OPEN = "<<<"
FENCE_CLOSE = ">>>"

# Answer-format lines, stated last in every prompt.
YES_NO_FORMAT = "Answer with exactly one word: yes or no."
JSON_FORMAT = "Return only a JSON array of strings."

# -- step 1: chunk boundary --------------------------------------------------
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

# -- step 2: statement extraction --------------------------------------------
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

# -- step 3: is the statement supported by its source? -----------------------
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

# -- step 4: the three classification axes -----------------------------------
# The two topic-level questions also show the discussion the statement came
# from and the topics already in the diagram, because "is this a heading?"
# cannot be answered from the statement alone.  Q_DETAIL needs no context:
# numbers, proper nouns and concrete steps are visible in the statement.
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

# -- step 5: does a node belong under a topic? -------------------------------
# {a} is the STATEMENT and {b} the TOPIC, because SimilarityJudge always
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

# -- step 6: are two nodes the same matter? ----------------------------------
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

# Template SimilarityJudge uses for each pairwise question kind.
PAIRWISE_PROMPTS: dict[str, str] = {
    K_SAME: Q_SAME,
    K_BELONGS: Q_BELONGS,
}

# "Current topics:" text when the base diagram is still empty.
NO_TOPICS = "none"

# -- SpecManager: one node per chunk -----------------------------------------
# {max_chars} is SpecConfig.max_node_chars.
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

# Q_NODE's last line as it reads after formatting.  The example is there
# because a model that ignores a prose format line usually still copies a
# concrete example.  The braces are doubled in Q_NODE only because it goes
# through str.format.
Q_NODE_EXAMPLE_LINE = (
    'Example: {"summary": "...", "comprehensiveness": 40, '
    '"independence": 70, "detail": 20}'
)

# Length of the fallback node text when Q_NODE never returns a valid object
# (same 80 characters as manager.FALLBACK_STATEMENT_CHARS).
FALLBACK_NODE_CHARS = 80

# A fallback node scores zero on every axis, which makes it a satellite.
DEFAULT_NODE_SCORES: dict[str, int] = {
    "comprehensiveness": 0,
    "independence": 0,
    "detail": 0,
}

# -- retry suffixes ----------------------------------------------------------
RETRY_YES_NO_SUFFIX = "Reply with the single word yes or the single word no, nothing else."
RETRY_JSON_SUFFIX = "Reply with a JSON array of strings only, nothing else."
# Q_NODE expects an object; RETRY_JSON_SUFFIX would ask for an array.
RETRY_JSON_OBJECT_SUFFIX = "Reply with a JSON object only, nothing else."

# -- answer used when the model never gives a parsable yes/no ----------------
DEFAULT_ANSWERS: dict[str, bool] = {
    # A wrong merge destroys a fact for good, while a spurious extra node only
    # costs tokens, so an unreadable answer must not merge or attach.
    K_SAME: False,
    K_BELONGS: False,
    # A statement that cannot be checked against its source is dropped.
    K_SUPPORTED: False,
    # All three axes "no" makes a satellite, which is also the most common
    # level (207 of the 240 nodes in the hand-built reference diagram).
    K_COMPREHENSIVE: False,
    K_INDEPENDENT: False,
    K_DETAIL: False,
    # Keeping a split loses nothing; wrongly merging two chunks can hide the
    # second topic entirely.
    K_BOUNDARY: True,
    # K_NODE is never asked as yes/no (SpecManager parses its JSON reply);
    # the entry only keeps this dict complete over ALL_KINDS.  The real
    # Q_NODE fallback is DEFAULT_NODE_SCORES.
    K_NODE: False,
}
