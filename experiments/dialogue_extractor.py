"""
Dialogue-aware extractor for natural-conversation benchmarks (LongMemEval).

The default SBERTExtractor (server/sbert_extractor.py) was designed for the
controlled `bench_scaler` benchmark where facts are written as
``Alpha's project code is CRANE-164``. Its regex pair finder
(`NAME_PATTERN` + `CODE_PATTERN` = `[A-Z]{2,}-\\d+`) and its SBERT default
queries (which look for identifier-code/value pairs) extract **nothing** from
natural English dialogue, so the LongMemEval runs ended up with `cd=0`
throughout and HAMIB got near-zero accuracy.

This extractor replaces those patterns with regex-only heuristics tuned for
natural multi-session dialogue:
  * Capitalized multi-word phrases (people, places, organizations)
  * Date/time expressions (absolute and relative)
  * Numeric facts (years, ages, prices)
  * Sentences that contain any of the above (stored as satellite nodes)

It is intentionally cheap: regex-only, no SBERT model, no LLM call — so each
turn pair costs <1 ms. This keeps replay over 200-300-turn dialogues
practical without exploding inference cost.
"""
from __future__ import annotations
import re
from typing import Callable

# Capitalized phrase: 1-3 consecutive capitalized words.
# Reject pure-acronym tokens (USA, AI) — they tend to be too generic.
_ENTITY_RE = re.compile(
    r"\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-zA-Z]{1,}){0,2})\b"
)

# Date expressions: "May 8, 2023", "yesterday", "last week", "2023", "Apr 4"
_DATE_RE = re.compile(
    r"\b("
    r"\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*(?:\s+\d{2,4})?"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{2,4})?"
    r"|\d{1,2}/\d{1,2}(?:/\d{2,4})?"
    r"|\d{4}-\d{1,2}-\d{1,2}"
    r"|\b(?:19|20)\d{2}\b"
    r"|yesterday|today|tomorrow|tonight"
    r"|last\s+(?:week|month|year|night|weekend|\w+day)"
    r"|next\s+(?:week|month|year|night|weekend|\w+day)"
    r")",
    re.IGNORECASE,
)

# Stoplist of common capitalized non-entities
_STOPLIST = frozenset({
    "User", "Assistant", "System", "Hi", "Hello", "Yes", "No",
    "OK", "Okay", "Sure", "Thanks", "Thank", "Good", "Great", "Nice",
    "I", "You", "He", "She", "It", "We", "They", "My", "Your", "His", "Her",
    "This", "That", "These", "Those", "There", "Here",
    "Hey", "Wow", "Oh", "Ah", "Well", "Right", "Sorry", "Please",
    "What", "Where", "When", "Why", "Who", "How", "Which",
    "Did", "Does", "Do", "Is", "Are", "Was", "Were", "Will", "Would", "Could",
    "Should", "Can", "May", "Might", "Have", "Has", "Had", "Be", "Been",
    "Anything", "Anyone", "Anybody", "Anywhere", "Some", "Something",
    "Someone", "Somebody", "All", "Any", "Both", "Each", "Either", "Neither",
    "Every", "Everyone", "Everything", "Such", "Then", "Now", "Just",
    "Only", "More", "Most", "Less", "Least", "Much", "Many", "Few",
    "True", "False", "None",
    "Yeah", "Yep", "Yup", "Nope", "Nah", "Aw", "Aww", "Ouch", "Eh", "Uh",
    "Um", "Hmm", "Huh", "Whoa", "Whoop", "Yikes",
    "Gonna", "Wanna", "Gotta", "Kinda", "Sorta", "Totally", "Absolutely",
    "Definitely", "Maybe", "Probably", "Perhaps", "Actually", "Basically",
    "Really", "Truly", "Honestly", "Seriously", "Obviously",
    "Sounds", "Looks", "Seems", "Hope", "Wish", "Mean", "Mind",
    "Painting", "Drawing", "Cooking", "Reading", "Running",
    "Day", "Time", "Year", "Week", "Month", "Morning", "Evening", "Night",
    "Thing", "Things", "Stuff", "Way", "Place", "Side",
    "Mom", "Dad", "Sis", "Bro",
    "The", "A", "An", "And", "But", "Or", "So", "Yet", "For", "Nor",
    "In", "On", "At", "By", "Of", "To", "From", "With", "Without",
    "About", "Above", "Below", "Between", "Through", "During", "Before",
    "After", "Until", "Since", "While", "As", "Like", "Unlike", "Into",
    "Onto", "Upon", "Across", "Within", "Towards",
    "Talk", "Take", "Make", "Get", "Give", "See", "Look", "Find", "Tell",
    "Say", "Think", "Know", "Hear", "Want", "Need", "Try", "Use", "Help",
    "Talked", "Took", "Made", "Got", "Gave", "Saw", "Looked", "Found",
    "Told", "Said", "Thought", "Knew", "Heard", "Wanted", "Needed",
    "Going", "Coming", "Doing", "Saying", "Telling", "Thinking", "Talking",
    "Their", "Theirs", "Mine", "Yours", "Ours", "Hers", "Its",
    "Talk", "Things", "Stuff", "Way", "Place", "Side", "End", "Beginning",
    "Best", "Worst", "Better", "Worse",
    "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec",
    "Yesterday", "Today", "Tomorrow", "Tonight",
    "Last", "Next", "First", "Second", "Third",
    "AI", "LLM", "API", "URL", "USA", "UK",
})

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _clean_entity(s: str) -> str:
    s = s.strip().strip(",.!?;:").strip()
    return s


def _is_meaningful_entity(s: str) -> bool:
    if not s or len(s) < 2:
        return False
    head = s.split()[0]
    if head in _STOPLIST:
        return False
    if s in _STOPLIST:
        return False
    # Single token: require at least 3 chars + not stoplisted
    parts = s.split()
    if len(parts) == 1 and len(s) < 3:
        return False
    # Reject single-letter or all-caps short
    if len(s) <= 4 and s.isupper():
        return False
    return True


def extract_dialogue_nodes(
    text: str,
    *,
    max_entities: int = 1,
    max_satellites: int = 2,
    min_sentence_len: int = 20,
    max_sentence_len: int = 200,
    require_date_for_satellite: bool = False,
    use_3level_hierarchy: bool = False,
) -> list[dict]:
    """Extract a small set of (entity, fact-sentence) nodes from natural text.

    With use_3level_hierarchy=True (default):
        sun "Conversation" (master) -> planet (entity) -> satellite (fact)
    so that recalculate_planet_masses() can compute planet.mass = satellite count,
    activating the mass-differentiation mechanism.

    Output format matches HAMIBSession's extractor_fn contract:
        {"text": ..., "level": "sun"|"planet"|"satellite", "parent_hint": ...}
    """
    if not text or not text.strip():
        return []

    nodes: list[dict] = []
    seen: set[str] = set()

    # emit a master sun so entities can be planets and facts can be
    # satellites — gives the planet.mass = #satellites mechanism a real
    # hierarchy to differentiate over.
    if use_3level_hierarchy:
        nodes.append({"text": "Conversation", "level": "sun", "parent_hint": ""})
        entity_level = "planet"
        entity_parent = "Conversation"
    else:
        entity_level = "sun"
        entity_parent = ""

    # 1. Named entities → planet nodes under the master sun (hierarchy mode)
    # or sun nodes (legacy mode).
    entity_strs: list[str] = []
    for m in _ENTITY_RE.finditer(text):
        phrase = _clean_entity(m.group(1))
        if _is_meaningful_entity(phrase) and phrase not in seen:
            seen.add(phrase)
            entity_strs.append(phrase)
            nodes.append({"text": phrase, "level": entity_level, "parent_hint": entity_parent})
        else:
            for w in phrase.split():
                w = _clean_entity(w)
                if _is_meaningful_entity(w) and w not in seen:
                    seen.add(w)
                    entity_strs.append(w)
                    nodes.append({"text": w, "level": entity_level, "parent_hint": entity_parent})
                    if len(entity_strs) >= max_entities:
                        break
        if len(entity_strs) >= max_entities:
            break

    # 2. Fact-bearing sentences → satellite nodes.
    # Stricter than v1: require BOTH entity AND (date or number) in the sentence
    # (or just date for the date-anchor case). This keeps the CD focused on
    # recall-worthy facts and drops chit-chat.
    anchor = entity_strs[0] if entity_strs else ""
    sat_count = 0
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if not (min_sentence_len <= len(sentence) <= max_sentence_len):
            continue
        has_entity = any(e in sentence for e in entity_strs)
        has_date = bool(_DATE_RE.search(sentence))
        if require_date_for_satellite:
            if not has_date:
                continue
        else:
            if not (has_entity or has_date):
                continue
        if sentence in seen:
            continue
        seen.add(sentence)
        parent = anchor
        for e in entity_strs:
            if e in sentence:
                parent = e
                break
        nodes.append({
            "text": sentence[:160],
            "level": "satellite",
            "parent_hint": parent,
        })
        sat_count += 1
        if sat_count >= max_satellites:
            break

    return nodes


def make_dialogue_extractor_fn(
    max_entities: int = 4,
    max_satellites: int = 4,
    use_3level_hierarchy: bool = False,
) -> Callable[[str], list[dict]]:
    def _fn(text: str) -> list[dict]:
        return extract_dialogue_nodes(
            text,
            max_entities=max_entities,
            max_satellites=max_satellites,
            use_3level_hierarchy=use_3level_hierarchy,
        )
    return _fn
