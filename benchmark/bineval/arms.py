"""Build the context text that each bineval arm shows the reader.

An arm is one way of fitting the chat into the reader's prompt: the whole chat,
its most recent sessions, a fixed summary, or a serialized correlation diagram
(CD) cut to a token budget.  Every budget is ``raw_chat_tokens / ratio``, counted
with tiktoken ``cl100k_base`` against the same raw chat, so arms with the same
ratio get the same amount of context.  Everything here is deterministic and
CPU-only.

Arm names
---------
``full``              every session except ``EXCLUDED_SESSIONS``.
``trunc_{r}x``        the most recent sessions that fit in raw/r tokens.
``summary_9x``        ``results/pilot/summary_6x.txt`` verbatim.  The file name
                      says 6x, but its measured ratio is 9.0x, so the arm is
                      named after the measured ratio (recorded in ``meta``).
``cd_{policy}_{r}x``  a CD JSON serialized within raw/r tokens.
``oracle_cd_full``    ``results/pilot/oracle_cd_full.txt`` verbatim.  It uses an
                      old format with ``[PN{mass}]`` on every line, so the reader
                      runs it without mass injection.
``floor``             empty context.

Session format (do not change it: the published token counts 85,651 / 26,621 /
14,801 / 6,228 for trunc 2/6/10/20x depend on it)::

    === Session {n} ({date}) ===
    User: ...
    Assistant: ...

with sessions joined by a blank line.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import tiktoken

from communication.cd_serializer import CDSerializer
from models.correlation_diagram import CorrelationDiagram
from models.node import Node, NodeLevel

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHAT = REPO_ROOT / "benchmark" / "longchat" / "restaurant_chat_v2.json"
PILOT_DIR = REPO_ROOT / "benchmark" / "bineval" / "results" / "pilot"
EXCLUSIONS_MD = Path(__file__).resolve().parent / "EXCLUSIONS.md"

# Whole sessions of the restaurant chat that are excluded from scoring, as listed
# in EXCLUSIONS.md (section 1 and the v1.1 table).  A test parses EXCLUSIONS.md
# with excluded_sessions_from_markdown() and checks it against this set.
EXCLUDED_SESSIONS: frozenset[int] = frozenset({19, 24, 26, 32, 33, 35, 40})

TRUNC_HEADER = (
    "[NOTE: earlier sessions were dropped to fit a token budget; "
    "this is the most recent part of the conversation only]"
)


# --------------------------------------------------------------------------
# tokenisation
# --------------------------------------------------------------------------

def make_token_counter() -> Callable[[str], int]:
    """tiktoken ``cl100k_base`` token counter: the unit of every budget."""
    enc = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(enc.encode(text))


# --------------------------------------------------------------------------
# chat formatting
# --------------------------------------------------------------------------

def load_chat(path: str | Path = DEFAULT_CHAT) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def format_session(session: dict) -> str:
    """One session as a header line plus ``User:`` / ``Assistant:`` turns."""
    head = "=== Session %s (%s) ===" % (session.get("n"), session.get("date", ""))
    lines = [head]
    for turn in session.get("turns", []):
        role = "User" if turn.get("role") == "user" else "Assistant"
        lines.append("%s: %s" % (role, turn.get("content", "")))
    return "\n".join(lines)


def format_sessions(sessions: Iterable[dict]) -> str:
    return "\n\n".join(format_session(s) for s in sessions)


def raw_tokens(chat: dict, counter: Callable[[str], int] | None = None) -> int:
    """The raw-chat token count that every budget is divided from.

    Uses the count stored in the chat file when present (172,773 for the
    restaurant chat, the published number); otherwise counts the formatted chat.
    """
    declared = chat.get("total_tokens_tiktoken")
    if isinstance(declared, int) and declared > 0:
        return declared
    counter = counter or make_token_counter()
    return counter(format_sessions(chat.get("sessions", [])))


def included_sessions(chat: dict, excluded: Iterable[int] | None = None) -> list[dict]:
    drop = set(EXCLUDED_SESSIONS if excluded is None else excluded)
    return [s for s in chat.get("sessions", []) if int(s.get("n", -1)) not in drop]


def truncation_header_tokens(counter: Callable[[str], int] | None = None) -> int:
    """Token cost of the truncation note plus its blank-line separator.

    The note is part of the arm's context, so it is paid for out of the arm's
    budget; otherwise ``trunc_{r}x`` would get 24 more tokens of context than
    the other arms at the same ratio.
    """
    counter = counter or make_token_counter()
    return counter(TRUNC_HEADER + "\n\n")


def truncation_sessions(
    chat: dict,
    ratio: float,
    counter: Callable[[str], int] | None = None,
    *,
    header_tokens: int = 0,
) -> list[dict]:
    """The most recent sessions whose formatted tokens fit ``raw / ratio - header_tokens``.

    Walks back from the newest session and stops at the first one that does not
    fit; it never skips ahead to a smaller, older session, so the result is
    always a contiguous recent window.  Unlike ``full`` it keeps
    ``EXCLUDED_SESSIONS``: the published truncation arms were built that way.

    At the published ratios the 24-token header changes neither the session
    counts ({2: 23, 6: 7, 10: 4, 20: 2}) nor the body token counts; a test
    checks both.
    """
    counter = counter or make_token_counter()
    budget = raw_tokens(chat, counter) / float(ratio) - float(header_tokens)
    total = 0.0
    kept: list[dict] = []
    for session in reversed(list(chat.get("sessions", []))):
        n = counter(format_session(session))
        if total + n > budget:
            break
        total += n
        kept.append(session)
    kept.reverse()
    return kept


# --------------------------------------------------------------------------
# CD reconstruction
# --------------------------------------------------------------------------

def cd_from_records(records: list[dict]) -> CorrelationDiagram:
    """Rebuild a CorrelationDiagram from the ``nodes`` records of a CD JSON.

    Nodes are attached in three passes (suns, then planets, then satellites) so
    that every parent exists before its children; a planet or satellite whose
    parent is not in ``records`` is dropped.  ``build_cd_offline`` loads its
    checkpoints with this function too.
    """
    by_id: dict[str, Node] = {}
    for r in records:
        node = Node(
            text=r["text"],
            level=NodeLevel(r["level"]),
            mass=float(r["mass"]),
            node_id=r["node_id"],
            parent_id=r.get("parent_id"),
            created_turn=r.get("created_turn", -1),
        )
        by_id[node.node_id] = node

    nodes = [by_id[r["node_id"]] for r in records]
    cd = CorrelationDiagram()
    for node in nodes:
        if node.level == NodeLevel.SUN:
            cd.add_sun(node)
    for node in nodes:
        if node.level == NodeLevel.PLANET and node.parent_id in by_id:
            cd.add_planet(node, node.parent_id)
    for node in nodes:
        if node.level == NodeLevel.SATELLITE and node.parent_id in by_id:
            cd.add_satellite(node, node.parent_id)
    return cd


def load_cd(cd_json: str | Path) -> CorrelationDiagram:
    with Path(cd_json).open(encoding="utf-8") as f:
        data = json.load(f)
    return cd_from_records(data.get("nodes", []))


# --------------------------------------------------------------------------
# arm names
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ArmSpec:
    kind: str                     # full | trunc | summary | cd | oracle | floor
    ratio: float | None = None
    policy: str | None = None


_TRUNC_RE = re.compile(r"^trunc_(\d+(?:\.\d+)?)x$")
_CD_RE = re.compile(r"^cd_(mass|random|recency)_(\d+(?:\.\d+)?)x$")
_SUMMARY_RE = re.compile(r"^summary_(\d+(?:\.\d+)?)x$")


def parse_arm(arm: str) -> ArmSpec:
    """Decompose an arm name. Raises ValueError on an unknown name."""
    if arm == "full":
        return ArmSpec("full")
    if arm == "floor":
        return ArmSpec("floor")
    if arm == "oracle_cd_full":
        return ArmSpec("oracle")
    m = _TRUNC_RE.match(arm)
    if m:
        return ArmSpec("trunc", ratio=float(m.group(1)))
    m = _CD_RE.match(arm)
    if m:
        return ArmSpec("cd", ratio=float(m.group(2)), policy=m.group(1))
    m = _SUMMARY_RE.match(arm)
    if m:
        return ArmSpec("summary", ratio=float(m.group(1)))
    raise ValueError(
        "unknown arm %r (expected full | floor | oracle_cd_full | trunc_{r}x | "
        "summary_{r}x | cd_{mass,random,recency}_{r}x)" % (arm,)
    )


# --------------------------------------------------------------------------
# ArmContext
# --------------------------------------------------------------------------

@dataclass
class ArmContext:
    name: str
    text: str
    tokens: int
    meta: dict = field(default_factory=dict)


def _read_verbatim(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError("arm source file not found: %s" % path)
    return path.read_text(encoding="utf-8")


def build_arm_context(
    arm: str,
    *,
    chat: dict | None = None,
    cd_json: str | Path | None = None,
    ratio: float | None = None,
    policy: str | None = None,
    seed: int = 0,
    level_markers: bool = True,
    pilot_dir: Path = PILOT_DIR,
    counter: Callable[[str], int] | None = None,
) -> ArmContext:
    """Build one arm's context text and its metadata.

    A ratio or policy carried in the arm name wins over the keyword arguments,
    which are only used when the name does not carry them.
    """
    counter = counter or make_token_counter()
    spec = parse_arm(arm)
    eff_ratio = spec.ratio if spec.ratio is not None else ratio
    eff_policy = spec.policy or policy
    meta: dict = {"arm": arm, "kind": spec.kind}

    if spec.kind == "floor":
        text = ""

    elif spec.kind == "full":
        if chat is None:
            raise ValueError("arm 'full' needs chat=")
        kept = included_sessions(chat)
        text = format_sessions(kept)
        meta["sessions"] = [s.get("n") for s in kept]
        meta["excluded_sessions"] = sorted(EXCLUDED_SESSIONS)

    elif spec.kind == "trunc":
        if chat is None:
            raise ValueError("arm %r needs chat=" % arm)
        if eff_ratio is None:
            raise ValueError("arm %r needs a ratio" % arm)
        header_tokens = truncation_header_tokens(counter)
        kept = truncation_sessions(
            chat, eff_ratio, counter, header_tokens=header_tokens
        )
        text = TRUNC_HEADER + "\n\n" + format_sessions(kept)
        meta["sessions"] = [s.get("n") for s in kept]
        meta["n_sessions"] = len(kept)
        meta["budget"] = int(raw_tokens(chat, counter) / eff_ratio)
        meta["header_tokens"] = header_tokens
        meta["body_tokens"] = counter(format_sessions(kept))

    elif spec.kind == "summary":
        # The file keeps its old "6x" name; see the module docstring.
        text = _read_verbatim(pilot_dir / "summary_6x.txt")
        meta["source_file"] = "results/pilot/summary_6x.txt"

    elif spec.kind == "oracle":
        text = _read_verbatim(pilot_dir / "oracle_cd_full.txt")
        meta["source_file"] = "results/pilot/oracle_cd_full.txt"
        # Old format with [PN{mass}] on every line: run without mass injection.
        meta["legacy_pn_format"] = True

    elif spec.kind == "cd":
        if cd_json is None:
            raise ValueError("arm %r needs cd_json=" % arm)
        if chat is None:
            raise ValueError("arm %r needs chat= (the budget base is the raw chat)" % arm)
        if eff_ratio is None or eff_policy is None:
            raise ValueError("arm %r needs a ratio and a policy" % arm)
        cd = load_cd(cd_json)
        budget = int(raw_tokens(chat, counter) / eff_ratio)
        serializer = CDSerializer(level_markers=level_markers)
        text = serializer.to_context_block_budgeted(
            cd, budget, eff_policy, counter, seed  # type: ignore[arg-type]
        )
        meta.update(
            {
                "policy": eff_policy,
                "budget": budget,
                "seed": seed,
                "level_markers": level_markers,
                "cd_json": str(cd_json),
                "nodes_kept": sum(
                    1
                    for ln in text.splitlines()
                    if ln and ln not in ("<CONTEXT>", "</CONTEXT>")
                ),
            }
        )
    else:  # pragma: no cover - parse_arm covers every kind
        raise ValueError("unhandled arm kind %r" % spec.kind)

    tokens = counter(text)
    meta["tokens"] = tokens
    if chat is not None:
        base = raw_tokens(chat, counter)
        meta["raw_tokens"] = base
        meta["measured_ratio"] = round(base / tokens, 2) if tokens else None
    return ArmContext(name=arm, text=text, tokens=tokens, meta=meta)


# --------------------------------------------------------------------------
# EXCLUSIONS.md parser (used by the test; kept next to the constant it checks)
# --------------------------------------------------------------------------

_WHOLE_SESSION_HEADING = re.compile(
    r"^##\s*1\.\s*Whole-session exclusions[^\n]*?sessions\s*([0-9,\s]+)$", re.M
)
_V11_ROW = re.compile(r"^\|\s*Restaurant sessions\s*([0-9,\s]+?)\s*\(whole-session\)", re.M)


def excluded_sessions_from_markdown(path: str | Path = EXCLUSIONS_MD) -> set[int]:
    """Parse the whole-session restaurant exclusions out of EXCLUSIONS.md."""
    text = Path(path).read_text(encoding="utf-8")
    found: set[int] = set()
    hits = _WHOLE_SESSION_HEADING.findall(text) + _V11_ROW.findall(text)
    if not hits:
        raise ValueError("no whole-session exclusion lines found in %s" % path)
    for group in hits:
        found.update(int(n) for n in re.findall(r"\d+", group))
    return found
