"""Build a correlation diagram (CD) offline from a longchat JSON.

Feeds the chat to a CD manager turn by turn and writes the finished diagram as
JSON.  No generation model and no attention patch are loaded.

Turns are numbered sequentially over all sessions (one turn per message;
restaurant_chat_v2.json has 664), and every node records the turn it was
created on.  Per turn, the default path is the one in
``server.cms_session._update_cd``, wired up directly here because importing
``cms_session`` loads the generation model::

    provisional = CorrelationDiagram()
    for chunk in chunker.chunk_turn(user_text, assistant_text, turn):
        classifier.classify(chunk, provisional, extractor_fn, builder=builder)
    merger.merge(base, provisional)          # also normalizes mass and coordinates

``--extractor`` selects who extracts the facts and files them:

    sbert    (default, the floor) SBERT similarity plus regex rules from
             ``server.sbert_extractor``; reproduces the original online path.
    gemma    google/gemma-3-4b-it in 4-bit as the node extractor, with the same
             prompt and JSON fallback rules as cms_session's Gemma path.
    harness  ``management.harness.HarnessManager`` replaces the
             chunk / classify / merge block above.
    spec     ``management.harness.SpecManager``, the manager as the patent
             specification describes it; it updates once per user->assistant
             round trip instead of once per message.

Every extractor except sbert skips query turns: short user questions add no
facts and only pollute the diagram.

Long runs: a failing turn is logged and skipped, but
``--max-consecutive-failures`` failures in a row abort the run.  A checkpoint is
written to ``<out>.ckpt`` every ``--ckpt-interval`` turns, and ``--resume-from``
continues from one.

CLI::

    # sbert floor (default):
    python -m benchmark.bineval.build_cd_offline \
        --chat benchmark/longchat/restaurant_chat_v2.json \
        --out benchmark/bineval/results/cd/restaurant_cd.json

    # Gemma extractor:
    python -m benchmark.bineval.build_cd_offline \
        --chat benchmark/longchat/restaurant_chat_v2.json \
        --extractor gemma \
        --out benchmark/bineval/results/cd/restaurant_cd_gemma.json

    # Smoke: first 2 sessions only (any extractor):
    python -m benchmark.bineval.build_cd_offline \
        --chat benchmark/longchat/restaurant_chat_v2.json \
        --extractor gemma --max-sessions 2 \
        --out benchmark/bineval/results/cd/restaurant_cd_gemma_smoke.json

Output JSON: {"nodes": [{node_id, text, level, mass, parent_id, created_turn}],
"summary": {sun, planet, satellite, total, turns, unbudgeted_tokens,
failed_turns}}.  The sbert path is deterministic.  File I/O is utf-8; console
output is ASCII only (safe on a cp932 Windows console).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmark.bineval.arms import cd_from_records, make_token_counter
from communication.cd_serializer import CDSerializer
from management.graph_builder import GraphBuilder
from management.harness.chunking import is_query_turn_en
from management.graph_merger import GraphMerger
from management.node_classifier import NodeClassifier
from management.text_chunker import TextChunker
from models.correlation_diagram import CorrelationDiagram
from models.node import NodeLevel

# Japanese question phrases, copied from server.cms_session._QUERY_PHRASES (keep
# the two lists in sync).  Copied rather than imported because importing
# cms_session loads the generation model.
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


# vLLM + Qwen3.x emit a <think> trace by default; it eats the token budget and
# the yes/no answer never arrives.  So thinking is off unless the caller says
# otherwise (--judge-extra-body null sends nothing at all).
DEFAULT_JUDGE_EXTRA_BODY = '{"chat_template_kwargs": {"enable_thinking": false}}'

# Default model id of the Anthropic judge backend.  The CLI no longer offers
# that backend (the project makes no external API calls), so nothing in this
# module uses it.
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"

# --judge local: the manager's judge is the same open-weight model as the
# reader, served by vLLM on the same GPU, at 127.0.0.1 where the Modal
# phase_manager starts it.
DEFAULT_LOCAL_JUDGE_BASE_URL = "http://127.0.0.1:8000"

# Extractors whose per-turn work is done by a management/harness manager object.
MANAGER_EXTRACTORS = ("harness", "spec")

# How the chat is grouped into manager updates: the spec manager updates once
# per user->assistant round trip, every other extractor once per message.
# Checkpoints record the mode, because ``turn`` counts different things under
# the two, and resuming across them would skip the wrong turns.
PAIRING_ROUND_TRIP = "round_trip"
PAIRING_MESSAGE = "message"


def pairing_mode(pair_round_trips: bool) -> str:
    return PAIRING_ROUND_TRIP if pair_round_trips else PAIRING_MESSAGE


def git_sha(default: str = "unknown") -> str:
    """Short git sha of the working tree, or ``default``.

    Stamped into every checkpoint so that a resume across a code change is at
    least visible.  Never raises: no git, no repo, or git not on PATH all give
    ``default``.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:  # noqa: BLE001 -- must never raise
        return default
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else default


def _is_query_turn(user_text: str) -> bool:
    """Is this user message a question?  Same rule as ``cms_session._is_query_turn``.

    Questions add no new facts, so feeding them to the manager only pollutes
    the CD.  A message of at most 300 characters is a question when it ends
    with "?" (full- or half-width) or contains one of ``_QUERY_PHRASES``.  The
    bare "kudasai" ("please") is deliberately not matched on its own: requests
    that state a fact ("please record this exactly") contain it too, and
    matching it made the manager skip fact turns.
    """
    text = user_text.strip()
    if len(text) > 300:
        return False
    if text.endswith("？") or text.endswith("?"):
        return True
    return any(phrase in text for phrase in _QUERY_PHRASES)


def _safe_extractor_fn(raw_fn):
    """Wrap an extractor the way cms_session does: keep only dict entries with a
    'text' key, and fall back to a single satellite node (the first 80
    characters of the text) when the extractor raises or returns a non-list.
    """

    def fn(text: str) -> list[dict]:
        try:
            nodes = raw_fn(text)
            if isinstance(nodes, list):
                return [n for n in nodes if isinstance(n, dict) and "text" in n]
        except Exception:
            pass
        return [{"text": text[:80].strip(), "level": "satellite", "parent_hint": ""}]

    return fn


# ── Gemma extractor ─────────────────────────────────────────────────────────


def make_gemma_extractor_fn(model_id: str = "google/gemma-3-4b-it"):
    """Load ``model_id`` in 4-bit and return ``(extract_fn, unload_fn)``.

    ``extract_fn`` does what cms_session's Gemma extraction does: the prompt
    from ``server.cd_parser.extract_nodes_prompt``, greedy decoding, then the
    JSON between the first "[" and the last "]", keeping the dict entries that
    have a "text" key; anything else falls back to a single satellite node.
    Only the extraction model is loaded (no generation model, no attention
    patch).  The 4B model is the default because the 1B model extracted only
    about 60% of the facts (4B: 100%).

    ``unload_fn`` frees the model and the CUDA cache.
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    from server.cd_parser import extract_nodes_prompt

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    max_new_tokens = 256  # same budget as the server config

    def extract_fn(text: str) -> list[dict]:
        prompt = extract_nodes_prompt(text)
        target_device = model.device
        try:
            target_device = next(model.parameters()).device
        except Exception:
            pass
        inputs = tokenizer(prompt, return_tensors="pt").to(target_device)
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        new_ids = output_ids[0, input_ids.shape[1]:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=True)
        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start != -1 and end > start:
                nodes = json.loads(raw[start:end])
                if isinstance(nodes, list):
                    return [n for n in nodes if isinstance(n, dict) and "text" in n]
        except Exception:
            pass
        # Fallback: the whole text as one satellite node.
        return [{"text": text[:80].strip(), "level": "satellite", "parent_hint": ""}]

    def unload_fn() -> None:
        nonlocal model, tokenizer
        try:
            del model
            del tokenizer
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return extract_fn, unload_fn


def _round_trip_iter(chat: dict):
    """Yield (session_idx, user_text, assistant_text) once per round trip.

    The spec manager updates the diagram once per user->assistant exchange
    (specification paragraph 0036).  A user message is paired with the
    assistant message that follows it in the same session; an unpaired message
    is yielded alone with the other side empty, and a message with any other
    role is yielded as user text.
    """
    for s_idx, session in enumerate(chat.get("sessions", [])):
        pending_user: str | None = None
        for msg in session.get("turns", []):
            role, content = msg.get("role", ""), msg.get("content", "")
            if role == "user":
                if pending_user is not None:
                    yield s_idx, pending_user, ""
                pending_user = content
            elif role == "assistant":
                yield s_idx, (pending_user or ""), content
                pending_user = None
            else:
                yield s_idx, content, ""
        if pending_user is not None:
            yield s_idx, pending_user, ""


def _msg_iter(chat: dict):
    """Yield (session_idx, role, content) for every message of every session, in order."""
    for s_idx, session in enumerate(chat.get("sessions", [])):
        for msg in session.get("turns", []):
            yield s_idx, msg.get("role", ""), msg.get("content", "")


def build_cd(
    chat: dict,
    extractor_fn,
    *,
    apply_d7: bool = True,
    max_sessions: int | None = None,
    ckpt_path: Path | None = None,
    ckpt_interval: int = 50,
    resume_state: dict | None = None,
    manager=None,
    max_consecutive_failures: int = 5,
    pair_round_trips: bool = False,
) -> tuple[CorrelationDiagram, int, int]:
    """Feed the chat to the manager turn by turn.  Returns (cd, n_turns, failed_turns).

    ``n_turns`` counts every turn seen (messages, or round trips with
    ``pair_round_trips``), including skipped and resumed ones.

    apply_d7:       skip query-form user messages (``_is_query_turn``); with a
                    ``manager``, English questions are skipped too.
    max_sessions:   process only the first N sessions (smoke runs).
    ckpt_path:      if set, write a checkpoint every ``ckpt_interval`` turns.
    resume_state:   from ``_load_checkpoint``: turns before
                    ``resume_state["turn"]`` are skipped and
                    ``resume_state["cd"]`` is the starting diagram.
    manager:        a harness or spec manager.  ``manager.update(base,
                    user_text, assistant_text, turn)`` then replaces the
                    chunk / classify / merge block, and ``extractor_fn`` is
                    unused (pass None).
    max_consecutive_failures:
                    a failing turn is logged and skipped, but this many
                    failures in a row write a checkpoint and abort with
                    SystemExit.  Otherwise an unreachable judge would silently
                    produce an empty diagram after hundreds of warnings.
    pair_round_trips:
                    one update per user->assistant round trip instead of one
                    per message (the spec manager).
    """
    chunker = TextChunker()
    classifier = NodeClassifier()
    builder = GraphBuilder()
    merger = GraphMerger()
    # Stamped into every checkpoint this run writes.
    pairing = pairing_mode(pair_round_trips)
    code_version = git_sha()

    base = resume_state["cd"] if resume_state and "cd" in resume_state else CorrelationDiagram()
    resume_turn = resume_state["turn"] if resume_state else 0

    def is_query(text: str) -> bool:
        # Manager runs (English corpus) also apply the English rule; the gemma
        # and sbert runs keep the Japanese rule only, so their published
        # numbers still reproduce.
        return _is_query_turn(text) or (manager is not None and is_query_turn_en(text))

    turn = 0
    last_session_idx = 0
    failed_turns = 0
    consecutive_failures = 0
    if pair_round_trips:
        units = ((s, "pair", (u, a)) for s, u, a in _round_trip_iter(chat))
    else:
        units = _msg_iter(chat)

    for s_idx, role, content in units:
        if max_sessions is not None and s_idx >= max_sessions:
            break
        last_session_idx = s_idx

        if turn < resume_turn:  # already processed before the checkpoint
            turn += 1
            continue

        if role == "pair":
            user_text, assistant_text = content
            # A question and its answer (which only restates known facts) are
            # skipped together.
            if apply_d7 and user_text and is_query(user_text):
                turn += 1
                continue
        else:
            if apply_d7 and role == "user" and is_query(content):
                turn += 1
                continue
            user_text = "" if role == "assistant" else content
            assistant_text = content if role == "assistant" else ""

        try:
            if manager is not None:
                # The manager owns chunking, extraction, classification,
                # linking, merging and normalization.
                manager.update(base, user_text, assistant_text, turn)
            else:
                chunks = chunker.chunk_turn(user_text, assistant_text, turn)
                provisional = CorrelationDiagram()
                for chunk in chunks:
                    classifier.classify(chunk, provisional, extractor_fn, builder=builder)
                merger.merge(base, provisional)
            consecutive_failures = 0
        except Exception as e:  # noqa: BLE001 -- a bad turn must not kill the run
            msg = str(e).encode("ascii", "replace").decode("ascii")
            print("[warn] turn %d (session %d) failed, skipped: %s" % (turn, s_idx, msg))
            failed_turns += 1
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                if ckpt_path is not None:
                    _write_checkpoint(
                        ckpt_path, base, turn + 1, s_idx, manager=manager,
                        pairing=pairing, code_version=code_version,
                    )
                raise SystemExit(
                    "aborting: %d consecutive failed turns (last at turn %d): %s"
                    % (consecutive_failures, turn, msg)
                )

        turn += 1

        if ckpt_path is not None and turn % ckpt_interval == 0:
            _write_checkpoint(
                ckpt_path, base, turn, last_session_idx, manager=manager,
                pairing=pairing, code_version=code_version,
            )
            print("[ckpt] wrote %s at turn %d" % (ckpt_path.as_posix(), turn))

    return base, turn, failed_turns


def cd_to_records(cd: CorrelationDiagram) -> list[dict]:
    """Flatten the CD to a node record list in traversal order."""
    records: list[dict] = []
    for se in cd.suns:
        records.append(_node_record(se.sun))
        for pe in se.planets:
            records.append(_node_record(pe.planet))
            for sat in pe.satellites:
                records.append(_node_record(sat))
    return records


def _node_record(node) -> dict:
    return {
        "node_id": node.node_id,
        "text": node.text,
        "level": node.level.value,
        "mass": node.mass,
        "parent_id": node.parent_id,
        "created_turn": node.created_turn,
    }


def summarize(
    cd: CorrelationDiagram,
    n_turns: int,
    unbudgeted_tokens: int,
    failed_turns: int = 0,
) -> dict:
    counts = {NodeLevel.SUN.value: 0, NodeLevel.PLANET.value: 0, NodeLevel.SATELLITE.value: 0}
    for n in cd.all_nodes():
        counts[n.level.value] = counts.get(n.level.value, 0) + 1
    total = sum(counts.values())
    return {
        "sun": counts[NodeLevel.SUN.value],
        "planet": counts[NodeLevel.PLANET.value],
        "satellite": counts[NodeLevel.SATELLITE.value],
        "total": total,
        "turns": n_turns,
        "unbudgeted_tokens": unbudgeted_tokens,
        "failed_turns": failed_turns,
    }


# ── checkpoint / resume ─────────────────────────────────────────────────────


def _checkpoint_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.name + ".ckpt")


def _write_checkpoint(
    ckpt_path: Path,
    cd: CorrelationDiagram,
    turn: int,
    session_idx: int,
    *,
    manager=None,
    pairing: str = PAIRING_MESSAGE,
    code_version: str | None = None,
) -> None:
    """Write a checkpoint: the final-output schema plus a ``resume`` block.

    Written to a .tmp file and then renamed, so an interrupted write does not
    leave a half-written checkpoint.

    The resume block records ``pairing`` and ``code_version`` (checked by
    ``_load_checkpoint``) and, with a manager, its cumulative call and quality
    counters, so a resumed run reports the cost of the whole run.  The judge's
    answer cache is not saved: it can hold tens of thousands of entries and
    only saves calls, so a resumed run simply re-asks what it needs.
    """
    token_counter = make_token_counter()
    full_tokens = token_counter(CDSerializer().to_context_block(cd))
    resume = {
        "turn": turn,
        "session_idx": session_idx,
        "pairing": pairing,
        "code_version": code_version if code_version is not None else git_sha(),
    }
    if manager is not None:
        resume["harness_totals"] = dict(manager.totals)
        resume["harness_cache_hits"] = manager.total_cache_hits
    payload = {
        "nodes": cd_to_records(cd),
        "summary": summarize(cd, turn, full_tokens),
        "resume": resume,
    }
    tmp = ckpt_path.with_name(ckpt_path.name + ".tmp")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(ckpt_path)


def _load_checkpoint(ckpt_path: Path, *, expected_pairing: str | None = None) -> dict:
    """Read a checkpoint written by ``_write_checkpoint``.

    Returns {"cd", "turn", "pairing", "code_version", "harness_totals",
    "harness_cache_hits"}.  With ``expected_pairing``, a checkpoint written
    under the other pairing mode is refused with SystemExit: ``turn`` counts
    round trips in one mode and messages in the other, so resuming across them
    would skip the wrong turns.  A checkpoint without the field counts as
    "unknown" and is refused too, so stale state on a persistent volume is
    never reused silently.
    """
    with ckpt_path.open(encoding="utf-8") as f:
        data = json.load(f)
    resume_block = data.get("resume", {})
    ckpt_pairing = resume_block.get("pairing", "unknown")
    if expected_pairing is not None and ckpt_pairing != expected_pairing:
        raise SystemExit(
            "refusing to resume %s: it was written with pairing=%r but this run "
            "uses pairing=%r. 'turn' counts round trips under %r and messages "
            "under %r, so resuming across the two would skip a different set of "
            "turns and produce a diagram that matches neither run. Delete the "
            "checkpoint (or point --out at a fresh run id) and start over."
            % (ckpt_path.as_posix(), ckpt_pairing, expected_pairing,
               PAIRING_ROUND_TRIP, PAIRING_MESSAGE)
        )

    return {
        "cd": cd_from_records(data.get("nodes", [])),
        "turn": resume_block.get("turn", 0),
        "pairing": ckpt_pairing,
        "code_version": resume_block.get("code_version", "unknown"),
        "harness_totals": resume_block.get("harness_totals", {}),
        "harness_cache_hits": resume_block.get("harness_cache_hits", 0),
    }


def run_smoke(cd: CorrelationDiagram, serializer: CDSerializer, token_counter) -> None:
    """Print 6x-budget stats for all 3 policies + first 15 lines of mass block.

    Budget = unbudgeted_total_tokens // 6. ASCII-only output (cp932-safe).
    """
    full_block = serializer.to_context_block(cd)
    full_tokens = token_counter(full_block)
    budget = full_tokens // 6

    print("--- smoke test: budgeted serialization at 6x ---")
    print("unbudgeted serialized tokens: %d" % full_tokens)
    print("6x budget (tokens): %d" % budget)

    def node_count(block: str) -> int:
        # count node lines = non-wrapper, non-empty lines
        return sum(
            1 for ln in block.splitlines() if ln and ln not in ("<CONTEXT>", "</CONTEXT>")
        )

    for policy in ("mass", "random", "recency"):
        block = serializer.to_context_block_budgeted(
            cd, budget, policy, token_counter, seed=0
        )
        print(
            "policy=%-8s kept_tokens=%5d kept_nodes=%4d"
            % (policy, token_counter(block), node_count(block))
        )

    mass_block = serializer.to_context_block_budgeted(
        cd, budget, "mass", token_counter, seed=0
    )
    print("--- first 15 lines of mass-policy block (6x) ---")
    for ln in mass_block.splitlines()[:15]:
        # ASCII-safe console: replace any non-ascii with '?'
        print(ln.encode("ascii", "replace").decode("ascii"))


def _make_local_judge(args):
    """Build the ``--judge local`` backend: an OpenAI-compatible server, e.g.
    vLLM serving the same open-weight model as the reader."""
    from management.harness import OpenAICompatJudge

    if not args.judge_model:
        raise SystemExit("--judge local requires --judge-model")
    raw_extra = (
        DEFAULT_JUDGE_EXTRA_BODY
        if args.judge_extra_body is None
        else args.judge_extra_body
    )
    extra_body = json.loads(raw_extra)
    return OpenAICompatJudge(
        args.judge_base_url or DEFAULT_LOCAL_JUDGE_BASE_URL,
        args.judge_model,
        extra_body=extra_body,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a CD offline from a longchat JSON.")
    ap.add_argument("--chat", required=True, help="path to longchat chat JSON")
    ap.add_argument("--out", required=True, help="output CD JSON path")
    ap.add_argument(
        "--extractor",
        choices=("sbert", "gemma", "harness", "spec"),
        default="sbert",
        help=(
            "sbert (Arm A / floor, default), gemma (Arm B / LLM extractor), "
            "harness (management.harness.HarnessManager, the enhanced mode) or "
            "spec (management.harness.SpecManager, the spec-faithful default "
            "path of SPEC_FAITHFUL_DESIGN.md Stream S1)"
        ),
    )
    ap.add_argument(
        "--judge",
        choices=("fake", "local"),
        default=None,
        help=(
            "judge backend for --extractor harness/spec: fake (deterministic, no "
            "network, smoke only) or local (an OpenAI-compatible server such as "
            "vLLM on 127.0.0.1). Directive 5 forbids an external API, so the "
            "anthropic/openai choices were removed from the CLI"
        ),
    )
    ap.add_argument(
        "--judge-model",
        default=None,
        help=(
            "model id for the judge backend. REQUIRED for --judge local (a local "
            "server has no meaningful default)"
        ),
    )
    ap.add_argument(
        "--judge-base-url",
        default=None,
        help=(
            "base URL for --judge local (OpenAI-compatible /v1/chat/completions). "
            "Default: %s" % DEFAULT_LOCAL_JUDGE_BASE_URL
        ),
    )
    ap.add_argument(
        "--judge-extra-body",
        default=None,
        help=(
            "JSON object merged into the --judge local request body. Defaults to "
            "%s (thinking off: a reasoning trace blows the token budget and hides "
            "the answer). Pass the literal null to send nothing."
            % DEFAULT_JUDGE_EXTRA_BODY
        ),
    )
    ap.add_argument(
        "--level-markers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "serialize as [SN] / [PN mass] / [RN] with mass on PLANET nodes only "
            "(spec 0062/0079). Default: on for --extractor spec, off otherwise"
        ),
    )
    ap.add_argument(
        "--planet-mass-floor",
        type=float,
        default=None,
        help=(
            "lower bound of a planet's mass in normalize(). Default: 0.0 for "
            "--extractor spec (0062 literally), the diagram's own default (1.0) "
            "otherwise. Only the spec manager threads this into normalize()"
        ),
    )
    ap.add_argument(
        "--manager-workers",
        type=int,
        default=None,
        help=(
            "B5: number of threads used for the Q_NODE calls of ONE turn "
            "(--extractor spec only). Overrides config.yaml spec_manager."
            "max_workers. Linking and merging always stay sequential, so the "
            "resulting diagram is identical; only the wall time changes. The "
            "Modal phase_manager passes 8"
        ),
    )
    ap.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=5,
        help="abort the run after N consecutive failed turns (0 = never abort)",
    )
    ap.add_argument(
        "--shortlist-k",
        type=int,
        default=None,
        help="harness similarity shortlist size (0 = no shortlist, fully 1-to-1)",
    )
    ap.add_argument(
        "--extract-model-id",
        default="google/gemma-3-4b-it",
        help="HF model id for --extractor gemma (D-5 requires the 4B model)",
    )
    ap.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="process only the first N sessions (smoke)",
    )
    ap.add_argument(
        "--ckpt-interval",
        type=int,
        default=50,
        help="write a partial-CD checkpoint every N turns (gemma runs)",
    )
    ap.add_argument(
        "--resume-from",
        default=None,
        help="resume from a checkpoint JSON written by a prior run",
    )
    ap.add_argument("--seed", type=int, default=0, help="seed for random policy in smoke test")
    args = ap.parse_args()

    chat_path = Path(args.chat)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with chat_path.open(encoding="utf-8") as f:
        chat = json.load(f)

    # Only the spec manager updates once per round trip.  The mode is fixed
    # before the checkpoint is read, so a checkpoint written under the other
    # mode is refused rather than silently reused.
    pair_round_trips = args.extractor == "spec"
    resume_state: dict | None = None
    if args.resume_from:
        resume_state = _load_checkpoint(
            Path(args.resume_from), expected_pairing=pairing_mode(pair_round_trips)
        )
        print(
            "resuming from %s at turn %d (%d nodes, pairing=%s, code_version=%s)"
            % (
                args.resume_from,
                resume_state["turn"],
                len(list(resume_state["cd"].all_nodes())),
                resume_state["pairing"],
                resume_state["code_version"],
            )
        )

    unload_fn = None
    manager = None
    extractor_fn = None
    if args.extractor in MANAGER_EXTRACTORS:
        from management.harness import (
            HarnessConfig,
            HarnessManager,
            SpecConfig,
            SpecManager,
            make_driver_fake_judge,
            make_spec_fake_judge,
        )

        is_spec = args.extractor == "spec"
        m_config = SpecConfig.from_config() if is_spec else HarnessConfig.from_config()
        if args.shortlist_k is not None:
            m_config.shortlist_k = args.shortlist_k
        if args.manager_workers is not None:
            if not is_spec:
                raise SystemExit("--manager-workers requires --extractor spec")
            if args.manager_workers < 1:
                raise SystemExit("--manager-workers must be >= 1")
            m_config.max_workers = args.manager_workers
        if args.planet_mass_floor is not None:
            if not is_spec:
                raise SystemExit("--planet-mass-floor requires --extractor spec")
            m_config.planet_mass_floor = args.planet_mass_floor
        node_chars = (
            m_config.max_node_chars if is_spec else m_config.max_statement_chars
        )
        if args.judge is None:
            raise SystemExit(
                "--extractor %s requires --judge {fake,local} (no silent default: "
                "a fake judge would produce a meaningless CD)" % args.extractor
            )
        if args.judge == "fake":
            # The fake judge cannot rank anything, so an embedding shortlist
            # would only load an SBERT model for nothing.
            m_config.shortlist_k = 0
            judge = (
                make_spec_fake_judge(node_chars)
                if is_spec
                else make_driver_fake_judge(node_chars)
            )
        else:
            judge = _make_local_judge(args)
        print(
            "using %s manager (judge=%s, shortlist_k=%d, max_workers=%d)"
            % (
                args.extractor,
                args.judge,
                m_config.shortlist_k,
                getattr(m_config, "max_workers", 1),
            )
        )
        manager = (
            SpecManager(judge, config=m_config)
            if is_spec
            else HarnessManager(judge, config=m_config)
        )
        if resume_state and resume_state.get("harness_totals"):
            if is_spec:
                manager.load_totals(
                    resume_state["harness_totals"],
                    int(resume_state.get("harness_cache_hits", 0)),
                )
            else:
                manager.load_totals(resume_state["harness_totals"])
                manager.total_cache_hits += int(
                    resume_state.get("harness_cache_hits", 0)
                )
            print("restored manager totals from the checkpoint")
    elif args.extractor == "gemma":
        print("loading Gemma extractor (%s, 4-bit nf4, device_map=auto)..." % args.extract_model_id)
        raw_fn, unload_fn = make_gemma_extractor_fn(args.extract_model_id)
        extractor_fn = _safe_extractor_fn(raw_fn)
    elif args.extractor == "sbert":
        # Lazy import so the module top stays torch-free even if this file is imported.
        from server.sbert_extractor import make_extractor_fn

        print("loading SBERT extractor (all-MiniLM family; local cache if present)...")
        extractor_fn = _safe_extractor_fn(make_extractor_fn())

    ckpt_path = _checkpoint_path(out_path)
    # Query turns are skipped for every extractor except sbert, which must
    # reproduce the original online path (and its published numbers) exactly.
    apply_d7 = args.extractor in ("gemma",) + MANAGER_EXTRACTORS
    print("building CD over all turns (extractor=%s, D-7=%s)..." % (args.extractor, apply_d7))
    t0 = time.perf_counter()
    try:
        cd, n_turns, failed_turns = build_cd(
            chat,
            extractor_fn,
            apply_d7=apply_d7,
            max_sessions=args.max_sessions,
            ckpt_path=ckpt_path,
            ckpt_interval=args.ckpt_interval,
            resume_state=resume_state,
            manager=manager,
            max_consecutive_failures=(
                args.max_consecutive_failures
                if args.max_consecutive_failures > 0
                else 10**9
            ),
            pair_round_trips=pair_round_trips,
        )
    finally:
        if unload_fn is not None:
            unload_fn()
    elapsed = time.perf_counter() - t0

    # The spec run serializes with level markers ([SN] / [PN mass] / [RN], mass
    # on planet lines only), as the specification describes.
    level_markers = (
        (args.extractor == "spec") if args.level_markers is None else args.level_markers
    )
    serializer = CDSerializer(level_markers=level_markers)
    token_counter = make_token_counter()
    full_tokens = token_counter(serializer.to_context_block(cd))

    records = cd_to_records(cd)
    summary = summarize(cd, n_turns, full_tokens, failed_turns)
    if manager is not None:
        summary["harness_calls"] = manager.call_totals()
        summary["harness_cache_hits"] = manager.total_cache_hits
        summary["harness_quality"] = manager.quality_totals()

    payload = {"nodes": records, "summary": summary}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("--- build summary ---")
    print("extractor: %s" % args.extractor)
    print("level markers: %s" % level_markers)
    print("turns processed: %d" % summary["turns"])
    print(
        "nodes: sun=%d planet=%d satellite=%d total=%d"
        % (summary["sun"], summary["planet"], summary["satellite"], summary["total"])
    )
    print("unbudgeted serialized tokens: %d" % full_tokens)
    print("failed turns: %d" % failed_turns)
    if manager is not None:
        calls = summary["harness_calls"]
        quality = summary["harness_quality"]
        total_calls = sum(calls.values())
        print(
            "harness judge calls: total=%d (%s), cache hits=%d"
            % (
                total_calls,
                " ".join("%s=%d" % (k, v) for k, v in sorted(calls.items())) or "none",
                summary["harness_cache_hits"],
            )
        )
        print(
            "harness quality: %s"
            % " ".join("%s=%d" % (k, v) for k, v in sorted(quality.items()))
        )
        if total_calls and quality.get("defaulted", 0) / total_calls > 0.2:
            print(
                "WARNING: %.1f%% of the judge calls fell back to the per-question "
                "default -- the answers are not trustworthy"
                % (100.0 * quality["defaulted"] / total_calls)
            )
    per_turn = (elapsed / n_turns) if n_turns else 0.0
    print("wall time: %.1fs (%.3fs/turn over %d turns)" % (elapsed, per_turn, n_turns))
    print("wrote: %s" % out_path.as_posix())

    run_smoke(cd, serializer, token_counter)


if __name__ == "__main__":
    main()
