"""Build a CorrelationDiagram offline from a longchat JSON, no gen model.

WO-1 / WO-2a (RESEARCH_PROGRAM sec.4). This drives the SAME management path as
the published v10 arm (`server.cms_session._update_cd`, cms_session.py:356-385)
but assembles the components DIRECTLY so that importing this module never pulls
in the torch generation stack (cms_session.py:50 imports the model at module
top).

Two extractor arms (WO-2 two-extractor design):
  --extractor sbert  (default)  Arm A / floor: server.sbert_extractor
                                (SBERT cosine + regex hybrid). Torch-free at
                                import time; the model loads lazily in setup().
  --extractor gemma             Arm B / spec-compliant manager (sec.41 revival):
                                google/gemma-3-4b-it in 4-bit (BitsAndBytesConfig
                                nf4, device_map="auto") loaded via plain
                                `transformers`. Extraction model ONLY -- no
                                generation model, no attention patch, no
                                mass_weighted_* import. The extract function
                                replicates cms_session._llm_extract_fn's Gemma
                                path exactly (same prompt via
                                server.cd_parser.extract_nodes_prompt, same
                                find('[')/rfind(']')+json.loads fallback rules,
                                same 3-axis scoring passed through
                                NodeClassifier). Applies the D-ledger:
                                D-5 (4B model, not 1B), D-7 (compound
                                query-phrase skip); D-3 mass normalization is
                                handled by the existing GraphMerger/serializer.

Pathway per turn (mirrors _update_cd):
    provisional = CorrelationDiagram()
    for chunk in chunker.chunk_turn(user_text, assistant_text, turn):
        classifier.classify(chunk, provisional, extractor_fn, builder=builder)
    merger.merge(base, provisional)          # normalize (mass + coords) inside

The longchat file stores each session as a list of {role, content} messages.
Per canon WO-1 the turn numbering is "sequential over all sessions' turns in
order" (restaurant_chat_v2.json has 664 such messages). We treat every message
as one turn with a global sequential index, build one provisional CD per
message, and stamp created_turn = that index onto every node born on that turn
(management/node_classifier.py threads chunk.turn).

D-7 (query-turn skip): applied to the GEMMA arm ONLY (RESEARCH_PROGRAM WO-2
arm B D-ledger requirement). The SBERT arm (arm A / floor) must reproduce the
v10 path exactly, so it is left unchanged (no D-7). User messages that match the
compound query phrases (cms_session._is_query_turn rules) introduce no new facts
and only pollute the CD, so the gemma manager skips them.

Robustness (gemma runs are long -- ~664 turns):
  * per-turn try/except: one bad turn logs an error and is skipped, the run
    does not die at turn 500.
  * incremental checkpoint every --ckpt-interval turns (default 50): the partial
    CD is written to <out>.ckpt as {nodes, summary, resume: {turn, session_idx,
    pairing, code_version}}. A checkpoint whose `pairing` differs from the
    current run's is REFUSED (F3): `turn` counts round trips in the spec arm
    and messages elsewhere, so resuming across the two skips the wrong turns.
  * --resume-from <ckpt>: resume a crashed run from a checkpoint.

CLI:
    # Arm A (floor, default):
    python -m benchmark.bineval.build_cd_offline \
        --chat benchmark/longchat/restaurant_chat_v2.json \
        --out benchmark/bineval/results/cd/restaurant_cd.json

    # Arm B (spec-compliant Gemma manager):
    python -m benchmark.bineval.build_cd_offline \
        --chat benchmark/longchat/restaurant_chat_v2.json \
        --extractor gemma \
        --out benchmark/bineval/results/cd/restaurant_cd_gemma.json

    # Smoke: first 2 sessions only (either arm):
    python -m benchmark.bineval.build_cd_offline \
        --chat benchmark/longchat/restaurant_chat_v2.json \
        --extractor gemma --max-sessions 2 \
        --out benchmark/bineval/results/cd/restaurant_cd_gemma_smoke.json

Output JSON: {"nodes": [{text, level, mass, parent_id, created_turn, node_id}],
"summary": {sun, planet, satellite, total, turns, unbudgeted_tokens}}.
The SBERT path is deterministic. All file I/O is utf-8; console prints are
ASCII-only (cp932-safe).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import tiktoken

from communication.cd_serializer import CDSerializer
from management.graph_builder import GraphBuilder
from management.harness.chunking import is_query_turn_en
from management.graph_merger import GraphMerger
from management.node_classifier import NodeClassifier
from management.text_chunker import TextChunker
from models.correlation_diagram import CorrelationDiagram
from models.node import NodeLevel

# ── D-7: query-turn skip (verbatim from cms_session._is_query_turn,
#         cms_session.py:324-352). Copied here rather than imported so this
#         module stays torch-free (server.cms_session imports the model at
#         module top). Keep in sync with cms_session._QUERY_PHRASES. ──────────
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


# H5: vLLM + Qwen3.x emit a <think> trace by default. It eats the token budget
# and the yes/no answer never arrives, so thinking is OFF unless the caller says
# otherwise (--judge-extra-body null sends nothing at all).
DEFAULT_JUDGE_EXTRA_BODY = '{"chat_template_kwargs": {"enable_thinking": false}}'

# Only the Anthropic backend has a meaningful default model id.  The anthropic
# and openai CLI choices were REMOVED (SPEC_FAITHFUL_DESIGN.md directive 5: no
# external API).  The backend classes stay for their unit tests.
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"

# --judge local: the manager's judge is the SAME open-weight model as the reader,
# served by vLLM on the same GPU (directive 5).  127.0.0.1 by default because
# that is where the Modal phase_manager starts it.
DEFAULT_LOCAL_JUDGE_BASE_URL = "http://127.0.0.1:8000"

# Extractors whose per-turn work is done by a management/harness manager object.
MANAGER_EXTRACTORS = ("harness", "spec")

# F3: how the driver groups the chat into manager updates. The spec arm updates
# ONCE PER user->assistant ROUND TRIP (0036); every other arm updates once per
# MESSAGE. A checkpoint written under one mode counts turns on a different axis,
# so resuming across modes silently fast-forwards past the wrong turns.
PAIRING_ROUND_TRIP = "round_trip"
PAIRING_MESSAGE = "message"


def pairing_mode(pair_round_trips: bool) -> str:
    return PAIRING_ROUND_TRIP if pair_round_trips else PAIRING_MESSAGE


def git_sha(default: str = "unknown") -> str:
    """Short git sha of the working tree, or ``default``.

    Stamped into every checkpoint so a resume across a code change is at least
    VISIBLE in the artifact. Never raises: no git, no repo and a git that is not
    on PATH all return the default.
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
    except Exception:  # noqa: BLE001 -- 'never raises' means never
        return default
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else default


def _is_query_turn(user_text: str) -> bool:
    """Mirror cms_session._is_query_turn (cms_session.py:336-352) exactly.

    Query turns introduce no new facts; feeding them to the manager only
    pollutes the CD. Heuristic: short question-form text. Note that bare
    "ください" is deliberately NOT matched (fact-introducing sentences like
    "確実に記録してください" also contain it) -- only the compound
    query-specific phrases are used (D-7).
    """
    text = user_text.strip()
    if len(text) > 300:
        return False
    if text.endswith("？") or text.endswith("?"):
        return True
    return any(phrase in text for phrase in _QUERY_PHRASES)


def make_token_counter():
    """tiktoken cl100k_base token counter (API-model arm convention)."""
    enc = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(enc.encode(text))


def _safe_extractor_fn(raw_fn):
    """Wrap the raw extractor with the same defensive filtering cms_session uses
    (cms_session.py:426-433): keep only list[dict] entries carrying 'text',
    fall back to a single satellite node on failure. Keeps the offline path
    behaviourally identical to the online _llm_extract_fn injection route.
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


# ── Arm B: spec-compliant Gemma extractor ──────────────────────────────────


def make_gemma_extractor_fn(model_id: str = "google/gemma-3-4b-it"):
    """Load google/gemma-3-4b-it in 4-bit via plain transformers and return an
    extract fn that replicates cms_session._llm_extract_fn's Gemma path
    (cms_session.py:435-451) exactly.

    This is the EXTRACTION model only: no generation model, no MassWeightedGemma,
    no attention patch, no mass_weighted_* import. D-5 requires the 4B model
    (not 1B).

    The extract fn:
      * builds the prompt with server.cd_parser.extract_nodes_prompt (the SAME
        text the online path uses; cd_parser is torch-free at import time);
      * greedy-decodes JSON, parses with the SAME fallback rules as
        cms_session (find('['), rfind(']')+1, json.loads, keep list[dict] with
        'text'), else a single satellite node.

    Returns: (extract_fn, unload_fn). unload_fn frees the model + CUDA cache.
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

    # Extraction should be deterministic: greedy decode (do_sample=False).
    # Budget mirrors server config (max_new_tokens 256 on the 6GB card).
    max_new_tokens = 256

    def extract_fn(text: str) -> list[dict]:
        # Mirror cms_session._llm_extract_fn (Gemma path), cms_session.py:435-451.
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
        # フォールバック: テキスト全体を satellite ノードとして扱う
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
    """Yield (session_idx, user_text, assistant_text) per ROUND TRIP (spec 0036:
    the diagram is updated once per user->assistant exchange).

    A user message is paired with the assistant message that follows it in the
    same session. An unpaired message (session ends after a user turn, or an
    assistant turn without a preceding user turn) is yielded alone with the
    other side empty. Turn numbering in build_cd is then the round-trip index.
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
    """Yield (session_idx, role, content) over all sessions' messages in order.

    Turn numbering is the global sequential index of the yielded messages
    (WO-1 canon). session_idx lets --max-sessions truncate and lets the
    checkpoint record how far we got.
    """
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
    """Drive the management path turn-by-turn. Returns (cd, n_turns, failed_turns).

    n_turns is the total number of {role, content} messages processed
    (sequential turn numbering over all sessions).

    apply_d7:      skip management on query-form user messages (D-7).
    max_sessions:  process only the first N sessions (smoke).
    ckpt_path:     if set, write a partial-CD checkpoint every ckpt_interval turns.
    manager:       harness manager (management.harness.HarnessManager). When
                   given, the chunker/classifier/merger block is REPLACED by
                   manager.update(base, user_text, assistant_text, turn); the
                   extractor_fn argument is then unused (pass None).
    resume_state:  {"turn": int, "session_idx": int} from a prior checkpoint;
                   messages with a global turn index < resume_state["turn"] are
                   skipped (the base CD must be reloaded by the caller).

    Robustness: each turn's management is wrapped in try/except so a single bad
    turn logs and is skipped rather than killing a 664-turn run. H4: the failures
    are COUNTED, and `max_consecutive_failures` consecutive failures abort the run
    (SystemExit after a checkpoint) -- an unreachable judge used to produce a
    silently empty diagram after 664 logged warnings.
    """
    chunker = TextChunker()
    classifier = NodeClassifier()
    builder = GraphBuilder()
    merger = GraphMerger()
    # F3: stamped into every checkpoint this run writes.
    pairing = pairing_mode(pair_round_trips)
    code_version = git_sha()

    base = resume_state["cd"] if resume_state and "cd" in resume_state else CorrelationDiagram()
    resume_turn = resume_state["turn"] if resume_state else 0

    turn = 0
    last_session_idx = 0
    failed_turns = 0
    consecutive_failures = 0
    if pair_round_trips:
        # Spec 0036: one manager update per user->assistant round trip.
        units = ((s, "pair", (u, a)) for s, u, a in _round_trip_iter(chat))
    else:
        units = ((s, role, content) for s, role, content in _msg_iter(chat))

    for s_idx, role, content in units:
        if max_sessions is not None and s_idx >= max_sessions:
            break
        last_session_idx = s_idx

        # Resume: fast-forward past already-processed turns.
        if turn < resume_turn:
            turn += 1
            continue

        # D-7: skip management on query-form user messages (both arms).
        # H3c: the corpus is English, so the harness arm ALSO applies the English
        # query-turn rule. The gemma/sbert arms keep the Japanese rule only, so
        # their published numbers stay reproducible.
        if role == "pair":
            user_text, assistant_text = content
            # D-7 on the round trip: a query-form user message and its answer
            # (which restates known facts) are both skipped.
            if apply_d7 and user_text and (
                _is_query_turn(user_text)
                or (manager is not None and is_query_turn_en(user_text))
            ):
                turn += 1
                continue
        else:
            if apply_d7 and role == "user":
                if _is_query_turn(content) or (
                    manager is not None and is_query_turn_en(content)
                ):
                    turn += 1
                    continue
            user_text = "" if role == "assistant" else content
            assistant_text = content if role == "assistant" else ""

        try:

            if manager is not None:
                # Harness arm: the manager owns chunking, extraction,
                # classification, provisional linking, merging and normalize.
                manager.update(base, user_text, assistant_text, turn)
            else:
                chunks = chunker.chunk_turn(user_text, assistant_text, turn)

                provisional = CorrelationDiagram()
                for chunk in chunks:
                    classifier.classify(chunk, provisional, extractor_fn, builder=builder)
                merger.merge(base, provisional)
            consecutive_failures = 0
        except Exception as e:  # noqa: BLE001 -- a bad turn must not kill the run
            # ASCII-safe error log (cp932-safe console).
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
    """Write a partial-CD checkpoint. Same node schema as the final output,
    plus a resume block. Atomic-ish: write to .tmp then replace.

    H12: the harness manager's cumulative counters (call totals + quality
    counters) are persisted in the resume block and restored by --resume-from,
    so a resumed run reports the whole run's cost. The judge ANSWER CACHE is
    deliberately NOT persisted: it can hold tens of thousands of entries, it is
    only an economy, and a resumed run simply re-asks what it needs.

    F3: ``resume.pairing`` and ``resume.code_version`` identify WHAT produced
    this checkpoint. ``turn`` counts round trips in the spec arm and messages
    everywhere else, so a checkpoint resumed under the other mode fast-forwards
    past a different set of turns and produces a diagram that matches neither
    run. ``_load_checkpoint`` refuses that combination outright.
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
    """Rebuild a CorrelationDiagram from a checkpoint's node records and return
    {"cd": cd, "turn": int}. Reconstructs the sun/planet/satellite hierarchy
    from parent_id linkage.

    F3: when ``expected_pairing`` is given, a checkpoint whose ``resume.pairing``
    differs is REFUSED (SystemExit naming both values). A checkpoint written
    before this field existed reports ``unknown`` and is refused as well -- the
    whole point is that stale state on a persistent volume must not be reused
    silently.
    """
    from models.node import Node

    with ckpt_path.open(encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("nodes", [])
    resume_block = data.get("resume", {})
    resume_turn = resume_block.get("turn", 0)
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

    cd = CorrelationDiagram()
    by_id: dict[str, Node] = {}
    # First pass: instantiate all nodes.
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
    # Second pass: attach in traversal order (suns, then planets, then sats).
    # CorrelationDiagram.add_* signatures put the node first, then the parent id
    # (add_planet(node, sun_id), add_satellite(node, planet_id)).
    for r in records:
        node = by_id[r["node_id"]]
        if node.level == NodeLevel.SUN:
            cd.add_sun(node)
    for r in records:
        node = by_id[r["node_id"]]
        if node.level == NodeLevel.PLANET and node.parent_id in by_id:
            cd.add_planet(node, node.parent_id)
    for r in records:
        node = by_id[r["node_id"]]
        if node.level == NodeLevel.SATELLITE and node.parent_id in by_id:
            cd.add_satellite(node, node.parent_id)

    return {
        "cd": cd,
        "turn": resume_turn,
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
    """Build the `--judge local` backend (S1.3): an OpenAI-compatible server,
    e.g. vLLM serving the SAME open-weight model as the reader (directive 5).
    """
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

    # Resume state (rebuild base CD + turn cursor from a checkpoint).
    # F3: the pairing mode this run will use, decided BEFORE the checkpoint is
    # read so a stale checkpoint can be refused rather than silently reused.
    run_pairing = pairing_mode(args.extractor == "spec")
    resume_state: dict | None = None
    if args.resume_from:
        resume_state = _load_checkpoint(
            Path(args.resume_from), expected_pairing=run_pairing
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
            # H10 / S1.3: the fake judge cannot rank anything, so the embedding
            # shortlist would only load an SBERT model for nothing.
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
    # D-7 (compound query-phrase skip) is a D-ledger requirement for the
    # spec-compliant Gemma manager (RESEARCH_PROGRAM WO-2 arm B) ONLY. The SBERT
    # floor arm (arm A) must reproduce the v10 path exactly, so D-7 is NOT
    # applied there -- adding it would change the arm A / SBERT behavior.
    apply_d7 = args.extractor in ("gemma",) + MANAGER_EXTRACTORS
    print("building CD over all turns (extractor=%s, D-7=%s)..." % (args.extractor, apply_d7))
    _t0 = time.perf_counter()
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
            # Spec 0036: the spec manager updates once per round trip.
            pair_round_trips=(args.extractor == "spec"),
        )
    finally:
        if unload_fn is not None:
            unload_fn()
    _elapsed = time.perf_counter() - _t0

    # S1.3: the spec arm's summary/smoke output uses the marker format
    # ([SN] / [PN mass] / [RN]); mass appears on PLANET lines only (0062 / 0079).
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
    per_turn = (_elapsed / n_turns) if n_turns else 0.0
    print("wall time: %.1fs (%.3fs/turn over %d turns)" % (_elapsed, per_turn, n_turns))
    print("wrote: %s" % out_path.as_posix())

    run_smoke(cd, serializer, token_counter)


if __name__ == "__main__":
    main()
