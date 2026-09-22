"""Build the correlation diagram over the redacted session (DESIGN.md 6 build_cd).

One ``SpecManager.update()`` per ROUND TRIP (A2 / 0036): user_text = the human
message, assistant_text = the events in order, each prefixed by its kind tag.
Judge = Jev through the hooks of ``jev_judge``; node text = local summarizer.

    python -m benchmark.mcbuild_bench.build_cd \
        --session benchmark/mcbuild_bench/data/session_redacted.json \
        --out benchmark/mcbuild_bench/data/cd.json \
        --jev-accounting benchmark/mcbuild_bench/data/jev_calls.jsonl \
        --summarizer-accounting benchmark/mcbuild_bench/data/summarizer_calls.jsonl \
        --summarizer-url http://127.0.0.1:8001/v1 \
        --gpu-csv benchmark/mcbuild_bench/data/gpu_manager.csv \
        --max-jev-requests N --max-jev-input-tokens T [--max-round-trips N] [--exclude-rt 36]

    python -m benchmark.mcbuild_bench.build_cd --project --session <session.json> [--exclude-rt 36]

Corpus (H22 (c)): the session is loaded through ``corpus.load_corpus`` — the
round trips named by ``--exclude-rt`` (default ``36``, the retrospective;
``none`` disables) are dropped by a checked filter, and ``manifest.session_sha256``
is the sha256 of the FILTERED content (``session_file_sha256`` keeps the raw
file's hash; ``exclude_rt`` and ``n_round_trips`` are recorded).  run_arms binds
its cells to the same corpus hash.

Budget guard (Astra round 3 item 1): ``--max-jev-requests`` / ``--max-jev-input-tokens``
are REQUIRED.  ``BudgetedJev`` wraps the client and raises ``JevStop("budget
exceeded: ...")`` BEFORE sending the request that would exceed either budget
(requests: the next request would be number max+1; input tokens: the tokens
already consumed reached the budget, so the next request can only exceed it --
the token budget can therefore be overshot by at most one request).  The
partial CD is written through the existing transactional path; the manifest
records ``jev_budget`` (budgets + counts).  ``--project`` prints the chunk
count per round trip (``SpecManager.chunk`` with the SpecConfig defaults, no
Jev) and the worst-case request totals of ``jev_judge.project_requests``, then the
expected-spend ESTIMATE of ``jev_judge.estimate_input_tokens`` at kept fraction 0.5 (H23).

Manifest totals (item 4): ``jev_requests`` / ``jev_input_tokens_total`` /
``jev_output_tokens_total`` / ``jev_cost_usd`` are summed over THIS run's Jev
accounting JSONL (one line per HTTP attempt; tokens over the lines with usage),
exactly like the summarizer totals; ``JevCounters`` only feed ``harness_calls``.
On ``--resume`` the ``prior_manifest`` is re-summed from the accounting files it
names when both exist (``prior_manifest_reconciled: true``), otherwise its
stored values are kept and ``prior_manifest_reconciled: false`` is recorded.
The manifest also records the diagram ``capacities`` (config.yaml ``graph.max_*``,
item 2; the merger raises instead of dropping a node at those limits).

``build()`` is pure with respect to the clients (it only needs a configured
SpecManager), so the tests drive it with fakes.  A JevStop / SummarizerStop
propagates out of ``build()``; ``main()`` then writes the partial diagram with
``"stopped": {"reason": repr(exc), "type": type(exc).__name__}`` to --out and
exits non-zero (D1 / D2).  ANY other exception writes the same partial output
and is re-raised (F3 / F12): no partial CD is ever lost.

Each round trip is TRANSACTIONAL: ``build()`` takes ``cd.clone()`` before
``manager.update`` and, on any exception, restores the pre-turn state into the
SAME ``cd`` object (``cd.suns = snapshot.suns``; the object is never rebound
because ``make_manager``'s ``sun_texts_fn`` closes over it).  The partial
therefore holds exactly the COMPLETED round trips and ``summary.turns`` equals
their count.  (The former "> 254 suns -> JevStop" rule is withdrawn: H22 (a)
asks the sun Choice in batches of <= 254.)

Resume (F2, DECISIONS H9): ``--resume <partial_cd.json>`` reloads ``nodes``
(``benchmark.bineval.arms.cd_from_records``), skips the first ``summary["turns"]``
round trips of the (filtered) corpus list (build() appends one report per round
trip processed, so ``turns`` is the count already processed; positions, not
``idx`` values, so an excluded middle round trip cannot shift the restart) and
requires the partial's ``manifest.session_sha256`` to equal the current corpus
sha256 (same session file AND same exclusion set).  Accounting files AND the GPU CSV carry ``--run-id`` before
their extension (``jev_calls.<run_id>.jsonl``, ``gpu_manager.<run_id>.csv``);
``main()`` refuses to start (SystemExit) when any of the three already exists,
so a reused run id can never append to an earlier run's files.  The default
run id is the UTC timestamp plus 6 hex chars.  The manifest totals cover THIS run's files only
and, on resume, carry ``resumed_from`` and ``prior_manifest`` so the report
reader sums the cost.  ``summary.turns`` and ``summary.dropped_chunks`` are
cumulative over the resumed chain; ``harness_calls`` / ``harness_quality`` are
this run's manager only.

The JevClient / SummarizerClient / GpuSampler modules are imported lazily inside
``main()`` with exactly the constructor signatures of DESIGN.md 6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from benchmark.bineval.arms import cd_from_records
from benchmark.bineval.build_cd_offline import _node_record, git_sha
from benchmark.mcbuild_bench.corpus import (
    DEFAULT_EXCLUDE_RT,
    DEFAULT_EXCLUDE_RT_CLI,
    load_corpus,
    parse_exclude_rt,
)
from benchmark.mcbuild_bench.errors import JevStop, SummarizerStop
from benchmark.mcbuild_bench.jev_judge import (
    JevCounters,
    JevJudgeLLM,
    JevKeepFn,
    JevNodeFn,
    JevSimilarityJudge,
    estimate_input_tokens,
    project_requests,
)
from management.harness.spec_manager import SpecConfig, SpecManager, SpecTurnReport
from models.correlation_diagram import CorrelationDiagram
from models.node import NodeLevel

JEV_MODEL = "jev-latest"
PUBLISHED_PRICE_PER_MTOK = 0.042  # USD per M input tokens (DESIGN.md 3; recorded, not asserted)
PROJECT_KEPT_FRACTION = 0.5  # H23: kept fraction assumed by the --project spend ESTIMATE
EVENT_KINDS = ("text", "tool_use", "tool_result", "harness_note")


# -- round trip -> manager texts -------------------------------------------------


def round_trip_texts(round_trip: dict) -> tuple[str, str]:
    """(user_text, assistant_text) of one round trip.

    assistant_text joins the events in order with "\\n", each prefixed by its
    kind tag on its own line, e.g. "[tool_use]\\n<text>".
    """
    user_text = str(round_trip.get("human", "") or "")
    parts = [
        f"[{event.get('kind', 'text')}]\n{event.get('text', '') or ''}"
        for event in round_trip.get("events", [])
    ]
    return user_text, "\n".join(parts)


# -- payload -----------------------------------------------------------------------


def cd_records(cd: CorrelationDiagram) -> list[dict]:
    """Node records in traversal order, same shape as build_cd_offline._node_record."""
    records: list[dict] = []
    for se in cd.suns:
        records.append(_node_record(se.sun))
        for pe in se.planets:
            records.append(_node_record(pe.planet))
            for sat in pe.satellites:
                records.append(_node_record(sat))
    return records


def summary_of(
    cd: CorrelationDiagram,
    reports: list[SpecTurnReport],
    manager: SpecManager,
    *,
    prior_summary: dict | None = None,
) -> dict:
    """``turns`` / ``dropped_chunks`` add the resumed partial's counts (F2) so
    that ``turns`` always equals the number of round trips processed so far."""
    counts = {level.value: 0 for level in NodeLevel}
    for node in cd.all_nodes():
        counts[node.level.value] += 1
    prior = prior_summary or {}
    return {
        "sun": counts[NodeLevel.SUN.value],
        "planet": counts[NodeLevel.PLANET.value],
        "satellite": counts[NodeLevel.SATELLITE.value],
        "total": sum(counts.values()),
        "turns": int(prior.get("turns", 0)) + len(reports),
        "dropped_chunks": int(prior.get("dropped_chunks", 0)) + sum(r.dropped for r in reports),
        "harness_calls": manager.call_totals(),
        "harness_quality": manager.quality_totals(),
    }


def payload_of(
    cd: CorrelationDiagram,
    reports: list[SpecTurnReport],
    manager: SpecManager,
    *,
    stopped: dict | None = None,
    prior_summary: dict | None = None,
) -> dict:
    payload: dict[str, Any] = {
        "nodes": cd_records(cd),
        "summary": summary_of(cd, reports, manager, prior_summary=prior_summary),
    }
    if stopped is not None:
        payload["stopped"] = stopped
    return payload


def stopped_of(exc: BaseException) -> dict:
    """The ``stopped`` block of a partial CD (F3 / F12)."""
    return {"reason": repr(exc), "type": type(exc).__name__}


# -- budget guard (Astra round 3 item 1) ------------------------------------------------


class BudgetedJev:
    """Counting wrapper around a Jev client (``ask(state, questions)``).

    ``requests_sent`` counts ``ask`` calls (429/529 retries inside one call are
    attempts, not requests); ``input_tokens`` sums ``usage.input_tokens`` of the
    completed calls.  Before EVERY request the wrapper checks both budgets and
    raises ``JevStop("budget exceeded: ...")`` instead of sending:

      * ``requests_sent + 1 > max_requests``
      * ``input_tokens >= max_input_tokens`` (the tokens already consumed
        reached the budget; the next request could only exceed it -- the
        budget can be overshot by at most one request, whose size is unknown
        before it is sent).
    """

    def __init__(self, jev: Any, *, max_requests: int, max_input_tokens: int) -> None:
        for name, value in (("max_requests", max_requests), ("max_input_tokens", max_input_tokens)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        self._jev = jev
        self.max_requests = max_requests
        self.max_input_tokens = max_input_tokens
        self.requests_sent = 0
        self.input_tokens = 0
        self._lock = threading.Lock()

    def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            if self.requests_sent + 1 > self.max_requests:
                raise JevStop(
                    "budget exceeded: requests %d > max %d" % (self.requests_sent + 1, self.max_requests)
                )
            if self.input_tokens >= self.max_input_tokens:
                raise JevStop(
                    "budget exceeded: input_tokens %d >= max %d" % (self.input_tokens, self.max_input_tokens)
                )
            self.requests_sent += 1
        result = self._jev.ask(state, questions)
        usage = result.get("usage") if isinstance(result, dict) else None
        if isinstance(usage, dict):
            with self._lock:
                self.input_tokens += int(usage.get("input_tokens", 0) or 0)
        return result

    def state(self) -> dict[str, int]:
        """The ``jev_budget`` block of the manifest."""
        with self._lock:
            return {
                "max_requests": self.max_requests,
                "max_input_tokens": self.max_input_tokens,
                "requests_sent": self.requests_sent,
                "input_tokens": self.input_tokens,
            }


# -- request projection (Astra round 3 item 1; no Jev) --------------------------------------


def project_main(session_path: Path, exclude_idx=DEFAULT_EXCLUDE_RT) -> int:
    """Print the chunk count per round trip (``SpecManager.chunk`` with the
    SpecConfig defaults, exactly as ``build`` chunks) and the worst-case Jev
    request totals of :func:`jev_judge.project_requests`, over the corpus
    (session minus ``exclude_idx``, H22 (c)).

    The diagram before round trip i is bounded by the cumulative chunk count
    (every chunk kept, one node each): suns, planets and satellites each <=
    chunks so far (no sun cap since H22 (a) batches the Choice).  The
    classification floor (one request per chunk) is the minimum any run costs.
    """
    corpus = load_corpus(session_path, exclude_idx)
    config = SpecConfig(shortlist_k=0)
    manager = SpecManager(JevJudgeLLM(), config=config)
    round_trips = corpus.round_trips
    counts: list[tuple[int, int]] = []
    for round_trip in round_trips:
        user_text, assistant_text = round_trip_texts(round_trip)
        turn = int(round_trip["idx"])
        counts.append((turn, len(manager.chunk(user_text, assistant_text, turn))))
    total_chunks = sum(n for _, n in counts)
    print(
        "projection (no Jev): session=%s exclude_rt=%s round_trips=%d chunks=%d chunk_max_chars=%d"
        % (Path(session_path).as_posix(), json.dumps(list(corpus.exclude_rt)), len(counts),
           total_chunks, config.chunk_max_chars)
    )
    seen = 0
    worst_total = 0
    for turn, n_chunks in counts:
        suns, planets, satellites = seen, seen, seen
        worst = project_requests(n_chunks, suns, planets, n_satellites=satellites)
        worst_total += worst
        print(
            "rt %02d: chunks=%d cd_before(suns<=%d planets<=%d satellites<=%d) worst=%d"
            % (turn, n_chunks, suns, planets, satellites, worst)
        )
        seen += n_chunks
    print("classification floor (1 request per chunk): %d" % total_chunks)
    print("worst-case total requests: %d" % worst_total)
    print(
        "assumptions: every chunk kept (1 node each); diagram sizes before a round trip "
        "bounded by the cumulative chunk count; "
        "formula = jev_judge.project_requests (see its docstring; H19 = b, H22 (a), H23)"
    )
    # H23: expected spend (an ESTIMATE, not the budget guard) at kept_fraction 0.5.
    est = estimate_input_tokens(total_chunks, PROJECT_KEPT_FRACTION, n_round_trips=max(1, len(counts)))
    terms = " ".join(
        "%s=%d req/%d tok" % (name, t["requests"], t["input_tokens"]) for name, t in est["terms"].items()
    )
    print(
        "ESTIMATE (jev_judge.estimate_input_tokens, kept_fraction=%.1f, kept_nodes=%d): %s"
        % (PROJECT_KEPT_FRACTION, est["kept_nodes"], terms)
    )
    print(
        "ESTIMATE total: requests=%d input_tokens=%d usd=%.4f at %.3f USD/M input tokens "
        "(assumptions: %s)"
        % (est["requests_total"], est["input_tokens_total"], est["usd"], est["price_per_mtok_usd"],
           json.dumps(est["assumptions"], sort_keys=True))
    )
    return 0


# -- the loop ------------------------------------------------------------------------


def build(
    session: dict,
    manager: SpecManager,
    *,
    max_round_trips: int | None = None,
    cd: CorrelationDiagram | None = None,
    reports: list[SpecTurnReport] | None = None,
    progress: Callable[[int, SpecTurnReport], None] | None = None,
    start_idx: int = 0,
    prior_summary: dict | None = None,
) -> dict:
    """Run every round trip through ``manager.update`` and return the payload
    ({"nodes", "summary"}).

    ``cd`` and ``reports`` may be passed in so that the caller still holds the
    partial state when a JevStop / SummarizerStop propagates out of here.
    ``start_idx`` (F2 resume) skips the first ``start_idx`` round trips of the
    list (= ``summary.turns``, the count already processed; positions rather
    than ``idx`` values because the corpus may exclude round trips, H22 (c));
    ``max_round_trips`` applies to the session list before that filter.
    """
    cd = cd if cd is not None else CorrelationDiagram()
    reports = reports if reports is not None else []
    round_trips = list(session["round_trips"])
    if max_round_trips is not None:
        round_trips = round_trips[: max(0, int(max_round_trips))]
    for round_trip in round_trips[max(0, int(start_idx)):]:
        turn = int(round_trip["idx"])
        user_text, assistant_text = round_trip_texts(round_trip)
        snapshot = cd.clone()  # transactional round trip (item 3)
        try:
            report = manager.update(cd, user_text, assistant_text, turn=turn)
        except BaseException:
            # Roll the SAME object back to the pre-turn state: the partial the
            # caller writes from ``cd`` then holds only completed round trips.
            cd.suns = snapshot.suns
            raise
        reports.append(report)
        if progress is not None:
            progress(turn, report)
    return payload_of(cd, reports, manager, prior_summary=prior_summary)


def make_manager(
    jev: Any,
    summarizer: Any,
    counters: JevCounters | None = None,
    *,
    cd: CorrelationDiagram | None = None,
) -> SpecManager:
    """SpecManager wired to Jev + summarizer exactly as DESIGN.md 6 prescribes.

    ``cd`` is the diagram this manager will build (the object later passed to
    ``manager.update`` / ``build(cd=...)``): its current sun texts route the
    K_BELONGS decisions over sun candidates to the `sun` Choice question (F1 /
    H8).  Without ``cd`` every K_BELONGS decision takes the `belongs` noul path.
    """
    counters = counters if counters is not None else JevCounters()
    sun_texts_fn: Callable[[], set[str]] | None = None
    if cd is not None:
        sun_texts_fn = lambda: {se.sun.text for se in cd.suns}  # noqa: E731 -- closes over cd
    # H1 (D3 one-request contract): keep_fn and node_fn share ONE Jev request per
    # chunk ({keep, 3 axes}); JevKeepFn reads the answer JevNodeFn caches.  The
    # cache is single-slot, which is only valid with the sequential manager
    # (SpecConfig.max_workers == 1, the D4 default asserted here).
    config = SpecConfig(shortlist_k=0)
    assert config.max_workers == 1, "H1 one-request cache requires the sequential manager"
    node_fn = JevNodeFn(jev, summarizer, counters)
    return SpecManager(
        JevJudgeLLM(),
        # D4: spec defaults (chunk 400 / node 120 / floor 0.0 / sequential);
        # shortlist_k=0 because the injected similarity has no embedding shortlist.
        config=config,
        node_fn=node_fn,
        keep_fn=JevKeepFn(node_fn),
        similarity=JevSimilarityJudge(jev, counters, sun_texts_fn=sun_texts_fn),
    )


# -- resume (F2) -------------------------------------------------------------------------


def session_sha256(path: Path) -> str:
    """sha256 of the session FILE bytes (recorded as ``session_file_sha256``;
    the binding hash ``session_sha256`` is the corpus hash, H22 (c))."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_run_id() -> str:
    """UTC timestamp + 6 hex chars, e.g. ``20260918T101010Z-3f9a1c`` (item 10)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)


def with_run_id(path: Path, run_id: str) -> Path:
    """``data/jev_calls.jsonl`` + ``r1`` -> ``data/jev_calls.r1.jsonl``."""
    path = Path(path)
    return path.with_name(f"{path.stem}.{run_id}{path.suffix}")


def resume_state(partial: dict, session_sha: str) -> tuple[CorrelationDiagram, int, dict]:
    """(cd, start_idx, prior_manifest) from a partial CD payload.

    Raises ValueError when the partial was built from a different corpus
    (``manifest.session_sha256`` missing or different: another session file or
    another exclusion set).
    """
    manifest = partial.get("manifest")
    if not isinstance(manifest, dict):
        manifest = {}
    prior_sha = manifest.get("session_sha256")
    if prior_sha != session_sha:
        raise ValueError(
            "resume refused: partial manifest.session_sha256=%r does not match the "
            "current corpus (session file + --exclude-rt) sha256 %s" % (prior_sha, session_sha)
        )
    cd = cd_from_records(partial["nodes"])
    start_idx = int(partial["summary"]["turns"])
    return cd, start_idx, manifest


# -- manifest -------------------------------------------------------------------------


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001 -- absent package on the local CPU machine
        return None


def summarizer_totals(accounting_path: Path) -> dict[str, int]:
    """Totals over the summarizer accounting lines
    ({"ts","prompt_tokens","completion_tokens","latency_ms","retried","chars"}, DESIGN.md 6)."""
    totals = {"summarizer_calls": 0, "summarizer_prompt_tokens": 0, "summarizer_completion_tokens": 0}
    if not accounting_path.exists():
        return totals
    with accounting_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            totals["summarizer_calls"] += 1
            totals["summarizer_prompt_tokens"] += int(rec.get("prompt_tokens", 0) or 0)
            totals["summarizer_completion_tokens"] += int(rec.get("completion_tokens", 0) or 0)
    return totals


def jev_totals(accounting_path: Path) -> dict[str, int | float]:
    """Totals over the Jev accounting lines (jev_client.JevClient._record, one
    per HTTP attempt): ``jev_requests`` = number of lines; the token totals
    sum the lines with non-null usage; cost per D5."""
    requests = 0
    input_tokens = 0
    output_tokens = 0
    if accounting_path.exists():
        with accounting_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                requests += 1
                if rec.get("input_tokens") is not None:
                    input_tokens += int(rec["input_tokens"])
                if rec.get("output_tokens") is not None:
                    output_tokens += int(rec["output_tokens"])
    return {
        "jev_requests": requests,
        "jev_input_tokens_total": input_tokens,
        "jev_output_tokens_total": output_tokens,
        "jev_cost_usd": input_tokens * PUBLISHED_PRICE_PER_MTOK / 1e6,
    }


def reconcile_prior_manifest(prior: dict) -> tuple[dict, bool]:
    """Re-sum a resumed run's manifest totals from the accounting files it
    names (``jev_accounting`` / ``summarizer_accounting``).  Both files must
    exist; otherwise the stored values are kept and False is returned."""
    jev_path = prior.get("jev_accounting")
    summarizer_path = prior.get("summarizer_accounting")
    if not (
        isinstance(jev_path, str) and Path(jev_path).exists()
        and isinstance(summarizer_path, str) and Path(summarizer_path).exists()
    ):
        return dict(prior), False
    reconciled = dict(prior)
    reconciled.update(jev_totals(Path(jev_path)))
    reconciled.update(summarizer_totals(Path(summarizer_path)))
    return reconciled, True


def manifest_of(
    *,
    wall_s: float,
    gpu_csv: str,
    summarizer_accounting: Path,
    session_sha: str,
    run_id: str,
    jev_accounting: Path,
    jev_budget: dict[str, int] | None = None,
    capacities: dict[str, int] | None = None,
    resumed_from: str | None = None,
    prior_manifest: dict | None = None,
    prior_manifest_reconciled: bool | None = None,
    exclude_rt: tuple[int, ...] = (),
    session_file_sha: str | None = None,
    n_round_trips: int | None = None,
) -> dict:
    """Totals cover THIS run only (its two accounting files); on resume
    ``resumed_from`` / ``prior_manifest`` let the reader sum the chain."""
    manifest: dict[str, Any] = {
        "git_sha": git_sha(),
        "transformers_version": _package_version("transformers"),
        "vllm_version": _package_version("vllm"),
        "jev_model": JEV_MODEL,
    }
    manifest.update(jev_totals(jev_accounting))
    manifest["published_price_per_mtok"] = PUBLISHED_PRICE_PER_MTOK
    manifest.update(summarizer_totals(summarizer_accounting))
    manifest["wall_s"] = wall_s
    manifest["gpu_csv"] = gpu_csv
    manifest["session_sha256"] = session_sha  # corpus hash (filtered content, H22 (c))
    manifest["session_file_sha256"] = session_file_sha
    manifest["exclude_rt"] = list(exclude_rt)
    manifest["n_round_trips"] = n_round_trips
    manifest["run_id"] = run_id
    manifest["jev_accounting"] = str(jev_accounting)
    manifest["summarizer_accounting"] = str(summarizer_accounting)
    if jev_budget is not None:
        manifest["jev_budget"] = dict(jev_budget)
    if capacities is not None:
        manifest["capacities"] = dict(capacities)
    if resumed_from is not None:
        manifest["resumed_from"] = resumed_from
        manifest["prior_manifest"] = prior_manifest if prior_manifest is not None else {}
        manifest["prior_manifest_reconciled"] = bool(prior_manifest_reconciled)
    return manifest


def write_json(path: Path, payload: dict) -> None:
    """Atomic write (item G): tmp file next to ``path`` + ``os.replace``, so a
    reader (or a crash mid-write) never sees a half-written CD."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _ascii(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


# -- CLI ------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the mcbuild-bench CD with Jev + local summarizer.")
    ap.add_argument("--session", required=True, help="data/session_redacted.json")
    ap.add_argument("--project", action="store_true",
                    help="print chunk counts per round trip and the worst-case Jev request "
                         "totals (no Jev, no other arguments needed), then exit")
    # Required unless --project (checked below; argparse cannot express that).
    ap.add_argument("--out", default=None, help="output CD JSON (data/cd.json)")
    ap.add_argument("--jev-accounting", default=None, help="data/jev_calls.jsonl")
    ap.add_argument("--summarizer-accounting", default=None, help="data/summarizer_calls.jsonl")
    ap.add_argument("--summarizer-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--gpu-csv", default=None, help="data/gpu_manager.csv")
    ap.add_argument("--max-jev-requests", type=int, default=None,
                    help="stop (JevStop, partial CD written) before the request that would be "
                         "number N+1 (required)")
    ap.add_argument("--max-jev-input-tokens", type=int, default=None,
                    help="stop before the next request once the Jev input tokens consumed reach T "
                         "(required)")
    ap.add_argument("--max-round-trips", type=int, default=None, help="process only the first N round trips (V4)")
    ap.add_argument("--resume", default=None, metavar="PARTIAL_CD_JSON",
                    help="continue from a partial CD written by an earlier run (same session sha256)")
    ap.add_argument("--run-id", default=None,
                    help="tag appended before the extension of both accounting files and "
                         "the GPU CSV (default: UTC %%Y%%m%%dT%%H%%M%%SZ-<6 hex>)")
    ap.add_argument("--exclude-rt", default=DEFAULT_EXCLUDE_RT_CLI,
                    help="H22 (c): comma-separated round-trip indices dropped from the corpus "
                         "(checked: they must exist); 'none' disables (default: %(default)s)")
    args = ap.parse_args(argv)

    try:
        exclude_rt = parse_exclude_rt(args.exclude_rt)
    except ValueError as e:
        ap.error(str(e))
    if args.project:
        try:
            return project_main(Path(args.session), exclude_rt)
        except ValueError as e:
            raise SystemExit(str(e)) from None
    required = ("out", "jev_accounting", "summarizer_accounting", "gpu_csv",
                "max_jev_requests", "max_jev_input_tokens")
    missing = ["--" + name.replace("_", "-") for name in required if getattr(args, name) is None]
    if missing:
        ap.error("the following arguments are required: " + ", ".join(missing))
    if args.max_jev_requests < 1 or args.max_jev_input_tokens < 1:
        ap.error("--max-jev-requests and --max-jev-input-tokens must be positive")

    # Other agents' modules (DESIGN.md 6 signatures), imported lazily so build()
    # stays testable without them.
    from benchmark.mcbuild_bench.gpu_sampler import GpuSampler
    from benchmark.mcbuild_bench.jev_client import JevClient
    from benchmark.mcbuild_bench.summarizer_client import SummarizerClient

    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        print("TYPESAFE_API_KEY is not set", file=sys.stderr)
        return 2

    session_path = Path(args.session)
    try:
        corpus = load_corpus(session_path, exclude_rt)  # H22 (c): checked filter
    except ValueError as e:
        raise SystemExit(str(e)) from None
    session_sha = corpus.sha256
    session = {"round_trips": corpus.round_trips}
    print("[corpus] %s: %d of %d round trips (exclude_rt=%s) sha256=%s" % (
        session_path.as_posix(), len(corpus.round_trips), corpus.n_session_round_trips,
        json.dumps(list(corpus.exclude_rt)), session_sha))

    run_id = args.run_id if args.run_id else default_run_id()
    jev_accounting = with_run_id(Path(args.jev_accounting), run_id)
    summarizer_accounting = with_run_id(Path(args.summarizer_accounting), run_id)
    gpu_csv = with_run_id(Path(args.gpu_csv), run_id)
    existing = [str(p) for p in (jev_accounting, summarizer_accounting, gpu_csv) if p.exists()]
    if existing:
        raise SystemExit(
            "refusing to start: run id %r already has output files (a reused run id "
            "would append to them): %s" % (run_id, ", ".join(existing))
        )

    # Resume (F2): the partial's nodes become the starting diagram; the round
    # trips already processed (summary.turns) are skipped.
    cd = CorrelationDiagram()
    start_idx = 0
    prior_summary: dict | None = None
    prior_manifest: dict | None = None
    prior_reconciled: bool | None = None
    if args.resume:
        with Path(args.resume).open(encoding="utf-8") as f:
            partial = json.load(f)
        try:
            cd, start_idx, prior_manifest = resume_state(partial, session_sha)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
        prior_summary = partial.get("summary") or {}
        # Item 4: the prior totals are re-summed from the prior accounting files.
        prior_manifest, prior_reconciled = reconcile_prior_manifest(prior_manifest)
        print("[resume] %s: %d nodes, continuing at round trip idx %d (prior manifest %s)"
              % (args.resume, len(cd), start_idx,
                 "reconciled from its accounting files" if prior_reconciled
                 else "kept as stored: accounting files not found"))

    # Item 2: the effective capacities (config.yaml graph.max_*) are logged and
    # recorded in the manifest before the run; the merger raises at those limits.
    capacities = cd.capacities()
    print("[capacities] %s" % " ".join("%s=%d" % kv for kv in capacities.items()))

    jev = BudgetedJev(
        JevClient(
            api_key=api_key,
            model=JEV_MODEL,
            timeout_s=60,
            max_attempts=3,
            accounting_path=jev_accounting,
        ),
        max_requests=args.max_jev_requests,
        max_input_tokens=args.max_jev_input_tokens,
    )
    summarizer = SummarizerClient(
        base_url=args.summarizer_url,
        model="summarizer",
        accounting_path=summarizer_accounting,
    )
    counters = JevCounters()
    manager = make_manager(jev, summarizer, counters, cd=cd)
    # (t0 below is rebound when the sampler starts; progress() reads it at call time.)

    out_path = Path(args.out)
    reports: list[SpecTurnReport] = []
    t0 = time.perf_counter()

    def progress(turn: int, report: SpecTurnReport) -> None:
        print(
            "[rt %d] chunks=%d dropped=%d nodes=%d added=%d calls=%d cd_nodes=%d"
            % (turn, report.chunks, report.dropped, report.nodes, report.added,
               report.total_calls(), len(cd))
        )
        # Item G: the CD on disk always holds every COMPLETED round trip
        # (atomic write), not only the final or the exception-path output.
        finish(payload_of(cd, reports, manager, prior_summary=prior_summary),
               time.perf_counter() - t0)

    def finish(payload: dict, wall_s: float) -> None:
        payload["manifest"] = manifest_of(
            wall_s=wall_s, gpu_csv=str(gpu_csv),
            summarizer_accounting=summarizer_accounting,
            session_sha=session_sha, run_id=run_id, jev_accounting=jev_accounting,
            jev_budget=jev.state(), capacities=capacities,
            resumed_from=args.resume, prior_manifest=prior_manifest,
            prior_manifest_reconciled=prior_reconciled,
            exclude_rt=corpus.exclude_rt, session_file_sha=corpus.session_file_sha256,
            n_round_trips=len(corpus.round_trips),
        )
        write_json(out_path, payload)

    sampler = GpuSampler(gpu_csv)
    sampler.start()
    t0 = time.perf_counter()  # wall clock starts with the sampler, not at CLI entry
    stopped: dict | None = None
    try:
        payload = build(
            session, manager, max_round_trips=args.max_round_trips,
            cd=cd, reports=reports, progress=progress,
            start_idx=start_idx, prior_summary=prior_summary,
        )
    except (JevStop, SummarizerStop) as e:
        # D1 / D2: the expected stop conditions -> partial output, exit 1.
        stopped = stopped_of(e)
        payload = payload_of(cd, reports, manager, stopped=stopped, prior_summary=prior_summary)
    except BaseException as e:
        # F3 / F12: anything else still leaves the partial CD on disk, then re-raises.
        sampler.stop()
        finish(
            payload_of(cd, reports, manager, stopped=stopped_of(e), prior_summary=prior_summary),
            time.perf_counter() - t0,
        )
        raise
    finally:
        wall_s = time.perf_counter() - t0
        sampler.stop()

    finish(payload, wall_s)

    s = payload["summary"]
    print("--- build summary ---")
    print("round trips: %d  dropped chunks: %d" % (s["turns"], s["dropped_chunks"]))
    print("nodes: sun=%d planet=%d satellite=%d total=%d" % (s["sun"], s["planet"], s["satellite"], s["total"]))
    print("harness calls: %s" % " ".join("%s=%d" % kv for kv in sorted(s["harness_calls"].items())))
    print("harness quality: %s" % " ".join("%s=%d" % kv for kv in sorted(s["harness_quality"].items())))
    m = payload["manifest"]
    print("jev requests=%d tokens: in=%d out=%d cost_usd=%.4f  wall=%.1fs" % (
        m["jev_requests"], m["jev_input_tokens_total"], m["jev_output_tokens_total"],
        m["jev_cost_usd"], m["wall_s"]))
    b = m["jev_budget"]
    print("jev budget: requests %d/%d input_tokens %d/%d" % (
        b["requests_sent"], b["max_requests"], b["input_tokens"], b["max_input_tokens"]))
    print("wrote: %s" % out_path.as_posix())
    if stopped is not None:
        print("STOPPED: %s" % _ascii(stopped["reason"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
