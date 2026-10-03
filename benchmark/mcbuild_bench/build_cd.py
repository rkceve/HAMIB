"""Manager phase: build the correlation diagram (CD) over the corpus.

This is the first stage of the pipeline.  The CD written here is what
windows.build_window serializes into the proposed arm's window, which run_arms
hands to the reader.

Each round trip of the corpus goes through ``SpecManager.update()`` once: the
user text is the human message, the assistant text is the round trip's events
in order, each prefixed by its kind tag.  The manager's decisions are answered
by Jev (through the hooks in jev_judge) and node texts are written by the local
summarizer (summarizer_client).

    python -m benchmark.mcbuild_bench.build_cd \
        --session benchmark/mcbuild_bench/data/session_redacted.json \
        --out benchmark/mcbuild_bench/data/cd.json \
        --jev-accounting benchmark/mcbuild_bench/data/jev_calls.jsonl \
        --summarizer-accounting benchmark/mcbuild_bench/data/summarizer_calls.jsonl \
        --summarizer-url http://127.0.0.1:8001/v1 \
        --gpu-csv benchmark/mcbuild_bench/data/gpu_manager.csv \
        --max-jev-requests N --max-jev-input-tokens T [--max-round-trips N] [--exclude-rt 36]

    python -m benchmark.mcbuild_bench.build_cd --project --session <session.json> [--exclude-rt 36]

``--project`` sizes a run without calling Jev: it prints the chunk count per
round trip, the worst-case request count and an expected-spend estimate.

Jev costs money, so ``--max-jev-requests`` and ``--max-jev-input-tokens`` are
required.  ``BudgetedJev`` stops the run (``JevStop``) before the request that
would exceed either budget.

A stopped run loses nothing:

- Each round trip is a transaction: on any exception the diagram is rolled
  back to its state before that round trip, so the partial CD holds exactly
  the completed round trips and ``summary.turns`` is their count.
- The CD on disk is rewritten atomically after every round trip.
- ``JevStop`` / ``SummarizerStop`` (the expected stop conditions) write the
  partial CD with a ``stopped`` block and exit 1; any other exception writes
  the same partial CD and is re-raised.
- ``--resume <partial_cd.json>`` reloads the partial's nodes and skips the
  round trips it already processed.  The partial must come from the same
  corpus (same ``manifest.session_sha256``).

Every run has a run id (default: UTC timestamp + 6 hex chars) that is inserted
into the names of the Jev / summarizer accounting files and the GPU CSV, and a
run refuses to start when any of those files exists, so two runs never append
to the same file.  The manifest totals are summed from THIS run's accounting
files.  On resume the manifest also carries ``resumed_from`` and
``prior_manifest`` (re-summed from the prior run's accounting files when they
still exist) so the cost of a resumed chain can be added up;
``summary.turns`` and ``summary.dropped_chunks`` are cumulative over the
chain, while ``harness_calls`` / ``harness_quality`` cover this run only.  The
manifest also records the diagram's capacity limits (config.yaml
``graph.max_*``); the merger raises at those limits instead of dropping a node.

``build()`` only needs a configured SpecManager, so the tests drive it with
fakes; the real clients are imported inside ``main()``.
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
from collections.abc import Iterator
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
PUBLISHED_PRICE_PER_MTOK = 0.042  # USD per M input tokens (published price; recorded, not measured)
PROJECT_KEPT_FRACTION = 0.5  # kept fraction assumed by the --project spend ESTIMATE
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
    """Node counts and manager totals.  On resume, ``turns`` / ``dropped_chunks``
    add the partial's counts, so ``turns`` is always the number of round trips
    processed so far."""
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
    """The ``stopped`` block of a partial CD: why the run ended early."""
    return {"reason": repr(exc), "type": type(exc).__name__}


# -- spending guard ------------------------------------------------------------------


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


# -- request projection (no Jev) ---------------------------------------------------------


def project_main(session_path: Path, exclude_idx=DEFAULT_EXCLUDE_RT) -> int:
    """Print the chunk count per round trip (``SpecManager.chunk`` with the
    SpecConfig defaults, exactly as ``build`` chunks) and the worst-case Jev
    request totals of :func:`jev_judge.project_requests` over the corpus, then
    the expected spend of :func:`jev_judge.estimate_input_tokens`.

    The diagram before round trip i is bounded by the cumulative chunk count
    (every chunk kept, one node each): suns, planets and satellites are each
    <= the chunks so far.  The classification floor (one request per chunk) is
    the minimum any run costs.
    """
    corpus = load_corpus(session_path, exclude_idx)
    config = SpecConfig(shortlist_k=0)
    manager = SpecManager(JevJudgeLLM(), config=config)
    counts: list[tuple[int, int]] = []
    for round_trip in corpus.round_trips:
        user_text, assistant_text = round_trip_texts(round_trip)
        turn = int(round_trip["idx"])
        counts.append((turn, len(manager.chunk(user_text, assistant_text, turn))))
    total_chunks = sum(n for _, n in counts)
    print(
        "projection (no Jev): session=%s exclude_rt=%s round_trips=%d chunks=%d chunk_max_chars=%d"
        % (Path(session_path).as_posix(), json.dumps(list(corpus.exclude_rt)), len(counts),
           total_chunks, config.chunk_max_chars)
    )
    seen = 0  # chunks before this round trip = upper bound of every node count
    worst_total = 0
    for turn, n_chunks in counts:
        worst = project_requests(n_chunks, seen, seen, n_satellites=seen)
        worst_total += worst
        print(
            "rt %02d: chunks=%d cd_before(suns<=%d planets<=%d satellites<=%d) worst=%d"
            % (turn, n_chunks, seen, seen, seen, worst)
        )
        seen += n_chunks
    print("classification floor (1 request per chunk): %d" % total_chunks)
    print("worst-case total requests: %d" % worst_total)
    print(
        "assumptions: every chunk kept (1 node each); diagram sizes before a round trip "
        "bounded by the cumulative chunk count; "
        "formula = jev_judge.project_requests (see its docstring; H19 = b, H22 (a), H23)"
    )
    # Expected spend: an ESTIMATE to read before a run, not the budget guard.
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
    ``start_idx`` (resume) skips the first ``start_idx`` round trips of the
    list (= ``summary.turns``, the count already processed; list positions
    rather than ``idx`` values, so an excluded round trip cannot shift the
    restart point); ``max_round_trips`` applies to the list before that skip.
    """
    cd = cd if cd is not None else CorrelationDiagram()
    reports = reports if reports is not None else []
    round_trips = list(session["round_trips"])
    if max_round_trips is not None:
        round_trips = round_trips[: max(0, int(max_round_trips))]
    for round_trip in round_trips[max(0, int(start_idx)):]:
        turn = int(round_trip["idx"])
        user_text, assistant_text = round_trip_texts(round_trip)
        snapshot = cd.clone()
        try:
            report = manager.update(cd, user_text, assistant_text, turn=turn)
        except BaseException:
            # Roll the SAME object back (make_manager's sun_texts_fn holds a
            # reference to it): the partial the caller writes from ``cd`` then
            # holds only completed round trips.
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
    """SpecManager whose decisions go to Jev and whose node texts come from the
    summarizer.

    ``cd`` is the diagram this manager will build (the object later passed to
    ``manager.update`` / ``build(cd=...)``).  Its current sun texts let a
    K_BELONGS decision over suns use the `sun` Choice wording; without ``cd``
    every such decision uses the `planet` wording.
    """
    counters = counters if counters is not None else JevCounters()
    sun_texts_fn: Callable[[], set[str]] | None = None
    if cd is not None:
        sun_texts_fn = lambda: {se.sun.text for se in cd.suns}  # noqa: E731 -- closes over cd
    # keep_fn and node_fn share ONE Jev request per chunk (keep + the three
    # axes): JevKeepFn sends it and JevNodeFn reads the cached answer.  The
    # cache holds a single chunk, which is only correct when the manager works
    # through chunks one at a time.
    config = SpecConfig(shortlist_k=0)
    assert config.max_workers == 1, "H1 one-request cache requires the sequential manager"
    node_fn = JevNodeFn(jev, summarizer, counters)
    return SpecManager(
        JevJudgeLLM(),
        # The spec manager's defaults (chunk 400 chars, node 120 chars, planet
        # mass floor 0.0, sequential); shortlist_k=0 because the Jev similarity judge
        # has no embedding shortlist.
        config=config,
        node_fn=node_fn,
        keep_fn=JevKeepFn(node_fn),
        similarity=JevSimilarityJudge(jev, counters, sun_texts_fn=sun_texts_fn),
    )


# -- resume ------------------------------------------------------------------------------


def session_sha256(path: Path) -> str:
    """sha256 of the session FILE bytes.  Recorded as ``session_file_sha256``
    only; artifacts are bound to the corpus hash (``corpus.sha256``)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_run_id() -> str:
    """UTC timestamp + 6 hex chars, e.g. ``20260918T101010Z-3f9a1c``."""
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


def _jsonl_records(path: Path) -> Iterator[dict]:
    """The records of a JSONL file (blank lines skipped; none if it is missing)."""
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarizer_totals(accounting_path: Path) -> dict[str, int]:
    """Totals over the summarizer accounting lines
    (``{"ts", "prompt_tokens", "completion_tokens", "latency_ms", "retried", "chars"}``)."""
    totals = {"summarizer_calls": 0, "summarizer_prompt_tokens": 0, "summarizer_completion_tokens": 0}
    for rec in _jsonl_records(accounting_path):
        totals["summarizer_calls"] += 1
        totals["summarizer_prompt_tokens"] += int(rec.get("prompt_tokens", 0) or 0)
        totals["summarizer_completion_tokens"] += int(rec.get("completion_tokens", 0) or 0)
    return totals


def jev_totals(accounting_path: Path) -> dict[str, int | float]:
    """Totals over the Jev accounting lines (``JevClient._record``, one per HTTP
    attempt): ``jev_requests`` = number of lines; the token totals sum the lines
    that carry usage; cost = input tokens x the published price."""
    records = list(_jsonl_records(accounting_path))
    input_tokens = sum(int(r["input_tokens"]) for r in records if r.get("input_tokens") is not None)
    output_tokens = sum(int(r["output_tokens"]) for r in records if r.get("output_tokens") is not None)
    return {
        "jev_requests": len(records),
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
        **jev_totals(jev_accounting),
        "published_price_per_mtok": PUBLISHED_PRICE_PER_MTOK,
        **summarizer_totals(summarizer_accounting),
        "wall_s": wall_s,
        "gpu_csv": gpu_csv,
        "session_sha256": session_sha,  # hash of the corpus after the exclusion
        "session_file_sha256": session_file_sha,
        "exclude_rt": list(exclude_rt),
        "n_round_trips": n_round_trips,
        "run_id": run_id,
        "jev_accounting": str(jev_accounting),
        "summarizer_accounting": str(summarizer_accounting),
    }
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
    """Atomic write: tmp file next to ``path`` + ``os.replace``, so a reader
    (or a crash mid-write) never sees a half-written CD."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _ascii(text: str) -> str:
    """``text`` with non-ASCII characters replaced (a Windows console may not encode them)."""
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
    ap.add_argument("--max-round-trips", type=int, default=None,
                    help="process only the first N round trips (for short trial runs)")
    ap.add_argument("--resume", default=None, metavar="PARTIAL_CD_JSON",
                    help="continue from a partial CD written by an earlier run (same session sha256)")
    ap.add_argument("--run-id", default=None,
                    help="tag appended before the extension of both accounting files and "
                         "the GPU CSV (default: UTC %%Y%%m%%dT%%H%%M%%SZ-<6 hex>)")
    ap.add_argument("--exclude-rt", default=DEFAULT_EXCLUDE_RT_CLI,
                    help="comma-separated round-trip indices to drop from the corpus (the "
                         "default is the session retrospective); each must exist in the "
                         "session; 'none' disables (default: %(default)s)")
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

    # Imported here so that build() stays testable without the real clients.
    from benchmark.mcbuild_bench.gpu_sampler import GpuSampler
    from benchmark.mcbuild_bench.jev_client import JevClient
    from benchmark.mcbuild_bench.summarizer_client import SummarizerClient

    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        print("TYPESAFE_API_KEY is not set", file=sys.stderr)
        return 2

    session_path = Path(args.session)
    try:
        corpus = load_corpus(session_path, exclude_rt)
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

    # Resume: the partial's nodes become the starting diagram; the round trips
    # it already processed (summary.turns) are skipped.
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
        prior_manifest, prior_reconciled = reconcile_prior_manifest(prior_manifest)
        print("[resume] %s: %d nodes, continuing at round trip idx %d (prior manifest %s)"
              % (args.resume, len(cd), start_idx,
                 "reconciled from its accounting files" if prior_reconciled
                 else "kept as stored: accounting files not found"))

    # Log the diagram's capacity limits before the run (also kept in the manifest).
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

    out_path = Path(args.out)
    reports: list[SpecTurnReport] = []

    def progress(turn: int, report: SpecTurnReport) -> None:
        print(
            "[rt %d] chunks=%d dropped=%d nodes=%d added=%d calls=%d cd_nodes=%d"
            % (turn, report.chunks, report.dropped, report.nodes, report.added,
               report.total_calls(), len(cd))
        )
        # Rewrite the CD after every round trip, so a crash at any point leaves
        # every completed round trip on disk.
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
        # The expected stop conditions: write the partial CD, exit 1.
        stopped = stopped_of(e)
        payload = payload_of(cd, reports, manager, stopped=stopped, prior_summary=prior_summary)
    except BaseException as e:
        # Anything else still leaves the partial CD on disk, then re-raises.
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
