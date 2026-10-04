"""Build all_numbers.json (one flat record per cell) and raw_manifest.json for a results folder.

Same layout as results/a100_2026-09-20/all_numbers.json so the two rounds can be read side by
side; round-2 cells add inject / bias_cap and the marker counts the reader saw. Run:
    python collect_numbers.py --raw <dir with cell dirs> --scores scores.json --questions q.json
                              --out all_numbers.json [--date D] [--extra extra.json]
    python collect_numbers.py --manifest <results dir> --out raw_manifest.json --pod <id> [--archive-sha X]
"""
from __future__ import annotations
import argparse, hashlib, json, statistics, sys
from pathlib import Path


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def cell_record(cell_dir: Path, score: dict | None, kinds: dict[str, str]) -> dict:
    meta = json.loads((cell_dir / "meta.json").read_text(encoding="utf-8"))
    pq = meta["per_question"]
    arm = meta["arm"]
    block_key = "A" if arm == "A" else ("B" if arm == "B" else "proposed")
    verdicts: dict[str, str] = {}
    if score is not None:
        items = score[block_key]["items"] if block_key in score else []
        verdicts = {it["qid"]: it["verdict"] for it in items}
    rec = {
        "cell": cell_dir.name, "arm": arm, "W": meta.get("W", "full") if arm != "A" else "full",
        "w": meta["w"], "n_questions": len(pq),
        "pass_total": sum(v == "pass" for v in verdicts.values()),
        "pass_facts": sum(v == "pass" and kinds.get(q) != "absent" for q, v in verdicts.items()),
        "pass_absent": sum(v == "pass" and kinds.get(q) == "absent" for q, v in verdicts.items()),
        "indeterminate": sum(v == "indeterminate" for v in verdicts.values()),
        "window_tokens": meta["window_tokens"], "cd_tokens": meta["cd_tokens"],
        "n_recent_rts": meta["n_recent_rts"], "evicted_planets": meta["evicted_planets"],
        "planet_lines": meta["planet_lines"], "chat_template": meta["chat_template"],
        "mean_prompt_tokens": _mean([x["prompt_tokens"] for x in pq.values()]),
        "mean_attn_flops": _mean([x["attn_flops_prefill"] + x["attn_flops_decode"] for x in pq.values()]),
        "mean_wall_ms": _mean([x["wall_ms_total"] for x in pq.values()]),
        "mean_energy_j": _mean([x.get("energy_joules") for x in pq.values()]),
    }
    if arm == "proposed":
        rec.update({
            "inject": meta.get("inject"), "bias_cap": meta.get("bias_cap"),
            "mean_positions_biased": _mean([x["positions_found"] for x in pq.values()]),
            "mean_planet_spans": _mean([x["planet_spans"] for x in pq.values()]),
            "mean_satellite_spans": _mean([x["satellite_spans"] for x in pq.values()]),
            "bias_applied_calls_total": sum(x["bias_applied_calls"] for x in pq.values()),
        })
    rec["per_question_verdict"] = verdicts
    return rec


def build_numbers(a) -> dict:
    scores = json.loads(Path(a.scores).read_text(encoding="utf-8"))
    by_cell = {c["cell"]: c for c in scores["cells"]}
    kinds = {q["qid"]: q["kind"] for q in json.loads(Path(a.questions).read_text(encoding="utf-8"))}
    raw = Path(a.raw)
    out = {"kind": "mcbuild_bench_all_numbers", "date": a.date, "cells": []}
    if a.extra:
        out.update(json.loads(Path(a.extra).read_text(encoding="utf-8")))
    # A once: scored inside every proposed cell on that cell's subset; report it on the full set
    a_dir = raw / scores.get("a_dir", "A_full")
    if (a_dir / "meta.json").exists():
        full = max(scores["cells"], key=lambda c: c["n_questions"])
        out["cells"].append(cell_record(a_dir, {"A": full["A"]}, kinds))
    for d in sorted(p for p in raw.iterdir() if p.is_dir() and (p / "meta.json").exists()):
        if d == a_dir:
            continue
        if d.name in by_cell:
            out["cells"].append(cell_record(d, by_cell[d.name], kinds))
        else:  # truncation control: scored inside the proposed cell at the same W
            host = next((c for c in scores["cells"] if c.get("b_dir") == d.name), None)
            out["cells"].append(cell_record(d, {"B": host["B"]} if host else None, kinds))
    return out


def build_manifest(a) -> dict:
    root = Path(a.manifest)
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "raw_manifest.json":
            b = p.read_bytes()
            files.append({"path": p.relative_to(root).as_posix(), "bytes": len(b),
                          "sha256": hashlib.sha256(b).hexdigest()})
    return {"kind": "mcbuild_bench_raw_manifest", "date": a.date, "pod": a.pod,
            "archive_sha256": a.archive_sha, "pod_files_verified_by_sha256": a.archive_sha is not None,
            "files": files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw"); ap.add_argument("--scores"); ap.add_argument("--questions")
    ap.add_argument("--manifest"); ap.add_argument("--pod"); ap.add_argument("--archive-sha")
    ap.add_argument("--extra"); ap.add_argument("--date", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    doc = build_manifest(a) if a.manifest else build_numbers(a)
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", a.out, "cells" if not a.manifest else "files", len(doc["cells"] if not a.manifest else doc["files"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
