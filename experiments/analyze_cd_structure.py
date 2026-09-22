"""
Standalone CD structural quality analysis — runs the management unit
(no LLM inference, no GPU required) and computes quality metrics on the
resulting Correlation Diagrams.

For each LongMemEval question the script:
  1. Replays the haystack sessions through the management pipeline
     (dialogue_extractor + TextChunker + NodeClassifier + GraphMerger)
  2. Captures the final CD state
  3. Computes structural quality metrics

Usage:
    python -m experiments.analyze_cd_structure \
        --n-questions 200 \
        --output results/cd_structure_analysis.json

    # Fast mode (skip expensive embedding-based metrics):
    python -m experiments.analyze_cd_structure \
        --n-questions 5 --skip-duplicates --skip-scorer \
        --output results/cd_structure_smoke.json
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from store.cd_store import CDStore
from models.correlation_diagram import CorrelationDiagram
from management.text_chunker import TextChunker
from management.node_classifier import NodeClassifier
from management.graph_builder import GraphBuilder
from management.graph_merger import GraphMerger
from communication.cd_serializer import CDSerializer
from experiments.dialogue_extractor import make_dialogue_extractor_fn
from utils.config import get

LONGMEMEVAL_PATH = os.environ.get(
    "HAMIB_LONGMEMEVAL_PATH",
    str(_ROOT / "data" / "longmemeval_s"),
)


def _load_questions(n: int):
    with open(LONGMEMEVAL_PATH) as f:
        data = json.load(f)
    return data[:n]


def _flat_pairs(sessions: list) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for sess in sessions:
        pending_user = None
        for turn in sess:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            tx = f"{role.capitalize()}: {content}" if (role and content) else content
            if role == "user":
                if pending_user is not None:
                    out.append((pending_user, ""))
                pending_user = tx
            elif role == "assistant":
                u = pending_user or ""
                out.append((u, tx))
                pending_user = None
        if pending_user is not None:
            out.append((pending_user, ""))
    return out


def _update_cd_standalone(
    cd: CorrelationDiagram,
    chunker: TextChunker,
    classifier: NodeClassifier,
    builder: GraphBuilder,
    merger: GraphMerger,
    extractor_fn,
    user_text: str,
    assistant_text: str,
    turn: int,
) -> None:
    provisional = CorrelationDiagram()
    chunks = chunker.chunk_turn(user_text, assistant_text, turn)
    for chunk in chunks:
        classifier.classify(chunk, provisional, extractor_fn, builder=builder)
    merger.merge(cd, provisional)


def compute_cd_metrics(
    cd: CorrelationDiagram,
    *,
    skip_duplicates: bool = False,
    skip_scorer: bool = False,
    cap_sun_mass: float = 1.0,
    cap_planet_mass: float = 0.5,
    cap_sat_mass: float = 0.1,
) -> dict:
    nodes = list(cd.all_nodes())
    n_nodes = len(nodes)
    if n_nodes == 0:
        return {"n_nodes": 0}

    n_suns = len(cd.suns)
    n_planets = sum(len(se.planets) for se in cd.suns)
    n_satellites = n_nodes - n_suns - n_planets

    orphan_suns = sum(1 for se in cd.suns if not se.planets)
    orphan_sun_rate = orphan_suns / n_suns if n_suns > 0 else 0.0

    masses = [n.mass for n in nodes]
    total_mass = sum(masses)
    if total_mass > 0 and n_nodes > 1:
        probs = [m / total_mass for m in masses]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        max_entropy = math.log(n_nodes)
        mass_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        mass_entropy = 0.0

    serializer = CDSerializer()
    for se in cd.suns:
        if se.sun.mass > cap_sun_mass:
            se.sun.mass = cap_sun_mass
        for pe in se.planets:
            if pe.planet.mass > cap_planet_mass:
                pe.planet.mass = cap_planet_mass
            for sat in pe.satellites:
                if sat.mass > cap_sat_mass:
                    sat.mass = cap_sat_mass
    context_block = serializer.to_context_block(cd)
    serialized_chars = len(context_block)

    result = {
        "n_nodes": n_nodes,
        "n_suns": n_suns,
        "n_planets": n_planets,
        "n_satellites": n_satellites,
        "hierarchy_depth_ratio": round(n_satellites / n_nodes, 4),
        "orphan_sun_rate": round(orphan_sun_rate, 4),
        "mass_entropy": round(mass_entropy, 4),
        "serialized_chars": serialized_chars,
    }

    if not skip_duplicates:
        result["duplicate_node_rate"] = _compute_duplicate_rate(nodes)

    if not skip_scorer:
        from evaluation.scorer import Scorer
        scorer = Scorer()
        scores = scorer.score(cd)
        result["scorer_contradiction"] = scores["contradiction"]
        result["scorer_concentration"] = scores["concentration"]
        result["scorer_total"] = scores["total"]

    return result


def _compute_duplicate_rate(nodes, threshold: float = 0.92, max_sample: int = 300) -> float:
    from utils.similarity import embed

    texts = [n.text for n in nodes]
    if len(texts) < 2:
        return 0.0

    sampled = len(texts) > max_sample
    if sampled:
        import random
        rng = random.Random(42)
        indices = rng.sample(range(len(texts)), max_sample)
        texts = [texts[i] for i in indices]

    vecs = embed(texts)
    sim_matrix = vecs @ vecs.T
    n = len(texts)
    dup_count = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            if sim_matrix[i, j] >= threshold:
                dup_count += 1
    return round(dup_count / total_pairs, 6) if total_pairs > 0 else 0.0


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def _dist_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    import statistics
    return {
        "n": len(values),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "p25": round(_percentile(values, 25), 2),
        "p75": round(_percentile(values, 75), 2),
    }


def main():
    ap = argparse.ArgumentParser(description="CD structural quality analysis")
    ap.add_argument("--n-questions", type=int, default=200)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-entities", type=int, default=1)
    ap.add_argument("--max-satellites", type=int, default=2)
    ap.add_argument("--skip-duplicates", action="store_true",
                    help="Skip expensive duplicate node rate computation")
    ap.add_argument("--skip-scorer", action="store_true",
                    help="Skip Scorer metrics (requires SBERT embedding)")
    args = ap.parse_args()

    questions = _load_questions(args.n_questions)
    print(f"Loaded {len(questions)} questions from {LONGMEMEVAL_PATH}", flush=True)

    extractor_fn = make_dialogue_extractor_fn(
        max_entities=args.max_entities,
        max_satellites=args.max_satellites,
    )

    per_question: list[dict] = []
    t_total_start = time.perf_counter()

    for qi, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        store = CDStore()
        cd = store.get_current()
        chunker = TextChunker()
        classifier = NodeClassifier()
        builder = GraphBuilder()
        merger = GraphMerger()

        pairs = _flat_pairs(q["haystack_sessions"])
        for turn_i, (u, a) in enumerate(pairs, 1):
            try:
                _update_cd_standalone(
                    cd, chunker, classifier, builder, merger,
                    extractor_fn, u, a, turn_i,
                )
            except Exception:
                pass

        metrics = compute_cd_metrics(
            cd,
            skip_duplicates=args.skip_duplicates,
            skip_scorer=args.skip_scorer,
        )
        build_time = time.perf_counter() - t0
        metrics.update({
            "i": qi,
            "qid": q["question_id"],
            "qtype": q["question_type"],
            "n_pairs_replayed": len(pairs),
            "build_time_s": round(build_time, 2),
        })
        per_question.append(metrics)

        if qi in (1, 2, 3, 5, 10, 25, 50, 100, 150, 200) or qi == len(questions):
            print(f"  [{qi}/{len(questions)}] qid={q['question_id'][:8]}"
                  f"  nodes={metrics['n_nodes']}"
                  f"  suns={metrics['n_suns']}"
                  f"  chars={metrics['serialized_chars']}"
                  f"  {build_time:.1f}s", flush=True)

    total_time = time.perf_counter() - t_total_start

    metric_keys = [
        "n_nodes", "n_suns", "n_planets", "n_satellites",
        "hierarchy_depth_ratio", "orphan_sun_rate", "mass_entropy",
        "serialized_chars",
    ]
    if not args.skip_duplicates:
        metric_keys.append("duplicate_node_rate")
    if not args.skip_scorer:
        metric_keys.extend(["scorer_contradiction", "scorer_concentration", "scorer_total"])

    summary = {}
    for key in metric_keys:
        vals = [float(r[key]) for r in per_question if key in r]
        summary[key] = _dist_stats(vals)

    output = {
        "config": {
            "n_questions": len(questions),
            "max_entities": args.max_entities,
            "max_satellites": args.max_satellites,
            "similarity_threshold": get("management", "similarity_threshold", 0.75),
            "skip_duplicates": args.skip_duplicates,
            "skip_scorer": args.skip_scorer,
            "total_time_s": round(total_time, 1),
        },
        "summary": summary,
        "per_question": per_question,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDone in {total_time:.0f}s -> {args.output}", flush=True)
    print(f"  nodes: mean={summary['n_nodes']['mean']}"
          f"  median={summary['n_nodes']['median']}"
          f"  min={summary['n_nodes']['min']}"
          f"  max={summary['n_nodes']['max']}", flush=True)
    if "serialized_chars" in summary:
        print(f"  serialized_chars: mean={summary['serialized_chars']['mean']}"
              f"  median={summary['serialized_chars']['median']}", flush=True)


if __name__ == "__main__":
    main()
