"""Score the 3-condition evaluation."""
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
WS = re.compile(r"\s+")


def normalize(s):
    s = unicodedata.normalize("NFKC", str(s).lower())
    for ch in ("‐", "‑", "‒", "–", "—", "－"):
        s = s.replace(ch, "-")
    s = PUNCT.sub(" ", s)
    return WS.sub(" ", s).strip()


def score_strict(gold, pred):
    g = normalize(gold)
    p = normalize(pred)
    return bool(g) and g in p


def parse_answers(text, n):
    answers = {}
    for line in text.strip().split("\n"):
        m = re.match(r"\s*A(\d+)\s*[:\-]\s*(.+)", line.strip(), re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= n:
                answers[idx] = m.group(2).strip()
    return answers


def main():
    questions = json.loads((ROOT / "restaurant_questions.json").read_text(encoding="utf-8"))
    n = len(questions)

    conditions = ["full", "truncated", "summarized"]
    results = {}

    for cond in conditions:
        path = ROOT / "eval_results" / f"answers_{cond}.txt"
        text = path.read_text(encoding="utf-8")
        answers = parse_answers(text, n)
        per_q = []
        correct = 0
        for i, q in enumerate(questions, 1):
            pred = answers.get(i, "")
            ok = score_strict(q["answer"], pred)
            if ok:
                correct += 1
            per_q.append({
                "qid": q["qid"],
                "category": q["category"],
                "gold": q["answer"],
                "predicted": pred,
                "correct": ok,
            })
        results[cond] = {
            "correct": correct,
            "total": n,
            "accuracy": round(correct / n, 4),
            "items": per_q,
        }
        print(f"{cond}: {correct}/{n} = {correct/n:.1%}")

    # Category breakdown
    cats = {}
    for cond in conditions:
        cats[cond] = {}
        for item in results[cond]["items"]:
            c = item["category"]
            cats[cond].setdefault(c, {"correct": 0, "total": 0})
            cats[cond][c]["total"] += 1
            cats[cond][c]["correct"] += int(item["correct"])

    print("\nCategory breakdown:")
    print(f"{'Category':<20} {'Full':<10} {'Truncated':<12} {'Summarized':<12}")
    for cat in ["early_static", "mid_static", "update_tracking", "synthesis"]:
        row = f"{cat:<20}"
        for cond in conditions:
            d = cats[cond].get(cat, {"correct": 0, "total": 0})
            row += f" {d['correct']}/{d['total']:<8}"
        print(row)

    output = {"per_condition": results, "by_category": cats}
    (ROOT / "eval_results" / "scored.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nSaved scored.json")


if __name__ == "__main__":
    main()
