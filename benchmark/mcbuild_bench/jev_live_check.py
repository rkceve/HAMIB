"""One-shot LIVE contract check of the Jev API (needs TYPESAFE_API_KEY).

Sends the three question types with the exact request shapes documented in
DESIGN.md §3, once each, through the real ``JevClient`` (real HTTP), and
asserts the answer shapes the code relies on.  Cost: a few hundred input
tokens (well under one cent at $0.042 / M).  Nothing about the key is printed;
the accounting lines go to the path given with ``--accounting``.

Usage (PowerShell):
    $env:TYPESAFE_API_KEY = "<key>"
    python -m benchmark.mcbuild_bench.jev_live_check --accounting <path>.jsonl

Exit 0 = every assertion held; non-zero = the live API deviates from the
documented contract (report the printed diff before any GPU is rented).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from benchmark.mcbuild_bench.jev_client import JevClient, argmax_level, choice_of, noul_of

STATE = "Help! My payouts have been failing for 3 days."

QUESTIONS = {
    "is_urgent": {
        "type": "noul",
        "instructions": "Does this convey urgency?",
        "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
    },
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Payments, invoicing, refunds",
            "technical": "Bugs, outages, integrations",
            "sales": "Pricing, upgrades, new accounts",
        },
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"],
    },
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--accounting", required=True, help="jsonl path for the accounting lines")
    args = ap.parse_args(argv)

    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        print("TYPESAFE_API_KEY is not set", file=sys.stderr)
        return 2

    client = JevClient(key, accounting_path=args.accounting)
    result = client.ask(STATE, QUESTIONS)
    answers = result["answers"]
    problems: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            problems.append(msg)

    a = answers["is_urgent"]
    check(a.get("type") == "noul", f"is_urgent.type = {a.get('type')!r}")
    check(0.0 <= noul_of(a) <= 1.0, "noul outside [0, 1]")

    c = answers["department"]
    check(c.get("type") == "choice", f"department.type = {c.get('type')!r}")
    check(choice_of(c) in QUESTIONS["department"]["criteria"], f"choice {c.get('choice')!r} not an option")
    check(set(c.get("probabilities", {})) == set(QUESTIONS["department"]["criteria"]),
          f"choice probabilities keys = {sorted(c.get('probabilities', {}))}")
    check(isinstance(c.get("confidence"), (int, float)), "choice.confidence missing")

    s = answers["frustration"]
    check(s.get("type") == "score", f"frustration.type = {s.get('type')!r}")
    check(set(s.get("legend", {})) == {"0", "1", "2"}, f"score legend keys = {sorted(s.get('legend', {}))}")
    check(all(k in {"0", "1", "2"} for k in s.get("probabilities", {})),
          f"score probability keys = {sorted(s.get('probabilities', {}))}")
    check(0 <= argmax_level(s, n_levels=3) <= 2, "argmax_level out of range")
    check(isinstance(s.get("score"), (int, float)), "score.score missing")

    usage = result["usage"]
    check(isinstance(usage.get("input_tokens"), int) and usage["input_tokens"] > 0, f"usage = {usage}")

    print(json.dumps({"answers": answers, "usage": usage, "latency_ms": result["latency_ms"],
                      "http_status": result["http_status"], "retries": result["retries"]},
                     ensure_ascii=False, indent=1))
    if problems:
        print("CONTRACT DEVIATIONS:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print("live contract check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
