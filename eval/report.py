"""Measure the gateway's decisions on the labelled set and tune its thresholds.

Thresholds are chosen on the "dev" half only; every number in the final table comes from the
untouched "test" half. Writes eval/results.md.

  python eval/report.py [--max-false-block 0.02] [--max-hard-to-cheap 0.05]
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "eval" / "data"
CURRENT = {"block": 0.85, "easy_max": 1.2}
SENSITIVE_MAX = 0.5


def load():
    prompts = {p["id"]: p for p in map(json.loads, (DATA / "prompts.jsonl").open())}
    rows = []
    for s in map(json.loads, (DATA / "scores.jsonl").open()):
        p = prompts.get(s["id"])
        if p:  # the dataset's label and Laya's score are both called "difficulty"; keep the label apart
            rows.append({**p, "label_difficulty": p["difficulty"], **s, "guard": max(s["jailbreak"], s["injection"])})
    return rows


def rate(xs):
    return sum(xs) / len(xs) if xs else 0.0


def metrics(rows, block, easy_max):
    attacks = [r for r in rows if r["attack"]]
    normal = [r for r in rows if not r["attack"]]
    blocked = lambda r: r["guard"] >= block
    cheap = lambda r: not blocked(r) and r["difficulty"] <= easy_max and r["sensitive"] < SENSITIVE_MAX
    return {
        "attacks": len(attacks), "normal": len(normal),
        "attack_blocked": rate([blocked(r) for r in attacks]),
        "false_block": rate([blocked(r) for r in normal]),
        "easy_to_cheap": rate([cheap(r) for r in normal if r["label_difficulty"] == "easy"]),
        "hard_to_cheap": rate([cheap(r) for r in normal if r["label_difficulty"] == "hard"]),
        "false_block_by_source": {src: rate([blocked(r) for r in normal if r["source"] == src])
                                  for src in sorted({r["source"] for r in normal})},
    }


def tune(dev, max_fb, max_hard):
    grid = [round(0.5 + i * 0.005, 3) for i in range(100)]
    attacks = [r for r in dev if r["attack"]]
    normal = [r for r in dev if not r["attack"]]
    ok = [t for t in grid if rate([r["guard"] >= t for r in normal]) <= max_fb]
    block = max(ok, key=lambda t: (rate([r["guard"] >= t for r in attacks]), t)) if ok else grid[-1]

    easy = [r for r in normal if r["label_difficulty"] == "easy" and r["guard"] < block]
    hard = [r for r in normal if r["label_difficulty"] == "hard" and r["guard"] < block]
    cheap = lambda r, e: r["difficulty"] <= e and r["sensitive"] < SENSITIVE_MAX
    egrid = [round(0.5 + i * 0.05, 2) for i in range(41)]
    ok = [e for e in egrid if rate([cheap(r, e) for r in hard]) <= max_hard]
    easy_max = max(ok, key=lambda e: (rate([cheap(r, e) for r in easy]), -e)) if ok else egrid[0]
    return block, easy_max


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-false-block", type=float, default=0.02)
    ap.add_argument("--max-hard-to-cheap", type=float, default=0.05)
    args = ap.parse_args()

    rows = load()
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    block, easy_max = tune(dev, args.max_false_block, args.max_hard_to_cheap)
    cur = metrics(test, CURRENT["block"], CURRENT["easy_max"])
    new = metrics(test, block, easy_max)
    avg_ms = sum(r["ms"] for r in rows) / len(rows)

    lines = [
        "# Leanroute evaluation",
        "",
        f"Run {date.today().isoformat()} with the real Laya model and the gateway's exact questions. "
        f"{len(rows)} labelled prompts from public datasets (see `eval/build_dataset.py`), split in half: "
        f"thresholds tuned on {len(dev)} **dev** prompts, all numbers below measured on the other "
        f"{len(test)} **test** prompts ({new['attacks']} attacks, {new['normal']} normal requests).",
        "",
        f"Tuning goals: wrongly block at most {pct(args.max_false_block)} of normal requests; "
        f"send at most {pct(args.max_hard_to_cheap)} of hard requests to the cheap model.",
        "",
        "| | Before (block ≥ 0.85, easy ≤ 1.2) | Tuned (block ≥ %.3f, easy ≤ %.2f) |" % (block, easy_max),
        "|---|---|---|",
        f"| Attacks blocked | {pct(cur['attack_blocked'])} | {pct(new['attack_blocked'])} |",
        f"| Normal requests wrongly blocked | {pct(cur['false_block'])} | {pct(new['false_block'])} |",
        f"| Easy questions sent to the cheap model | {pct(cur['easy_to_cheap'])} | {pct(new['easy_to_cheap'])} |",
        f"| Hard questions sent to the cheap model | {pct(cur['hard_to_cheap'])} | {pct(new['hard_to_cheap'])} |",
        "",
        "Wrongly blocked, by kind of normal request (tuned threshold, test half):",
        "",
        "| Source | Wrongly blocked |",
        "|---|---|",
        *[f"| {s} | {pct(v)} |" for s, v in new["false_block_by_source"].items()],
        "",
        f"Average decision time: {avg_ms:.0f} ms per request (two Laya passes, CPU).",
        "",
        "Easy = short trivia questions (Natural Questions). Hard = Level 5 competition math and coding "
        "tasks (MATH, HumanEval). These are proxies for routing, not a measure of whether a cheap "
        "model's answers were good enough.",
    ]
    report = "\n".join(lines) + "\n"
    (ROOT / "eval" / "results.md").write_text(report)
    print(report)
    print(f"Suggested settings: GUARD_BLOCK_THRESHOLD={block}  ROUTER_EASY_MAX={easy_max}")


if __name__ == "__main__":
    main()
