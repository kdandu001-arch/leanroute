# Leanroute evaluation

Run 2026-09-24 with the real Laya model and the gateway's exact questions. 1674 labelled prompts from public datasets (see `eval/build_dataset.py`), split in half: thresholds tuned on 801 **dev** prompts, all numbers below measured on the other 873 **test** prompts (253 attacks, 620 normal requests).

Tuning goals: wrongly block at most 2.0% of normal requests; send at most 5.0% of hard requests to the cheap model.

| | Before (block ≥ 0.85, easy ≤ 1.2) | Tuned (block ≥ 0.995, easy ≤ 1.05) |
|---|---|---|
| Attacks blocked | 71.9% | 60.9% |
| Normal requests wrongly blocked | 9.7% | 4.4% |
| Easy questions sent to the cheap model | 32.5% | 13.8% |
| Hard questions sent to the cheap model | 8.9% | 5.9% |

Wrongly blocked, by kind of normal request (tuned threshold, test half):

| Source | Wrongly blocked |
|---|---|
| deepset | 0.0% |
| dolly-brainstorming | 0.0% |
| dolly-classification | 0.0% |
| dolly-creative_writing | 0.0% |
| dolly-general_qa | 3.3% |
| dolly-open_qa | 0.0% |
| humaneval | 24.1% |
| jackhhao-benign | 0.0% |
| math-level5 | 10.4% |
| nq_open | 0.0% |

Average decision time: 164 ms per request (two Laya passes, CPU).

Easy = short trivia questions (Natural Questions). Hard = Level 5 competition math and coding tasks (MATH, HumanEval). These are proxies for routing, not a measure of whether a cheap model's answers were good enough.
