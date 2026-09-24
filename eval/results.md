# Leanroute evaluation

Run 2026-09-24 with the real models and the gateway's exact logic. 1674 labelled prompts from public datasets (`eval/build_dataset.py`), split in half by a hash of the text. Thresholds were chosen on the dev half; **every number below is from the 873-prompt test half**.

## Guard: blocking attacks without blocking normal users

"Fair" = sources the ProtectAI detector never trained on (142 attacks, 535 normal requests; many attacks are subtle injections, some in German). jackhhao = long role-play jailbreaks; ProtectAI trained on this source, so its number there is flattering.

| Guard mode | Attacks blocked (fair) | Normal requests wrongly blocked (fair) | Coding requests wrongly blocked | Math wrongly blocked | jackhhao jailbreaks blocked |
|---|---|---|---|---|---|
| precise (default): ProtectAI ≥ 0.66 | 35.9% | 0.4% | 0.0% | 0.0% | 80.2% |
| broad: ProtectAI ≥ 0.66, or Laya both ≥ 0.99 | 43.7% | 2.2% | 8.0% | 6.2% | 98.2% |
| laya: Laya both scores ≥ 0.92 | 26.1% | 3.0% | 13.8% | 8.3% | 99.1% |
| old default: Laya either score ≥ 0.85 | 50.0% | 11.0% | 54.0% | 16.7% | 100.0% |

## Routing: easy vs. hard

How well Laya's difficulty score separates easy from hard requests: AUC **0.67** (0.5 = coin flip, 1.0 = perfect).

| ROUTER_EASY_MAX | Easy questions sent to the cheap model | Hard questions sent to the cheap model |
|---|---|---|
| 1.0 | 8.8% | 5.2% |
| 1.1 | 17.5% | 8.9% |
| 1.2 (default) | 32.5% | 13.3% |
| 1.3 | 47.5% | 20.7% |
| 1.4 | 58.8% | 38.5% |

Easy = short trivia questions (Natural Questions). Hard = Level 5 competition math and coding tasks (MATH, HumanEval). These are proxies: they test whether routing separates clearly easy from clearly hard requests, not whether a cheap model's answers were good enough.

Average Laya decision time: 164 ms per request (two passes, CPU).
