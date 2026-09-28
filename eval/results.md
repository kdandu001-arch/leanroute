# Leanroute evaluation

Run 2026-09-28 with the real models and the gateway's exact logic. 1674 labelled prompts from public datasets (`eval/build_dataset.py`), split in half by a hash of the text. Thresholds were chosen on the dev half; **every number below is from the 873-prompt test half**.

## Guard: blocking attacks without blocking normal users

"Fair" = sources the ProtectAI detector never trained on (142 attacks, 535 normal requests; many attacks are subtle injections, some in German). jackhhao = long role-play jailbreaks; ProtectAI trained on this source, so its number there is flattering.

| Guard mode | Attacks blocked (fair) | Normal requests wrongly blocked (fair) | Coding requests wrongly blocked | Math wrongly blocked | jackhhao jailbreaks blocked |
|---|---|---|---|---|---|
| precise (default): ProtectAI ≥ 0.66 | 35.9% | 0.4% | 0.0% | 0.0% | 80.2% |
| broad: ProtectAI ≥ 0.66, or Laya both ≥ 0.99 | 43.7% | 2.2% | 8.0% | 6.2% | 98.2% |
| laya: Laya both scores ≥ 0.92 | 26.1% | 3.0% | 13.8% | 8.3% | 99.1% |
| old default: Laya either score ≥ 0.85 | 50.0% | 11.0% | 54.0% | 16.7% | 100.0% |

## Routing on real labels (default router)

The only test that matches the real question: *will the cheap model's answer be good enough?* `routellm/gpt4_dataset` (Apache-2.0) has real prompts whose cheap-model (Mixtral) answers GPT-4 scored 1-5; "good enough" = 4 or 5. Held-out validation split, 10,000 prompts; 86.5% of cheap answers were good enough with no routing at all.

| Share of traffic sent to the cheap model | Cheap answers good enough: Leanroute router | Laya difficulty |
|---|---|---|
| 30.0% | 96.0% | 90.0% |
| 50.0% | 94.0% | 89.4% |
| 70.0% | 92.3% | 89.4% |
| 85.0% | 91.3% | 89.2% |

On the same 1,000 prompts, AUC for spotting answers that won't be good enough: Leanroute router **0.72**, Laya **0.54** (0.5 = coin flip). At the default `ROUTER_THRESHOLD=0.107` the router sends 50.2% of all 10,000 validation prompts to the cheap model, and 94.1% of those answers were good enough.

On 14 hand-written prompts labelled by us as easy or hard, the router agreed with our labels on 10 (it sent three simple ones to the strong model and one textbook proof to the cheap model); Laya agreed on 13. Our hand labels are guesses about difficulty; the table above uses graded answers, so it is the stronger evidence, but the router is a clear improvement, not a solved problem.

## Routing: easy vs. hard (proxy test)

How well Laya's difficulty score separates easy from hard requests: AUC **0.67** (0.5 = coin flip, 1.0 = perfect).

| ROUTER_EASY_MAX | Easy questions sent to the cheap model | Hard questions sent to the cheap model |
|---|---|---|
| 1.0 | 8.8% | 5.2% |
| 1.1 | 17.5% | 8.9% |
| 1.2 (default) | 32.5% | 13.3% |
| 1.3 | 47.5% | 20.7% |
| 1.4 | 58.8% | 38.5% |

Easy = short trivia questions (Natural Questions). Hard = Level 5 competition math and coding tasks (MATH, HumanEval). These are proxies: they test whether routing separates clearly easy from clearly hard requests, not whether a cheap model's answers were good enough.

**Caution:** the hard examples are mostly math notation and code, so a router can score well here from surface features alone. RouteLLM's learned router scored AUC 0.99 on this test, yet could not tell plain-language easy and hard questions apart in a hand-written check (e.g. "Derive the backpropagation equations" scored as easier than "What year did World War II end?"). A better routing test needs labels for whether a cheap model's answer was actually good enough.

Average Laya decision time: 164 ms per request (two passes, CPU).
