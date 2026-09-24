"""Download a labelled evaluation set from public Hugging Face datasets.

Writes eval/data/prompts.jsonl, one prompt per line:
  {"id", "text", "attack": bool, "difficulty": "easy"|"hard"|null, "source", "split": "dev"|"test"}

Sources (fetched at build time, not redistributed in this repo):
  deepset/prompt-injections           Apache-2.0   injections (label 1) and normal requests (label 0)
  jackhhao/jailbreak-classification   Apache-2.0   jailbreaks and benign role-play prompts
  databricks/databricks-dolly-15k     CC-BY-SA-3.0 everyday instructions (normal requests)
  google-research-datasets/nq_open    CC-BY-SA-3.0 short trivia questions (easy)
  lighteval/MATH-Hard                 MIT          Level 5 competition math (hard)
  openai/openai_humaneval             MIT          coding tasks (hard)

"easy"/"hard" are proxies: trivia lookups vs. competition math and coding. They test whether routing
separates clearly easy from clearly hard requests, not whether a cheap model would answer correctly.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "prompts.jsonl"
API = "https://datasets-server.huggingface.co/rows"


def fetch(dataset: str, config: str, split: str, limit: int = 100_000):
    rows, offset = [], 0
    while offset < limit:
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                    "offset": offset, "length": 100})
        for attempt in range(8):
            try:
                with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
                    page = json.load(r)
                break
            except urllib.error.HTTPError as e:  # 429 = rate limited: wait as long as the server asks
                if attempt == 7:
                    raise
                time.sleep(float(e.headers.get("Retry-After") or min(60, 5 * 2 ** attempt)))
            except OSError:
                if attempt == 7:
                    raise
                time.sleep(min(60, 5 * 2 ** attempt))
        batch = [x["row"] for x in page["rows"]]
        rows += batch
        offset += len(batch)
        time.sleep(0.5)
        if not batch or offset >= page.get("num_rows_total", 0):
            break
    return rows[:limit]


def sample(rows, n, rng):
    return rows if len(rows) <= n else rng.sample(rows, n)


def main():
    rng = random.Random(1234)
    items = []

    def add(text, attack, difficulty, source):
        text = (text or "").strip()
        if text:
            items.append({"text": text, "attack": attack, "difficulty": difficulty, "source": source})

    deepset = fetch("deepset/prompt-injections", "default", "train") + fetch("deepset/prompt-injections", "default", "test")
    for r in deepset:
        add(r["text"], str(r["label"]) == "1", None, "deepset")

    jb = fetch("jackhhao/jailbreak-classification", "default", "train") + fetch("jackhhao/jailbreak-classification", "default", "test")
    for r in sample([r for r in jb if r["type"] == "jailbreak"], 200, rng):
        add(r["prompt"], True, None, "jackhhao-jailbreak")
    for r in sample([r for r in jb if r["type"] == "benign"], 150, rng):
        add(r["prompt"], False, None, "jackhhao-benign")

    dolly = fetch("databricks/databricks-dolly-15k", "default", "train", limit=3000)
    for r in sample([r for r in dolly if not r["context"]], 250, rng):
        add(r["instruction"], False, None, f"dolly-{r['category']}")

    for r in sample(fetch("google-research-datasets/nq_open", "nq_open", "validation", limit=1000), 150, rng):
        q = r["question"]
        add(q[0].upper() + q[1:] + "?", False, "easy", "nq_open")

    for r in sample(fetch("lighteval/MATH-Hard", "default", "test", limit=400), 100, rng):
        add(r["problem"], False, "hard", "math-level5")
    for r in fetch("openai/openai_humaneval", "openai_humaneval", "test"):
        add("Complete this Python function:\n" + r["prompt"], False, "hard", "humaneval")

    seen, out = set(), []
    for it in items:
        key = hashlib.sha1(it["text"].encode()).hexdigest()[:12]
        if key in seen:
            continue
        seen.add(key)
        it["id"] = key
        it["split"] = "dev" if int(key, 16) % 2 == 0 else "test"
        out.append(it)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    attacks = sum(i["attack"] for i in out)
    print(f"wrote {len(out)} prompts to {OUT}: {attacks} attacks, {len(out) - attacks} normal "
          f"({sum(i['difficulty'] == 'easy' for i in out)} easy, {sum(i['difficulty'] == 'hard' for i in out)} hard)")


if __name__ == "__main__":
    main()
