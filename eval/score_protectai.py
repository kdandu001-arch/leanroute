"""Score every prompt with ProtectAI's open-source prompt-injection detector (Apache-2.0).

Writes eval/data/scores_protectai.jsonl: {"id", "protectai": P(injection)}. Resumable.
Note: the detector was trained on jackhhao/jailbreak-classification, so results on that source are
not a fair test of it; eval/report.py reports sources it never saw separately.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
DATA = Path(__file__).resolve().parent / "data"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    injection = [i for i, name in model.config.id2label.items() if name.upper() == "INJECTION"][0]

    prompts = [json.loads(line) for line in (DATA / "prompts.jsonl").open()]
    out_path = DATA / "scores_protectai.jsonl"
    done = {json.loads(line)["id"] for line in out_path.open()} if out_path.exists() else set()
    todo = [p for p in prompts if p["id"] not in done]
    print(f"{len(todo)} prompts to score", flush=True)
    t0 = time.time()
    with out_path.open("a") as out, torch.inference_mode():
        for i in range(0, len(todo), 16):
            batch = todo[i:i + 16]
            enc = tok([p["text"] for p in batch], truncation=True, max_length=512, padding=True, return_tensors="pt")
            probs = model(**enc).logits.softmax(-1)[:, injection].tolist()
            for p, s in zip(batch, probs):
                out.write(json.dumps({"id": p["id"], "protectai": round(s, 5)}) + "\n")
            out.flush()
            if (i // 16) % 10 == 0:
                print(f"  {i + len(batch)}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
