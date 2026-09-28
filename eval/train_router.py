"""Train Leanroute's own router on real "was the cheap answer good enough?" labels.

Data: routellm/gpt4_dataset (Apache-2.0): 109k real prompts, each with a Mixtral answer that GPT-4
scored 1-5. Label: the prompt NEEDS the strong model if Mixtral scored below 4.
Model: bge-small-en-v1.5 embeddings (MIT) + logistic regression. Runs on a laptop (Apple GPU / CPU).

Writes server/app/router_head.json (the trained weights, ours under this repo's license) and prints
results on the held-out validation split, the hand-written check set, and Laya on the same labels.

  python eval/train_router.py [--laya-url http://localhost:8000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "eval" / "data" / "routing"
EMBEDDER = "BAAI/bge-small-en-v1.5"
OUT = ROOT / "server" / "app" / "router_head.json"
URL = "https://huggingface.co/api/datasets/routellm/gpt4_dataset/parquet/default/{split}/0.parquet"


def load_split(split):
    path = DATA / f"{split}.parquet"
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.format(split=split), path)
    t = pq.read_table(path, columns=["prompt", "mixtral_score"]).to_pydict()
    return t["prompt"], np.array([int(s) < 4 for s in t["mixtral_score"]])


class Embedder:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(EMBEDDER)
        self.model = AutoModel.from_pretrained(EMBEDDER).to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, texts, batch=64):
        out = []
        order = np.argsort([len(t) for t in texts])  # similar lengths per batch = less padding
        for i in range(0, len(texts), batch):
            idx = order[i:i + batch]
            enc = self.tok([texts[j] for j in idx], truncation=True, max_length=512, padding=True,
                           return_tensors="pt").to(self.device)
            cls = self.model(**enc).last_hidden_state[:, 0]
            out.append((idx, torch.nn.functional.normalize(cls, dim=-1).float().cpu().numpy()))
        emb = np.zeros((len(texts), out[0][1].shape[1]), dtype=np.float32)
        for idx, e in out:
            emb[idx] = e
        return emb


def cached(name, texts, embed):
    path = DATA / f"emb_{name}.npy"
    if path.exists():
        return np.load(path)
    t0 = time.time()
    emb = embed(texts)
    np.save(path, emb)
    print(f"  embedded {len(texts)} {name} prompts in {time.time() - t0:.0f}s", flush=True)
    return emb


def laya_difficulty(url, texts):
    sys.path.insert(0, str(ROOT / "server"))
    from app.gateway import ROUTER_QUESTIONS
    out = []
    for t in texts:
        body = json.dumps({"state": {"request": t[:3900]}, "questions": ROUTER_QUESTIONS}).encode()
        req = urllib.request.Request(f"{url.rstrip('/')}/v1/decide", body, {"Content-Type": "application/json"})
        out.append(json.load(urllib.request.urlopen(req, timeout=120))["answers"]["r_difficulty"]["score"])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laya-url", help="also score a validation sample with Laya via a running server")
    ap.add_argument("--laya-sample", type=int, default=1000)
    args = ap.parse_args()

    train_x, train_y = load_split("train")
    val_x, val_y = load_split("validation")
    hand = [json.loads(line) for line in (ROOT / "eval" / "handwritten.jsonl").open()]
    hand_y = np.array([h["label"] == "hard" for h in hand])
    print(f"train {len(train_x)} ({train_y.mean():.0%} need the strong model), validation {len(val_x)}", flush=True)

    embed = Embedder()
    Xtr, Xval = cached("train", train_x, embed), cached("validation", val_x, embed)
    Xhand = embed([h["text"] for h in hand])

    clf = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr, train_y)
    p_val, p_hand = clf.predict_proba(Xval)[:, 1], clf.predict_proba(Xhand)[:, 1]
    val_auc, hand_auc = roc_auc_score(val_y, p_val), roc_auc_score(hand_y, p_hand)
    print(f"\nLeanroute router  validation AUC {val_auc:.3f} | hand-written AUC {hand_auc:.3f}")
    for h, p in sorted(zip(hand, p_hand), key=lambda x: x[1]):
        print(f"   {p:.2f}  {h['label']:4}  {h['text'][:70]}")

    if args.laya_url:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(val_x), size=min(args.laya_sample, len(val_x)), replace=False)
        d = laya_difficulty(args.laya_url, [val_x[i] for i in idx])
        print(f"\nOn the same {len(idx)} validation prompts: Leanroute router AUC {roc_auc_score(val_y[idx], p_val[idx]):.3f}"
              f" vs Laya difficulty AUC {roc_auc_score(val_y[idx], d):.3f}")

    OUT.write_text(json.dumps({
        "embedding_model": EMBEDDER, "pooling": "cls+l2",
        "predicts": "P(the cheap model's answer is not good enough), i.e. GPT-4 scored Mixtral below 4/5",
        "trained_on": "routellm/gpt4_dataset train split (Apache-2.0)",
        "validation_auc": round(val_auc, 4), "handwritten_auc": round(hand_auc, 4),
        "intercept": float(clf.intercept_[0]), "coef": [round(float(c), 6) for c in clf.coef_[0]],
    }))
    print(f"\nsaved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
