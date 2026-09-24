"""Run every prompt in eval/data/prompts.jsonl through Laya with the gateway's exact questions.

Saves raw scores to eval/data/scores.jsonl (resumable: already-scored prompts are skipped), so
thresholds can be tuned by eval/report.py without re-running the model.

  python eval/score.py                               # load Laya in this process
  python eval/score.py --server http://localhost:8000   # reuse a running Leanroute server
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
from app.gateway import GUARD_QUESTIONS, ROUTER_QUESTIONS  # noqa: E402

DATA = ROOT / "eval" / "data"
MAX_TEXT = 3900  # the gateway reads up to 4000 chars; /v1/decide also counts the JSON wrapper


def local_predict():
    from app.engine import build_engine
    engine = build_engine()
    return engine.predict


def server_predict(url: str):
    def predict(state, questions):
        body = json.dumps({"state": state, "questions": questions}).encode()
        req = urllib.request.Request(f"{url.rstrip('/')}/v1/decide", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", help="URL of a running Leanroute server (default: load Laya locally)")
    args = ap.parse_args()
    predict = server_predict(args.server) if args.server else local_predict()

    prompts = [json.loads(line) for line in (DATA / "prompts.jsonl").open()]
    out_path = DATA / "scores.jsonl"
    done = {json.loads(line)["id"] for line in out_path.open()} if out_path.exists() else set()
    todo = [p for p in prompts if p["id"] not in done]
    print(f"{len(prompts)} prompts, {len(done)} already scored, {len(todo)} to go", flush=True)

    t_start = time.time()
    with out_path.open("a") as out:
        for i, p in enumerate(todo, 1):
            text = p["text"][:MAX_TEXT]
            t0 = time.perf_counter()
            g = predict({"prompt": text}, GUARD_QUESTIONS)["answers"]
            r = predict({"request": text}, ROUTER_QUESTIONS)["answers"]
            out.write(json.dumps({
                "id": p["id"],
                "jailbreak": g["g_jailbreak"]["noul"], "injection": g["g_injection"]["noul"],
                "difficulty": r["r_difficulty"]["score"], "difficulty_conf": r["r_difficulty"]["confidence"],
                "sensitive": r["r_sensitive"]["noul"], "ms": round((time.perf_counter() - t0) * 1000, 1),
            }) + "\n")
            out.flush()
            if i % 50 == 0 or i == len(todo):
                rate = i / (time.time() - t_start)
                print(f"  {i}/{len(todo)} scored, ~{(len(todo) - i) / rate / 60:.1f} min left", flush=True)


if __name__ == "__main__":
    main()
