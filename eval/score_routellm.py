"""Score the easy/hard prompts with RouteLLM's learned router (ROUTER=routellm in the gateway).

Writes eval/data/scores_routellm.json: {id: P(strong model clearly wins)}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
from app.router_model import RouteLLMRouter  # noqa: E402

DATA = ROOT / "eval" / "data"


def main():
    router = RouteLLMRouter()
    prompts = [json.loads(line) for line in (DATA / "prompts.jsonl").open()]
    todo = [p for p in prompts if p["difficulty"] in ("easy", "hard")]
    scores = {p["id"]: round(router.strong_win(p["text"]), 5) for p in todo}
    (DATA / "scores_routellm.json").write_text(json.dumps(scores))
    print(f"scored {len(scores)} prompts")


if __name__ == "__main__":
    main()
