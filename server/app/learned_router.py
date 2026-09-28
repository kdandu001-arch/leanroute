"""Leanroute's own router: predicts whether the cheap model's answer will be good enough.

Trained by eval/train_router.py on routellm/gpt4_dataset (Apache-2.0): real prompts whose cheap-model
answers GPT-4 scored 1-5. Uses bge-small-en-v1.5 embeddings (MIT) plus a logistic-regression head
stored in router_head.json. Results are in eval/results.md.
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path

HEAD = Path(__file__).resolve().parent / "router_head.json"


class LearnedRouter:
    name = "leanroute-router"

    def __init__(self, head_path: Path = HEAD):
        self.head_path = head_path
        self._lock = threading.Lock()
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            from transformers import AutoModel, AutoTokenizer  # installed with laya
            head = json.loads(self.head_path.read_text())
            tok = AutoTokenizer.from_pretrained(head["embedding_model"])
            model = AutoModel.from_pretrained(head["embedding_model"]).eval()
            self._loaded = (tok, model, head["coef"], head["intercept"])
        return self._loaded

    def needs_strong(self, text: str) -> float:
        """Probability that the cheap model's answer would not be good enough."""
        import torch
        with self._lock:
            tok, model, coef, intercept = self._load()
            with torch.inference_mode():
                enc = tok(text, truncation=True, max_length=512, return_tensors="pt")
                cls = torch.nn.functional.normalize(model(**enc).last_hidden_state[:, 0], dim=-1)[0]
            z = intercept + float(torch.dot(cls, torch.tensor(coef, dtype=cls.dtype)))
            return 1 / (1 + math.exp(-z))
