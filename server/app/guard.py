"""Prompt-injection detector used by the gateway's default guard.

ProtectAI's open-source DeBERTa classifier (Apache-2.0). In eval/results.md it blocked far fewer
normal requests than Laya's guard questions and never blocked a coding request. English only; it
was not trained to catch long role-play jailbreaks, which GUARD_MODE=broad adds Laya for.
"""
from __future__ import annotations

import os
import threading

MODEL = os.getenv("PROTECTAI_MODEL", "protectai/deberta-v3-base-prompt-injection-v2")


class ProtectAIGuard:
    name = "protectai"

    def __init__(self, model: str = MODEL):
        self.model_id = model
        self._lock = threading.Lock()
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # installed with laya
            tok = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_id).eval()
            label = [i for i, n in model.config.id2label.items() if n.upper() == "INJECTION"][0]
            self._loaded = (tok, model, label)
        return self._loaded

    def score(self, text: str) -> float:
        """Probability that `text` is a prompt injection."""
        import torch
        with self._lock:
            tok, model, label = self._load()
            with torch.inference_mode():
                enc = tok(text, truncation=True, max_length=512, return_tensors="pt")
                return float(model(**enc).logits.softmax(-1)[0, label])
