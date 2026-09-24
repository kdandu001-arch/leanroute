"""Optional learned router: RouteLLM's BERT classifier (ROUTER=routellm).

Trained by LMSYS on Chatbot Arena preferences augmented with GPT-4 judgments. It predicts how likely
a strong model's answer is to be clearly better than a weaker one's. In eval/results.md it separated
easy from hard requests far better than Laya's difficulty score.

Licensing: RouteLLM's code (github.com/lm-sys/RouteLLM) and training data (routellm/gpt4_dataset) are
Apache-2.0, but the model weights on Hugging Face carry no license file. Check that this is acceptable
for your use before enabling it commercially.
"""
from __future__ import annotations

import os
import threading

MODEL = os.getenv("ROUTELLM_MODEL", "routellm/bert_gpt4_augmented")


class RouteLLMRouter:
    name = "routellm"

    def __init__(self, model: str = MODEL):
        self.model_id = model
        self._lock = threading.Lock()
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # installed with laya
            tok = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=3).eval()
            self._loaded = (tok, model)
        return self._loaded

    def strong_win(self, text: str) -> float:
        """Probability that the strong model's answer is clearly better (RouteLLM's win rate)."""
        import torch
        with self._lock:
            tok, model = self._load()
            with torch.inference_mode():
                enc = tok(text, truncation=True, max_length=512, return_tensors="pt")
                return float(model(**enc).logits.softmax(-1)[0, 0])
