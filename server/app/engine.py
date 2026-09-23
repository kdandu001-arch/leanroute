"""Decision engines.

LayaEngine  - the real thing: Laya checkpoints via the `laya` package.
MockEngine  - a tiny keyword heuristic with the SAME output shape, used for
              tests, CI and offline UI development. Every mock response is
              labelled engine="mock" so it can never be mistaken for Laya.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import time
from typing import Any, Dict, Union

State = Union[str, dict, list]


class LayaEngine:
    name = "laya"

    def __init__(self, mode: str = "router", model_id: str = "convaiinnovations/laya",
                 subfolder: str | None = None, device: str | None = None):
        import laya  # imported lazily so mock mode works without torch

        self.mode = mode
        if mode == "router":
            self._impl = laya.Router(preload=True, device=device) if device else laya.Router(preload=True)
        else:
            self._impl = laya.load(model_id, device=device, subfolder=subfolder)

    def predict(self, state: State, questions: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        out = self._impl.predict(state, questions)
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        out["engine"] = self.name
        return out


_SIGNALS = {
    "scam": ["unpaid", "fee", "verify your account", "gift card", "wire", "suspended", "click", "prize",
             "winner", "lottery", "urgent", "24h", "within 24", "password", "ssn", "crypto", "bit.ly", ".co/"],
    "attack": ["ignore all", "ignore previous", "system prompt", "jailbreak", "api key", "developer mode",
               "disregard", "pretend you"],
    "urgent": ["asap", "urgent", "today", "immediately", "down", "deadline", "now", "24h"],
    "angry": ["cancel", "refund", "terrible", "worst", "competitor", "unacceptable", "twice"],
    "spam": ["free", "link in bio", "subscribe", "check out my", "!!!", "promo", "discount"],
    "hard": ["prove", "architecture", "refactor", "design", "analyze", "step by step", "optimi", "derive"],
}


def _hits(text: str, key: str) -> int:
    t = text.lower()
    return sum(1 for w in _SIGNALS[key] if w in t)


def _squash(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class MockEngine:
    """Deterministic stand-in with Laya's response schema. NOT a model."""

    name = "mock"

    def predict(self, state: State, questions: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        text = state if isinstance(state, str) else " ".join(str(v) for v in (state.values() if isinstance(state, dict) else state))
        answers: Dict[str, Any] = {}
        for qid, q in questions.items():
            ins = (q.get("instructions") or "").lower()
            key = "scam"
            if any(w in ins for w in ["jailbreak", "injection", "ignore its rules", "aimed at the ai"]):
                key = "attack"
            elif any(w in ins for w in ["urgent", "time pressure", "deadline"]):
                key = "urgent"
            elif any(w in ins for w in ["cancel", "refund", "money back", "frustrat", "leave"]):
                key = "angry"
            elif any(w in ins for w in ["spam", "advertis", "toxic"]):
                key = "spam"
            elif any(w in ins for w in ["hard", "difficult"]):
                key = "hard"
            score = _hits(text, key)
            if q["type"] == "noul":
                p = round(_squash(1.4 * score - 2.0), 4)
                answers[qid] = {"type": "noul", "noul": p, "confidence": round(max(p, 1 - p), 4)}
            elif q["type"] == "score":
                crit = q.get("criteria") or []
                k = max(len(crit), 2)
                lvl = min(k - 1, score)
                probs = [0.6 if i == lvl else 0.4 / (k - 1) for i in range(k)]
                answers[qid] = {"type": "score", "score": float(lvl),
                                "legend": {str(i): c for i, c in enumerate(crit)},
                                "probabilities": {str(i): round(p, 4) for i, p in enumerate(probs)},
                                "confidence": 0.6}
            else:
                crit = q.get("criteria") or {}
                keys = list(crit.keys()) if isinstance(crit, dict) else list(crit)
                raw = []
                for k_ in keys:
                    desc = f"{k_} {crit.get(k_) or '' if isinstance(crit, dict) else ''}".lower()
                    words = [w for w in re.findall(r"[a-z]{4,}", desc)]
                    overlap = sum(1 for w in words if w in text.lower())
                    h = int(hashlib.md5((k_ + text).encode()).hexdigest(), 16) % 100 / 1000
                    raw.append(overlap + h)
                if any(w in ins for w in ["what should", "kind of message"]) and _hits(text, "scam") >= 2:
                    raw = [r + (2 if i == 0 else 0) for i, r in enumerate(raw)]
                ex = [math.exp(r) for r in raw]
                s = sum(ex)
                probs = [e / s for e in ex]
                best = max(range(len(keys)), key=lambda i: probs[i])
                answers[qid] = {"type": "choice", "choice": keys[best],
                                "probabilities": {k_: round(p, 4) for k_, p in zip(keys, probs)},
                                "confidence": round(probs[best], 4)}
        return {"model": "mock-heuristic", "engine": self.name, "answers": answers,
                "usage": {"input_tokens": len(text.split()), "output_tokens": 0},
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


def build_engine():
    """LEANROUTE_ENGINE=laya (default) | mock."""
    kind = os.getenv("LEANROUTE_ENGINE", "laya").lower()
    if kind == "mock":
        return MockEngine()
    return LayaEngine(
        mode=os.getenv("LAYA_MODE", "router"),
        model_id=os.getenv("LAYA_MODEL", "convaiinnovations/laya"),
        subfolder=os.getenv("LAYA_SUBFOLDER") or None,
        device=os.getenv("LAYA_DEVICE") or None,
    )
