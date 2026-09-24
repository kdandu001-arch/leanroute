"""Where decisions are computed.

LocalEngine   runs Laya (and the prompt-injection detector) inside your process (pip install "leanroute[local]").
RemoteEngine  calls a Leanroute server (self-hosted or hosted) over HTTP.

Both return Laya's response shape:
{"answers": {qid: {"type": ..., "noul"|"choice"|"score": ..., "confidence": ...}}, ...}
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Protocol, Union

import httpx

State = Union[str, dict, list]
DETECTOR = "protectai/deberta-v3-base-prompt-injection-v2"


class Engine(Protocol):
    name: str

    def predict(self, state: State, questions: Dict[str, Any]) -> Dict[str, Any]: ...


class LocalEngine:
    name = "laya-local"

    def __init__(self, model: Optional[str] = None, device: Optional[str] = None, router: bool = True):
        try:
            import laya
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "Local mode needs the Laya package. Install it with:  pip install \"leanroute[local]\"\n"
                "Or point Leanroute at a server:  Leanroute(api_url=\"http://localhost:8000\")"
            ) from e
        self._detector = None
        kw = {"device": device} if device else {}
        if router and model is None:
            self._impl = laya.Router(preload=True, **kw)
        else:
            self._impl = laya.load(model or "convaiinnovations/laya", **kw)

    def injection_score(self, text: str) -> float:
        """P(prompt injection) from ProtectAI's open-source detector (Apache-2.0), loaded on first use."""
        if self._detector is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(DETECTOR)
            model = AutoModelForSequenceClassification.from_pretrained(DETECTOR).eval()
            label = [i for i, n in model.config.id2label.items() if n.upper() == "INJECTION"][0]
            self._detector = (tok, model, label, torch)
        tok, model, label, torch = self._detector
        with torch.inference_mode():
            enc = tok(text, truncation=True, max_length=512, return_tensors="pt")
            return float(model(**enc).logits.softmax(-1)[0, label])

    def predict(self, state, questions):
        t0 = time.perf_counter()
        out = self._impl.predict(state, questions)
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        out["engine"] = self.name
        return out


class RemoteEngine:
    name = "leanroute-remote"

    def __init__(self, api_url: str, api_key: Optional[str] = None, timeout: float = 10.0,
                 client: Optional[httpx.Client] = None):
        self.url = api_url.rstrip("/") + "/v1/decide"
        self.guard_url = api_url.rstrip("/") + "/v1/guard"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = client or httpx.Client(timeout=timeout, headers=headers)

    def predict(self, state, questions):
        body = {"state": state if not isinstance(state, list) else {"turns": state}, "questions": questions}
        r = self.http.post(self.url, json=body)
        r.raise_for_status()
        return r.json()

    def injection_score(self, text: str) -> float:
        r = self.http.post(self.guard_url, json={"text": text})
        r.raise_for_status()
        return float(r.json()["injection"])
