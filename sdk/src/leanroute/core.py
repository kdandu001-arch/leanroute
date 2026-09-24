"""Leanroute: a decision layer that sits in front of any LLM."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .engines import Engine, LocalEngine, RemoteEngine
from .questions import GUARD_QUESTIONS, ROUTER_QUESTIONS, yes_no

Messages = List[Dict[str, Any]]


class Blocked(Exception):
    """Raised when a prompt is blocked by the guardrail (jailbreak / prompt injection)."""

    def __init__(self, decision: "Decision"):
        super().__init__(f"Blocked by Leanroute: {decision.reason}")
        self.decision = decision


@dataclass
class Answer:
    type: str                      # "noul" (yes/no) | "choice" | "score" (level)
    value: Any                     # probability | chosen option | expected level
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)

    @property
    def yes(self) -> bool:
        """For yes/no questions: True when P(yes) >= 0.5."""
        return self.type == "noul" and float(self.value) >= 0.5

    def __repr__(self):
        return f"Answer({self.type}={self.value!r}, confidence={self.confidence:.2f})"


def _to_answers(raw: Dict[str, Any]) -> Dict[str, Answer]:
    out = {}
    for qid, a in (raw.get("answers") or {}).items():
        t = a.get("type")
        val = a.get("noul") if t == "noul" else a.get("choice") if t == "choice" else a.get("score")
        out[qid] = Answer(t, val, float(a.get("confidence", 0.0)), a.get("probabilities") or {})
    return out


@dataclass
class Decision:
    route: str                     # "blocked" | "cheap" | "strong" | "pinned" | "fallback"
    model: Optional[str]
    reason: str
    scores: Dict[str, float] = field(default_factory=dict)
    decision_ms: float = 0.0

    @property
    def blocked(self) -> bool:
        return self.route == "blocked"


@dataclass
class Policy:
    # Guard modes (same as the server's GUARD_MODE; measured in eval/results.md):
    #   precise  prompt-injection detector only: fewest normal requests blocked (default)
    #   broad    detector, plus Laya when its jailbreak and injection scores are both very high
    #   laya     Laya's jailbreak and injection scores only
    #   off      no guard
    # Engines without a detector (custom engines) fall back to "laya" for precise/broad.
    guard_mode: str = "precise"
    detector_threshold: float = 0.66  # block when P(injection) from the detector >= this
    laya_threshold: Optional[float] = None  # Laya's part: None = 0.99 in broad mode, 0.92 in laya mode
    easy_max: float = 1.2           # difficulty (0-3) at or below -> cheap model
    min_confidence: float = 0.0     # optional: require this confidence before trusting "easy". Off by default: Laya's difficulty confidence is low for every prompt, so it doesn't separate easy from hard
    sensitive_max: float = 0.5      # money/legal/medical/safety -> strong model

    def __post_init__(self):
        if self.guard_mode not in ("precise", "broad", "laya", "off"):
            raise ValueError(f"guard_mode must be precise, broad, laya or off (got {self.guard_mode!r})")


class Stats:
    """Counts routes and (if you give prices) estimates money saved vs. always using the strong model."""

    def __init__(self, prices: Optional[Dict[str, Tuple[float, float]]] = None):
        self.prices = prices or {}   # model -> (USD per 1M input tokens, USD per 1M output tokens)
        self._lock = threading.Lock()
        self.routes: Dict[str, int] = {}
        self.actual_usd = 0.0
        self.baseline_usd = 0.0
        self.decision_ms = 0.0

    def _cost(self, model, pin, pout):
        p = self.prices.get(model)
        return None if p is None else (pin * p[0] + pout * p[1]) / 1e6

    def record(self, d: Decision, strong_model: Optional[str], pin: int = 0, pout: int = 0):
        with self._lock:
            self.routes[d.route] = self.routes.get(d.route, 0) + 1
            self.decision_ms += d.decision_ms
            base = self._cost(strong_model, pin, pout) if strong_model else None
            act = 0.0 if d.route == "blocked" else self._cost(d.model, pin, pout)
            if base is not None and act is not None:
                self.baseline_usd += base
                self.actual_usd += act

    def summary(self) -> Dict[str, Any]:
        n = sum(self.routes.values())
        saved = self.baseline_usd - self.actual_usd
        return {
            "requests": n,
            "routes": dict(self.routes),
            "actual_usd": round(self.actual_usd, 6),
            "all_strong_usd": round(self.baseline_usd, 6),
            "saved_usd": round(saved, 6),
            "saved_pct": round(100 * saved / self.baseline_usd, 1) if self.baseline_usd else None,
            "avg_decision_ms": round(self.decision_ms / n, 1) if n else 0.0,
        }


def text_of(prompt: Union[str, Messages]) -> str:
    """Last user message from an OpenAI/Anthropic-style messages list, or the string itself."""
    if isinstance(prompt, str):
        return prompt
    for m in reversed(prompt or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


class Leanroute:
    """
    lr = Leanroute()                                   # Laya in-process  (pip install "leanroute[local]")
    lr = Leanroute(api_url="http://localhost:8000")    # or a Leanroute server

    lr.check("FREE crypto!!!", "Is this spam?")        -> 0.94
    lr.decide(text, {"team": choice("Which team?", ["billing", "tech", "sales"])})
    lr.route(messages, cheap="small-model", strong="big-model")  -> Decision
    client = lr.wrap(OpenAI(), cheap="...", strong="...")         # drop-in, model="auto"
    """

    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None,
                 engine: Optional[Engine] = None, policy: Optional[Policy] = None,
                 prices: Optional[Dict[str, Tuple[float, float]]] = None,
                 fail_open: bool = True, max_chars: int = 4000, **local_kwargs):
        if engine is not None:
            self.engine = engine
        elif api_url:
            self.engine = RemoteEngine(api_url, api_key)
        else:
            self.engine = LocalEngine(**local_kwargs)
        self.policy = policy or Policy()
        self.stats = Stats(prices)
        self.fail_open = fail_open
        self.max_chars = max_chars
        self.last: Optional[Decision] = None

    # ---- typed decisions -------------------------------------------------
    def decide(self, text: Union[str, dict], questions: Dict[str, dict], key: str = "text") -> Dict[str, Answer]:
        state = text if isinstance(text, dict) else {key: text[: self.max_chars]}
        return _to_answers(self.engine.predict(state, questions))

    def check(self, text: str, question: str) -> float:
        """One yes/no question -> probability of yes."""
        return float(self.decide(text, {"q": yes_no(question)})["q"].value)

    # ---- routing + guardrails -------------------------------------------
    def route(self, prompt: Union[str, Messages], cheap: Optional[str] = None,
              strong: Optional[str] = None) -> Decision:
        text = text_of(prompt)[: self.max_chars]
        t0 = time.perf_counter()
        p = self.policy
        scores: Dict[str, Any] = {}
        try:
            why_blocked = self._guard(text, scores)
            if why_blocked:
                d = Decision("blocked", None, why_blocked, scores, (time.perf_counter() - t0) * 1000)
                self.last = d
                return d
            a = _to_answers(self.engine.predict({"request": text}, ROUTER_QUESTIONS))
        except Exception as e:
            if not self.fail_open:
                raise
            d = Decision("fallback", strong, f"decision layer unavailable ({type(e).__name__}); using strong model")
            self.last = d
            return d
        ms = (time.perf_counter() - t0) * 1000
        diff, sens = a["r_difficulty"], float(a["r_sensitive"].value)
        scores.update({"difficulty": float(diff.value), "difficulty_confidence": diff.confidence, "sensitive": sens})
        if float(diff.value) <= p.easy_max and diff.confidence >= p.min_confidence and sens < p.sensitive_max:
            d = Decision("cheap", cheap, f"easy (difficulty {float(diff.value):.2f}, conf {diff.confidence:.2f})", scores, ms)
        else:
            why = "sensitive" if sens >= p.sensitive_max else f"difficulty {float(diff.value):.2f}, conf {diff.confidence:.2f}"
            d = Decision("strong", strong, why, scores, ms)
        self.last = d
        return d

    def _guard(self, text: str, scores: Dict[str, Any]) -> Optional[str]:
        """Returns why `text` should be blocked, or None. Records the scores it used."""
        p = self.policy
        mode = p.guard_mode
        detector = getattr(self.engine, "injection_score", None)
        if mode in ("precise", "broad") and detector is None:
            mode = "laya"
        if mode in ("precise", "broad"):
            s = float(detector(text))
            scores["injection_detector"] = s
            if s >= p.detector_threshold:
                return f"injection detector={s:.2f}"
        if mode in ("broad", "laya"):
            g = _to_answers(self.engine.predict({"prompt": text}, GUARD_QUESTIONS))
            jb, inj = float(g["g_jailbreak"].value), float(g["g_injection"].value)
            scores.update({"jailbreak": jb, "injection": inj})
            if min(jb, inj) >= (p.laya_threshold or (0.99 if mode == "broad" else 0.92)):
                return f"jailbreak={jb:.2f} injection={inj:.2f}"
        return None

    def guard(self, prompt: Union[str, Messages]) -> Decision:
        """Raise Blocked for jailbreak / injection attempts; otherwise return the decision."""
        d = self.route(prompt)
        if d.blocked:
            raise Blocked(d)
        return d

    # ---- drop-in client wrapper -----------------------------------------
    def wrap(self, client: Any, cheap: str, strong: str, on_block: str = "raise"):
        """Wrap an OpenAI-style or Anthropic client. Send model="auto" to let Leanroute choose."""
        from .wrap import wrap_client
        return wrap_client(self, client, cheap, strong, on_block)
