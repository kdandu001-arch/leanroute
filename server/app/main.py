"""Leanroute API — fast, calibrated decisions for AI apps, powered by Laya."""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Optional, Union

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .engine import build_engine
from .gateway import Gateway, GatewayConfig
from .templates import TEMPLATES
from .cache import ResponseCache
from .usage import UsageStore

MAX_CHARS = int(os.getenv("MAX_INPUT_CHARS", "4000"))
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "12"))


class DecideRequest(BaseModel):
    template: Optional[str] = Field(None, description="Template id, e.g. 'scam_check'")
    text: Optional[str] = Field(None, description="Plain text; placed under the template's state key")
    state: Optional[Union[Dict[str, Any], str]] = Field(None, description="Full state object (overrides text)")
    questions: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Custom typed questions")


class GuardRequest(BaseModel):
    text: str = Field(..., description="Text to check for prompt injection")


def _api_keys() -> Dict[str, str]:
    """Map each API key to its project. Entries are `project:key` or a bare `key` (project "default")."""
    keys: Dict[str, str] = {}
    for entry in os.getenv("LEANROUTE_API_KEYS", "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        project, sep, key = entry.partition(":")
        if sep and project.strip() and key.strip():
            keys[key.strip()] = project.strip()
        else:
            keys[entry] = "default"
    return keys


class RateLimiter:
    """Per-IP sliding window for anonymous Playground traffic."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: Dict[str, deque] = defaultdict(deque)

    def check(self, key: str):
        now = time.time()
        q = self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            raise HTTPException(429, "Playground rate limit reached. Get a free API key for more.")
        q.append(now)


def create_app(engine=None, gateway: Optional[Gateway] = None, store: Optional[UsageStore] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        if os.getenv("PRELOAD", "true").lower() == "true":
            get_engine()
            gw = get_gateway()  # also validates the gateway settings (e.g. GUARD_MODE) at startup
            if gw.cfg.guard_mode in ("precise", "broad"):
                gw.guard.score("warm up")
        yield

    app = FastAPI(title="Leanroute API", version="0.1.0", lifespan=lifespan,
                  description="Fast, calibrated decisions for AI apps. Powered by Laya (Apache 2.0).")
    origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    state: Dict[str, Any] = {"engine": engine, "gateway": gateway,
                             "store": store or (gateway.store if gateway else None)}
    limiter = RateLimiter(int(os.getenv("PLAYGROUND_RPM", "20")))
    public_playground = os.getenv("PUBLIC_PLAYGROUND", "true").lower() == "true"

    def get_engine():
        if state["engine"] is None:
            state["engine"] = build_engine()
        return state["engine"]

    def get_store() -> UsageStore:
        if state["store"] is None:
            state["store"] = UsageStore()
        return state["store"]

    def get_gateway() -> Gateway:
        if state["gateway"] is None:
            state["gateway"] = Gateway(get_engine(), store=get_store(), cache=ResponseCache())
        return state["gateway"]

    def auth(request: Request, authorization: Optional[str] = Header(None), allow_public: bool = False) -> Optional[str]:
        """Returns the caller's project, or None when the server has no keys configured (open mode)."""
        keys = _api_keys()
        token = (authorization or "").removeprefix("Bearer ").strip()
        if not keys:
            return None
        if token in keys:
            return keys[token]
        if allow_public and public_playground and not token:
            limiter.check(request.client.host if request.client else "anon")
            return None
        raise HTTPException(401, "Missing or invalid API key")

    def auth_public(request: Request, authorization: Optional[str] = Header(None)) -> Optional[str]:
        return auth(request, authorization, allow_public=True)

    def auth_private(request: Request, authorization: Optional[str] = Header(None)) -> Optional[str]:
        return auth(request, authorization, allow_public=False)

    @app.get("/health")
    def health():
        eng = state["engine"]
        return {"ok": True, "engine": getattr(eng, "name", None), "loaded": eng is not None}

    @app.get("/v1/templates")
    def templates():
        return {tid: {k: v for k, v in t.items()} for tid, t in TEMPLATES.items()}

    @app.post("/v1/decide", dependencies=[Depends(auth_public)])
    def decide(req: DecideRequest):
        tpl = TEMPLATES.get(req.template) if req.template else None
        if req.template and not tpl:
            raise HTTPException(404, f"Unknown template '{req.template}'")
        questions = req.questions or (tpl["questions"] if tpl else None)
        if not questions:
            raise HTTPException(422, "Provide 'questions' or a 'template'")
        if len(questions) > MAX_QUESTIONS:
            raise HTTPException(422, f"At most {MAX_QUESTIONS} questions per call")
        for qid, q in questions.items():
            if q.get("type") not in ("choice", "score", "noul") or not q.get("instructions"):
                raise HTTPException(422, f"Question '{qid}' needs type (choice|score|noul) and instructions")
            if q["type"] in ("choice", "score") and not q.get("criteria"):
                raise HTTPException(422, f"Question '{qid}' of type {q['type']} needs criteria")
        if req.state is not None:
            st = req.state
        elif req.text is not None:
            st = {(tpl or {}).get("state_key", "text"): req.text}
        else:
            raise HTTPException(422, "Provide 'text' or 'state'")
        if len(str(st)) > MAX_CHARS:
            raise HTTPException(413, f"Input longer than {MAX_CHARS} characters")
        return get_engine().predict(st, questions)

    @app.post("/v1/guard", dependencies=[Depends(auth_public)])
    def guard(req: GuardRequest):
        """Prompt-injection probability from the gateway's detector (used by the SDK's precise/broad guard)."""
        if len(req.text) > MAX_CHARS:
            raise HTTPException(413, f"Input longer than {MAX_CHARS} characters")
        t0 = time.perf_counter()
        score = get_gateway().guard.score(req.text)
        return {"injection": round(score, 5), "detector": getattr(get_gateway().guard, "name", "protectai"),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    @app.post("/v1/chat/completions")
    def chat(body: Dict[str, Any], project: Optional[str] = Depends(auth_private)):
        if body.get("stream"):
            raise HTTPException(400, "Streaming is not supported yet; send stream=false")
        try:
            return get_gateway().handle(body, project=project or "default")
        except httpx.HTTPStatusError as e:
            raise HTTPException(e.response.status_code, f"Upstream LLM error: {e.response.text[:300]}")
        except httpx.HTTPError as e:
            raise HTTPException(502, f"Upstream LLM unreachable: {e}")

    @app.get("/v1/stats")
    def stats(project: Optional[str] = Depends(auth_private)):
        """All-time totals. With API keys configured, each key sees only its own project."""
        return get_store().snapshot(project=project)

    @app.get("/v1/usage")
    def usage(days: int = 30, project: Optional[str] = Depends(auth_private)):
        """Everything the savings dashboard shows, for the last `days` days."""
        days = max(1, min(days, 365))
        since = time.time() - days * 86400
        st = get_store()
        totals = st.snapshot(project=project, since=since)
        quality = st.quality(project=project, since=since)
        basis = min(days, 7)
        recent_since = time.time() - basis * 86400
        recent_net = st.snapshot(project=project, since=recent_since)["saved_usd"] - st.quality(project=project, since=recent_since)["cost_usd"]
        cfg = state["gateway"].cfg if state["gateway"] else GatewayConfig()
        return {
            "project": project or "all",
            "days": days,
            "totals": totals,
            "daily": st.daily(project=project, since=since),
            "quality": quality,
            "net_saved_usd": round(totals["saved_usd"] - quality["cost_usd"], 6),
            "projected_monthly_saved_usd": round(recent_net / basis * 30, 6),
            "projection_basis_days": basis,
            "pricing": {"cheap_model": cfg.cheap_model, "strong_model": cfg.strong_model,
                        "cheap_usd_per_1m": [cfg.cheap_in, cfg.cheap_out],
                        "strong_usd_per_1m": [cfg.strong_in, cfg.strong_out]},
        }

    # Serve the website and dashboard from the same server: http://localhost:8000/
    web_dir = Path(os.getenv("WEB_DIR", Path(__file__).resolve().parents[2] / "web"))

    def page(name: str):
        f = web_dir / name
        if f.exists():
            return FileResponse(f)
        return {"ok": True, "docs": "/docs", "note": f"web/{name} not found"}

    @app.get("/", include_in_schema=False)
    def home():
        return page("index.html")

    @app.get("/dashboard", include_in_schema=False)
    def dashboard():
        return page("dashboard.html")

    return app


app = create_app()
