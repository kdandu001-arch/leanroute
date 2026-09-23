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
from .gateway import Gateway
from .templates import TEMPLATES

MAX_CHARS = int(os.getenv("MAX_INPUT_CHARS", "4000"))
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "12"))


class DecideRequest(BaseModel):
    template: Optional[str] = Field(None, description="Template id, e.g. 'scam_check'")
    text: Optional[str] = Field(None, description="Plain text; placed under the template's state key")
    state: Optional[Union[Dict[str, Any], str]] = Field(None, description="Full state object (overrides text)")
    questions: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Custom typed questions")


def _api_keys() -> set:
    return {k.strip() for k in os.getenv("LEANROUTE_API_KEYS", "").split(",") if k.strip()}


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


def create_app(engine=None, gateway: Optional[Gateway] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        if os.getenv("PRELOAD", "true").lower() == "true":
            get_engine()
        yield

    app = FastAPI(title="Leanroute API", version="0.1.0", lifespan=lifespan,
                  description="Fast, calibrated decisions for AI apps. Powered by Laya (Apache 2.0).")
    origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    state: Dict[str, Any] = {"engine": engine, "gateway": gateway}
    limiter = RateLimiter(int(os.getenv("PLAYGROUND_RPM", "20")))
    public_playground = os.getenv("PUBLIC_PLAYGROUND", "true").lower() == "true"

    def get_engine():
        if state["engine"] is None:
            state["engine"] = build_engine()
        return state["engine"]

    def get_gateway() -> Gateway:
        if state["gateway"] is None:
            state["gateway"] = Gateway(get_engine())
        return state["gateway"]

    def auth(request: Request, authorization: Optional[str] = Header(None), allow_public: bool = False):
        keys = _api_keys()
        token = (authorization or "").removeprefix("Bearer ").strip()
        if not keys or token in keys:
            return
        if allow_public and public_playground and not token:
            limiter.check(request.client.host if request.client else "anon")
            return
        raise HTTPException(401, "Missing or invalid API key")

    def auth_public(request: Request, authorization: Optional[str] = Header(None)):
        auth(request, authorization, allow_public=True)

    def auth_private(request: Request, authorization: Optional[str] = Header(None)):
        auth(request, authorization, allow_public=False)

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

    @app.post("/v1/chat/completions", dependencies=[Depends(auth_private)])
    def chat(body: Dict[str, Any]):
        if body.get("stream"):
            raise HTTPException(400, "Streaming is not supported yet; send stream=false")
        try:
            return get_gateway().handle(body)
        except httpx.HTTPStatusError as e:
            raise HTTPException(e.response.status_code, f"Upstream LLM error: {e.response.text[:300]}")
        except httpx.HTTPError as e:
            raise HTTPException(502, f"Upstream LLM unreachable: {e}")

    @app.get("/v1/stats", dependencies=[Depends(auth_private)])
    def stats():
        return get_gateway().stats.snapshot()

    # Serve the website from the same server: http://localhost:8000/
    web_index = Path(os.getenv("WEB_DIR", Path(__file__).resolve().parents[2] / "web")) / "index.html"

    @app.get("/", include_in_schema=False)
    def home():
        if web_index.exists():
            return FileResponse(web_index)
        return {"ok": True, "docs": "/docs", "note": "web/index.html not found"}

    return app


app = create_app()
