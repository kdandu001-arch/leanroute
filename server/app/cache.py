"""Exact-match response cache: identical requests are answered from storage for $0.

Off unless CACHE_TTL_SECONDS > 0. Requests are keyed by a SHA-256 hash of the project and the full
request body, so prompts are never stored in readable form. Responses ARE stored while caching is on.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .usage import DEFAULT_DB

_SCHEMA = """
CREATE TABLE IF NOT EXISTS response_cache (
    key          TEXT PRIMARY KEY,
    created      REAL NOT NULL,
    response     TEXT NOT NULL,
    baseline_usd REAL NOT NULL      -- what this request cost on the strong model, i.e. what a hit saves
);
"""


def request_key(project: str, body: Dict[str, Any]) -> str:
    body = {k: v for k, v in body.items() if k not in ("stream", "user")}
    raw = json.dumps({"project": project, "body": body}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


class ResponseCache:
    def __init__(self, ttl_seconds: Optional[float] = None, path: Optional[str] = None):
        self.ttl = float(os.getenv("CACHE_TTL_SECONDS", "0") or 0) if ttl_seconds is None else ttl_seconds
        path = path or os.getenv("LEANROUTE_DB") or str(DEFAULT_DB)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.ttl > 0

    def get(self, key: str) -> Optional[Tuple[Dict[str, Any], float]]:
        if not self.enabled:
            return None
        with self._lock:
            row = self._db.execute("SELECT created, response, baseline_usd FROM response_cache WHERE key = ?",
                                   (key,)).fetchone()
        if not row or time.time() - row[0] > self.ttl:
            return None
        return json.loads(row[1]), row[2]

    def put(self, key: str, response: Dict[str, Any], baseline_usd: float):
        if not self.enabled:
            return
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO response_cache VALUES (?,?,?,?)",
                             (key, time.time(), json.dumps(response, ensure_ascii=False), baseline_usd))
            self._db.execute("DELETE FROM response_cache WHERE created < ?", (time.time() - self.ttl,))
