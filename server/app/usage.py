"""Persistent usage log for the gateway: one row per request, never the prompt text."""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "leanroute.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    ts                REAL NOT NULL,
    project           TEXT NOT NULL,
    route             TEXT NOT NULL,      -- blocked | cheap | strong | pinned | cached
    model             TEXT,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd          REAL NOT NULL,
    baseline_usd      REAL NOT NULL,      -- what the same request costs on the strong model
    decision_ms       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS events_project_ts ON events (project, ts);
"""


class UsageStore:
    def __init__(self, path: Optional[str] = None):
        path = path or os.getenv("LEANROUTE_DB") or str(DEFAULT_DB)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def record(self, *, project: str, route: str, model: Optional[str], cost_usd: float, baseline_usd: float,
               decision_ms: float, prompt_tokens: int = 0, completion_tokens: int = 0, ts: Optional[float] = None):
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)",
                (ts or time.time(), project, route, model, int(prompt_tokens), int(completion_tokens),
                 float(cost_usd), float(baseline_usd), float(decision_ms)))

    def _where(self, project: Optional[str], since: Optional[float]):
        clauses, args = [], []
        if project is not None:
            clauses.append("project = ?")
            args.append(project)
        if since is not None:
            clauses.append("ts >= ?")
            args.append(since)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", args

    def snapshot(self, project: Optional[str] = None, since: Optional[float] = None) -> Dict[str, Any]:
        where, args = self._where(project, since)
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*), "
                "SUM(route='blocked'), SUM(route='cheap'), SUM(route='strong'), SUM(route='pinned'), SUM(route='cached'), "
                "COALESCE(SUM(cost_usd),0), COALESCE(SUM(baseline_usd),0), COALESCE(AVG(CASE WHEN route != 'cached' THEN decision_ms END),0) "
                f"FROM events{where}", args).fetchone()
        n, blocked, cheap, strong, pinned, cached, actual, baseline, avg_ms = row
        saved = baseline - actual
        return {
            "requests": n,
            "blocked": blocked or 0,
            "routed_cheap": cheap or 0,
            "routed_strong": strong or 0,
            "pinned": pinned or 0,
            "cached": cached or 0,
            "actual_cost_usd": round(actual, 6),
            "all_strong_cost_usd": round(baseline, 6),
            "saved_usd": round(saved, 6),
            "saved_pct": round(100 * saved / baseline, 1) if baseline else 0.0,
            "avg_decision_ms": round(avg_ms, 1),
        }

    def daily(self, project: Optional[str] = None, since: Optional[float] = None) -> List[Dict[str, Any]]:
        where, args = self._where(project, since)
        with self._lock:
            rows = self._db.execute(
                "SELECT date(ts, 'unixepoch') AS day, COUNT(*), SUM(route='blocked'), "
                "COALESCE(SUM(cost_usd),0), COALESCE(SUM(baseline_usd),0) "
                f"FROM events{where} GROUP BY day ORDER BY day", args).fetchall()
        return [{"date": d, "requests": n, "blocked": b or 0, "actual_cost_usd": round(a, 6),
                 "all_strong_cost_usd": round(base, 6), "saved_usd": round(base - a, 6)}
                for d, n, b, a, base in rows]
