"""Local usage log for `leanroute dashboard`: one row per request, never the prompt text.

Stored in ~/.leanroute/usage.db (override with LEANROUTE_DB; set LEANROUTE_RECORD=0 or
Leanroute(record_usage=False) to turn it off). Same columns as the server's usage log, except that costs
are empty when the app didn't pass prices to Leanroute(prices=...).
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB = Path.home() / ".leanroute" / "usage.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    ts                REAL NOT NULL,
    project           TEXT NOT NULL,
    route             TEXT NOT NULL,      -- blocked | cheap | strong | pinned | fallback
    model             TEXT,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd          REAL,               -- empty when no prices were given
    baseline_usd      REAL,               -- the same tokens on the strong model
    decision_ms       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS events_project_ts ON events (project, ts);
"""


def db_path(path: Optional[str] = None) -> str:
    return path or os.getenv("LEANROUTE_DB") or str(DEFAULT_DB)


class LocalUsage:
    def __init__(self, path: Optional[str] = None):
        path = db_path(path)
        if path != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def record(self, *, project: str, route: str, model: Optional[str], cost_usd: Optional[float],
               baseline_usd: Optional[float], decision_ms: float, prompt_tokens: int = 0,
               completion_tokens: int = 0, ts: Optional[float] = None):
        with self._lock, self._db:
            self._db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)",
                             (ts or time.time(), project, route, model, int(prompt_tokens), int(completion_tokens),
                              cost_usd, baseline_usd, float(decision_ms)))

    def _where(self, project: Optional[str], since: Optional[float]):
        clauses, args = [], []
        if project:
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
                "SELECT COUNT(*), SUM(route='blocked'), SUM(route='cheap'), SUM(route IN ('strong','fallback')), "
                "SUM(route='pinned'), COALESCE(SUM(cost_usd),0), COALESCE(SUM(baseline_usd),0), "
                "COALESCE(AVG(decision_ms),0), SUM(cost_usd IS NOT NULL AND baseline_usd IS NOT NULL) "
                f"FROM events{where}", args).fetchone()
        n, blocked, cheap, strong, pinned, actual, baseline, avg_ms, priced = row
        saved = baseline - actual
        return {"requests": n, "blocked": blocked or 0, "routed_cheap": cheap or 0, "routed_strong": strong or 0,
                "pinned": pinned or 0, "cached": 0, "priced_requests": priced or 0,
                "actual_cost_usd": round(actual, 6), "all_strong_cost_usd": round(baseline, 6),
                "saved_usd": round(saved, 6), "saved_pct": round(100 * saved / baseline, 1) if baseline else 0.0,
                "avg_decision_ms": round(avg_ms, 1)}

    def daily(self, project: Optional[str] = None, since: Optional[float] = None) -> List[Dict[str, Any]]:
        where, args = self._where(project, since)
        with self._lock:
            rows = self._db.execute(
                "SELECT date(ts, 'unixepoch') AS day, COUNT(*), SUM(route='blocked'), "
                "COALESCE(SUM(cost_usd),0), COALESCE(SUM(baseline_usd),0) "
                f"FROM events{where} GROUP BY day ORDER BY day", args).fetchall()
        return [{"date": d, "requests": n, "blocked": b or 0, "actual_cost_usd": round(a, 6),
                 "all_strong_cost_usd": round(base, 6), "saved_usd": round(base - a, 6)} for d, n, b, a, base in rows]

    def models(self, since: Optional[float] = None) -> Dict[str, Optional[str]]:
        """Most-used cheap and strong model names, for the dashboard's labels."""
        where, args = self._where(None, since)
        out: Dict[str, Optional[str]] = {"cheap": None, "strong": None}
        with self._lock:
            for route in out:
                row = self._db.execute(
                    f"SELECT model, COUNT(*) c FROM events{where}{' AND' if where else ' WHERE'} route = ? "
                    "AND model IS NOT NULL GROUP BY model ORDER BY c DESC LIMIT 1", args + [route]).fetchone()
                out[route] = row[0] if row else None
        return out

    def usage_report(self, days: int = 30, project: Optional[str] = None) -> Dict[str, Any]:
        """The same JSON shape as the server's GET /v1/usage, so both use one dashboard page."""
        days = max(1, min(int(days), 365))
        now = time.time()
        totals = self.snapshot(project, now - days * 86400)
        basis = min(days, 7)
        recent = self.snapshot(project, now - basis * 86400)["saved_usd"]
        models = self.models(now - days * 86400)
        return {"project": project or "all", "days": days, "totals": totals,
                "daily": self.daily(project, now - days * 86400),
                "quality": {"checked": 0, "passed": 0, "pass_rate": None, "cost_usd": 0.0},
                "net_saved_usd": totals["saved_usd"],
                "projected_monthly_saved_usd": round(recent / basis * 30, 6), "projection_basis_days": basis,
                "pricing": {"cheap_model": models["cheap"], "strong_model": models["strong"] or "the strong model",
                            "cheap_usd_per_1m": None, "strong_usd_per_1m": None},
                "source": "local"}
