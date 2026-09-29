"""`leanroute dashboard`: the savings dashboard for apps that use the pip package, no server needed.

Serves the same page as the Leanroute server's /dashboard, reading the local usage log
(~/.leanroute/usage.db). Listens on 127.0.0.1 only, so it's reachable from this computer alone.
"""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .usage import LocalUsage

PAGE = Path(__file__).resolve().parent / "dashboard.html"


def make_server(port: int = 8765, db: Optional[str] = None, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    usage = LocalUsage(db)
    page = PAGE.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, ctype: str):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ("/", "/dashboard"):
                return self._send(200, page, "text/html; charset=utf-8")
            if url.path == "/v1/usage":
                q = parse_qs(url.query)
                try:
                    days = int(q.get("days", ["30"])[0])
                except ValueError:
                    days = 30
                report = usage.usage_report(days=days, project=q.get("project", [None])[0])
                return self._send(200, json.dumps(report).encode(), "application/json")
            self._send(404, b"not found", "text/plain")

        def log_message(self, *args):  # keep the terminal quiet
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve(port: int = 8765, db: Optional[str] = None, open_browser: bool = True):
    server = make_server(port, db)
    url = f"http://127.0.0.1:{server.server_address[1]}/dashboard"
    print(f"Leanroute dashboard: {url}   (usage log: {LocalUsage(db).path})\nPress Ctrl+C to stop.", flush=True)
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
