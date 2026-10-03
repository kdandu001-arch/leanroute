"""Command line: `leanroute dashboard`, `leanroute stats`, `leanroute mcp`."""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__


def main(argv=None):
    ap = argparse.ArgumentParser(prog="leanroute", description="Leanroute: a toll gate in front of any LLM.")
    ap.add_argument("--version", action="version", version=f"leanroute {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    d = sub.add_parser("dashboard", help="open the savings dashboard for this computer's usage")
    d.add_argument("--port", type=int, default=8765)
    d.add_argument("--db", help="usage log path (default: ~/.leanroute/usage.db or $LEANROUTE_DB)")
    d.add_argument("--no-browser", action="store_true", help="don't open a browser tab")

    s = sub.add_parser("stats", help="print usage totals as JSON")
    s.add_argument("--days", type=int, default=30)
    s.add_argument("--db")

    m = sub.add_parser("mcp", help="run Leanroute as an MCP server (needs: pip install \"leanroute[mcp]\")")
    m.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")

    args = ap.parse_args(argv)
    if args.cmd == "dashboard":
        from .dashboard import serve
        serve(port=args.port, db=args.db, open_browser=not args.no_browser)
    elif args.cmd == "stats":
        from .usage import LocalUsage
        report = LocalUsage(args.db).usage_report(days=args.days)
        print(json.dumps({"days": report["days"], **report["totals"]}, indent=2))
    elif args.cmd == "mcp":
        from .mcp_server import serve
        serve(args.transport)
    else:
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
