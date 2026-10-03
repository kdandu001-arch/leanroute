# Changelog

## Unreleased

**Python package (`pip install leanroute`)**
- `leanroute mcp`: Leanroute as an MCP (Model Context Protocol) server with four tools: `route_prompt`, `guard_prompt`, `ask` and `usage_stats`. Install with `pip install "leanroute[mcp]"`; works with Claude Desktop, Claude Code, Cursor and other MCP clients.

## 0.3.0 (2026-09-29)

**Python package (`pip install leanroute`)**
- `leanroute dashboard`: the savings dashboard without running a server. The package now saves each request's route, model, tokens and cost (never prompt text) to `~/.leanroute/usage.db`, and the command opens the same dashboard page as the server, locally. Nothing extra to install.
- `leanroute stats` prints the totals as JSON.
- `Leanroute(project=...)` groups usage; `record_usage=False` or `LEANROUTE_RECORD=0` turns saving off; `LEANROUTE_DB` moves the file.

**Dashboard**
- Handles usage without prices: shows requests, routes and blocks, and asks for `prices=` instead of showing $0.

## 0.2.0 (2026-09-29)

**Python package (`pip install leanroute`)**
- Routing now uses Leanroute's trained router (in-process in local mode, via the server's `/v1/router` otherwise). On 10,000 held-out prompts with GPT-4-graded answers it sends about half of traffic to the cheap model with 94% of those answers good enough, vs about 89% with Laya's difficulty score. `Policy(router="laya")` keeps the old behaviour.
- Streaming through `lr.wrap(...)` records the real cost (OpenAI-style and Anthropic streams, sync and async); before, streamed calls were counted as $0.

**Server**
- Streaming (`stream: true`) through the OpenAI-compatible gateway.
- Retries for provider overload, 5xx and network errors; automatic fallback from the cheap to the strong model.
- Per-project monthly budgets (`LEANROUTE_BUDGETS`) and requests-per-minute (`PROJECT_RPM`), enforced before any LLM call.
- `X-Leanroute-Route/-Model/-Reason` response headers; `POST /v1/router` endpoint.
- Playground rate limit per visitor behind a proxy (`TRUST_PROXY`).

## 0.1.0 (2026-09-28)

First public release: OpenAI-compatible gateway with prompt-injection guard, trained router, savings dashboard, response cache and quality check; Python SDK; published evaluation.
