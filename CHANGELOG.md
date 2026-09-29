# Changelog

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
