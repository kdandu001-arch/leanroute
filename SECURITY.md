# Security

## Reporting a vulnerability

Please report security issues privately through GitHub:
**[Report a vulnerability](https://github.com/kdandu001-arch/leanroute/security/advisories/new)**.
Don't open a public issue for security problems. You'll get a reply within a few days.

## Scope notes

- **Guardrails are a first line of defence, not a guarantee.** The default prompt-injection detector wrongly
  blocks very few normal requests but misses many subtle attacks (see `eval/results.md`). Keep the usual
  protections in your application.
- When exposing a server publicly, set `LEANROUTE_API_KEYS`, restrict `CORS_ORIGINS`, and never commit `.env`.
- Leanroute stores counts and costs, not prompt text. With `CACHE_TTL_SECONDS` enabled, LLM responses are stored.
