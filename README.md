# Leanroute

**Fast, calibrated decisions for AI apps, powered by [Laya](https://huggingface.co/convaiinnovations/laya).**

An LLM is an expensive expert. Many requests an AI app sends it are simple decisions: *spam or not? which team? safe to send? cheap model or big model?* Leanroute puts Laya, a small open-source decision model, in front of your LLM so those get answered in milliseconds for $0, and only the hard questions reach the paid model.

```
Your app ─► Leanroute (Laya) ─┬─► answered by Laya ........ $0
                              ├─► cheap model (easy) ...... $
                              ├─► strong model (hard) ..... $$$
                              └─► blocked (jailbreak) ..... $0
```

## What's in the repo

| Folder | What it is |
|---|---|
| `sdk/` | **The `leanroute` Python package.** Install it in any project to put Laya in front of your LLM (`pip install leanroute`). |
| `server/` | FastAPI API: `/v1/decide`, `/v1/templates`, OpenAI-compatible `/v1/chat/completions` gateway, `/v1/stats`. Also serves the website at `/`. |
| `web/` | The website: landing page, live Playground, savings calculator, docs, pricing. One static file. |
| `examples/` | Drop-in OpenAI example, any-LLM routing, no-LLM decisions. |

## Use it in any project (SDK)

```bash
pip install "leanroute[local]"        # Laya runs inside your app
# or: pip install leanroute           # and point it at a Leanroute server
```

```python
from openai import OpenAI
from leanroute import Leanroute

lr = Leanroute()                                       # or Leanroute(api_url="http://localhost:8000")
client = lr.wrap(OpenAI(), cheap="gpt-4o-mini", strong="gpt-4o")
r = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "Hi!"}])
print(r.leanroute.route)                               # "cheap" | "strong"   (blocked prompts raise leanroute.Blocked)
```

Full SDK docs: [`sdk/README.md`](sdk/README.md). Until it's on PyPI, install from the repo: `pip install -e ./sdk`.

## Run it on your Mac

```bash
cd server
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # edit values if you like
uvicorn app.main:app --port 8000 --env-file .env   # --env-file loads your settings from .env
```

The first start downloads the Laya weights from Hugging Face (about 1-2 GB) and preloads them. Then open `http://localhost:8000/docs` for the interactive API docs.

Want the UI without the model? `LEANROUTE_ENGINE=mock uvicorn app.main:app --port 8000 --env-file .env` runs a keyword stand-in that returns the same response shape. Its responses are labelled `"engine": "mock"`; it is **not** Laya and its numbers mean nothing.

Then open **http://localhost:8000**: the server also serves the website, and the Playground connects to it automatically (green **Live API** badge).

Run the tests: `cd server && pytest -q` and `cd sdk && pytest -q`

## API

### `POST /v1/decide`

```bash
curl localhost:8000/v1/decide -H 'Content-Type: application/json' -d '{
  "template": "scam_check",
  "text": "USPS: unpaid $1.99 fee, pay within 24h: usps-redelivery-help.co/pay"
}'
```

Or bring your own questions:

```json
{
  "text": "Refund the duplicate charge or we cancel.",
  "questions": {
    "churn_risk": {"type": "noul", "instructions": "Might the customer cancel?"},
    "intent": {"type": "choice", "instructions": "What do they want?",
               "criteria": {"refund": "money back", "help": "support", "other": "anything else"}}
  }
}
```

Question types: `noul` (yes/no probability), `choice` (one of a few options), `score` (a level on a scale).
Templates: `scam_check`, `ticket_routing`, `email_triage`, `lead_score`, `guardrails`, `moderation`, `model_router`.

### `POST /v1/chat/completions` (LLM Cost Cutter)

Drop-in OpenAI-compatible endpoint. Send `model: "auto"` and Leanroute:

1. blocks prompt-injection attempts before any LLM is paid (no cost). The default guard (`GUARD_MODE=precise`) uses ProtectAI's open-source injection detector, which in testing almost never blocked a normal request; `broad` also uses Laya to catch more role-play jailbreaks, at the price of blocking some coding requests (see [`eval/results.md`](eval/results.md)),
2. asks Laya how hard and how sensitive the request is. Optional `ROUTER=routellm` uses RouteLLM's learned router instead, which separated easy from hard far better in our evaluation (87% of easy questions to the cheap model vs. 33% with Laya; see the licensing note in `server/app/router_model.py`), while Laya still sends sensitive requests to the strong model,
3. sends easy, non-sensitive requests to `CHEAP_MODEL`,
4. sends everything else to `STRONG_MODEL`.

The response includes a `leanroute` block (`route`, `reason`, `cost_usd`, `saved_usd`). `GET /v1/stats` shows totals and % saved versus sending everything to the strong model. Pin a specific model name to skip routing and keep only the guardrail. Streaming isn't supported yet.

### Response cache

Set `CACHE_TTL_SECONDS` (e.g. `3600`) and identical requests (same project, same messages and settings) are answered from storage for $0, skipping both Laya and the LLM. Off by default, since some apps want a fresh answer every time. **While it's on, LLM responses are stored**; prompts are stored only as a one-way hash. Blocked requests are never cached.

### Quality check: proof the savings are real

Most routers claim savings but never check whether the cheap model's answers were good enough. Set `QUALITY_CHECK_RATE=0.05` and Leanroute re-asks 5% of cheap-routed requests to the strong model in the background, then has the strong model judge whether the cheap answer was as good. The dashboard shows the pass rate (e.g. "91% of 271 sampled cheap answers were judged as good as gpt-4o's") and **subtracts what the checks cost from your savings**. Only pass/fail and cost are stored. The judge is the strong model itself, so treat the pass rate as a strong signal, not proof.

### Savings dashboard (Leanroute Ledger)

Open **http://localhost:8000/dashboard** to see what the gateway saved you: money saved, % saved, what it would have cost on the strong model, monthly pace, routing split, attacks blocked, and a per-day chart.

* Usage is saved to a small SQLite file (`server/data/leanroute.db`, or `LEANROUTE_DB`), so numbers survive restarts. **Only counts and costs are stored, never prompt text.**
* Give each customer their own dashboard with project-named keys: `LEANROUTE_API_KEYS=acme:lr_abc,beta:lr_def`. Each key sees only its own project.
* The same data is available as JSON: `GET /v1/usage?days=30` (and all-time totals at `GET /v1/stats`).

Works with any OpenAI-compatible upstream (OpenAI, OpenRouter, Groq, Together, a local Ollama at `http://localhost:11434/v1`). **Set the price variables in `.env` to your provider's current prices**; savings are only as accurate as those numbers.

## Deploy for $0

**API: Oracle Cloud Always Free (ARM VM, 2 OCPU / 12 GB RAM as of Aug 2026).**
1. Create an Ubuntu Ampere A1 instance, open port 8000 (or put Caddy in front for HTTPS).
2. Install Docker, clone this repo, then:
   ```bash
   cd server && cp .env.example .env   # set LEANROUTE_API_KEYS and CORS_ORIGINS!
   docker compose up -d --build
   ```
3. Expect a few hundred ms per decision on CPU. That's fine for a Playground and moderate traffic. Move to a GPU when you need ~35 ms.

Alternative for demos: run on your Mac and expose it with a free Cloudflare Tunnel (`cloudflared tunnel --url http://localhost:8000`). It's only online while your Mac is.

**Website: Cloudflare Pages (free, commercial use allowed).**
1. Set `window.LEANROUTE_API` in `web/index.html` to your API URL.
2. Cloudflare dashboard → Pages → connect the GitHub repo → build output directory `web`, no build command.
3. You get `https://<name>.pages.dev`. Add a custom domain later if you want.

**Before going public:** set `LEANROUTE_API_KEYS`, lock `CORS_ORIGINS` to your site, keep `PLAYGROUND_RPM` low.

## Honest limits

* Laya **decides**; it doesn't write. Summaries, answers and chat stay with the LLM.
* Short text only: roughly a page per call (`MAX_INPUT_CHARS`, default 4,000).
* Keep choice questions to a handful of options.
* **Measured accuracy, honestly.** On held-out labelled prompts from public datasets ([`eval/results.md`](eval/results.md)): the default guard wrongly blocks **0.4%** of normal requests and **no coding or math requests**, but catches only about **36%** of subtle injection attacks (many in German; the detector is English-only). The guard is a first line of defence, not a complete one. Laya's easy-vs-hard routing is weak (AUC 0.67); RouteLLM's router (`ROUTER=routellm`) scored 0.99 on the same test. Re-run the evaluation yourself: `python eval/build_dataset.py && python eval/score.py && python eval/score_protectai.py && python eval/score_routellm.py && python eval/report.py`.
* Base checkpoints are weak zero-shot on niche domains and ship over-confident. For production, **fine-tune on your own labelled examples and fit a temperature** (see the Laya model card). Start with conservative thresholds and keep a human in the loop for high-stakes actions.

## Roadmap

- [ ] Hosted Leanroute Pro ($19/month): no server to run
- [ ] API keys + usage in Supabase, Stripe billing
- [ ] Streaming in the gateway
- [x] Response cache for repeated prompts
- [x] Quality check: verify a sample of cheap answers against the strong model
- [ ] Fine-tune Studio: upload CSV → custom calibrated model
- [x] Savings dashboard page
- [ ] Dashboard for SDK users (SDK reports usage to a server)
- [ ] LangChain / CrewAI / MCP plug-ins

## License

Leanroute is free and open source under Apache 2.0, © 2026 Kushal. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Built on Laya (weights and package: Apache 2.0, © Convai Innovations), which is downloaded at install time and not included in this repo.
