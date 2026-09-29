# leanroute

**A toll gate in front of any LLM.** Website: [leanroute.online](https://leanroute.online) · Source: [GitHub](https://github.com/kdandu001-arch/leanroute)

Before your app pays an LLM, Leanroute checks three things, locally and for $0 per call:

1. **Is this an attack?** Prompt injections are blocked by an open-source detector before any LLM call.
2. **How hard is it?** Easy → your cheap model. Hard or sensitive → your strong model.
3. **Anything else you ask**, e.g. "is this spam?", "which team?", "hot lead?" → answered by [Laya](https://huggingface.co/convaiinnovations/laya) (Apache 2.0) with a probability, no LLM at all.

"Easy or hard" is decided by Leanroute's **trained router**, trained on 109k real prompts whose cheap-model answers GPT-4 graded: at the default setting about half of traffic goes to the cheap model and 94% of those answers were good enough ([how we measured](https://github.com/kdandu001-arch/leanroute/blob/main/eval/results.md)). **Streaming** works with the wrapper, and streamed answers are costed correctly.

```
your code ─► leanroute ─┬─ blocked ............ $0
                        ├─ cheap model ........ $
                        └─ strong model ....... $$$
```

## Install

```bash
pip install "leanroute[local]"    # runs Laya inside your app (downloads the model once)
pip install leanroute             # lightweight: talks to a Leanroute server instead
```

## 1. Drop-in: one line in front of your existing client

```python
from openai import OpenAI
from leanroute import Leanroute

lr = Leanroute()                                        # or Leanroute(api_url="http://localhost:8000")
client = lr.wrap(OpenAI(), cheap="gpt-4o-mini", strong="gpt-4o")

r = client.chat.completions.create(
    model="auto",                                      # Leanroute picks cheap vs strong
    messages=[{"role": "user", "content": "Capital of Australia?"}],
)
print(r.choices[0].message.content, r.leanroute.route)  # -> "Canberra", "cheap"
```

Everything else about the client is unchanged. Works with `OpenAI`, `AsyncOpenAI`, any OpenAI-compatible SDK (Groq, Together, OpenRouter, Ollama's `/v1`), and `Anthropic` (`client.messages.create(model="auto", ...)`).

* `model="auto"` → guard + route.
* Any real model name → guard only; your model is used ("pinned").
* Blocked prompts raise `leanroute.Blocked` before any tokens are billed.
* Streaming works as usual; the cost is recorded when the stream ends:

```python
stream = client.chat.completions.create(model="auto", messages=messages, stream=True)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
print(stream.leanroute.route)
```

## 2. Any LLM, any framework: ask for a route

```python
d = lr.route(messages, cheap="small-model", strong="big-model")
if d.blocked:
    return "Sorry, I can't help with that."
reply = call_my_llm(model=d.model, messages=messages)   # LangChain, LiteLLM, raw HTTP, anything
```

Or just the guardrail:

```python
from leanroute import Blocked
try:
    lr.guard(user_input)
except Blocked as e:
    print("blocked:", e.decision.reason)
```

## 3. Skip the LLM entirely for simple decisions

```python
from leanroute import yes_no, choice, level

lr.check("FREE crypto signals!!!", "Is this spam?")      # -> 0.94

a = lr.decide(ticket_text, {
    "team":    choice("Which team should handle this?", ["billing", "technical", "sales", "other"]),
    "urgent":  yes_no("Is there a deadline or time pressure?"),
    "anger":   level("How frustrated is the customer?", ["calm", "annoyed", "furious"]),
})
a["team"].value, a["team"].confidence, a["urgent"].yes
```

## Savings report

```python
lr = Leanroute(prices={"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00)})  # USD per 1M tokens in/out
...
lr.stats.summary()
# {'requests': 120, 'routes': {'cheap': 81, 'strong': 37, 'blocked': 2}, 'saved_usd': 1.84, 'saved_pct': 71.3, ...}
```

Use your provider's **current** prices; savings are only as accurate as those numbers.

## Dashboard: `leanroute dashboard`

Every request is also saved to a small file on your own computer (`~/.leanroute/usage.db`): route, model, token counts and cost. **Never prompt text**, and nothing is sent anywhere. To see it:

```bash
leanroute dashboard        # opens http://127.0.0.1:8765/dashboard in your browser
leanroute stats            # the same totals as JSON, for scripts
```

It's the same dashboard as the Leanroute server's: money saved, routing split, attacks blocked, a per-day chart. Dollar amounts need `prices=`; without them you still see requests, routes and blocks. Options: `--port`, `--db`, `--no-browser`. Group apps with `Leanroute(project="shop")`; turn saving off with `Leanroute(record_usage=False)` or `LEANROUTE_RECORD=0`, or move the file with `LEANROUTE_DB`.

## Tuning

```python
from leanroute import Policy
lr = Leanroute(policy=Policy(guard_mode="precise", router="leanroute", router_threshold=0.107))
```

`router_threshold` is the savings dial: lower = more cautious. On held-out data, 0.067 sent ~30% of traffic to the cheap model (96% of those answers good enough), 0.107 ~50% (94%), 0.165 ~70% (92%). `router="laya"` uses Laya's difficulty score instead.

**Guard modes** (measured in [`eval/results.md`](https://github.com/kdandu001-arch/leanroute/blob/main/eval/results.md)):

| `guard_mode` | What blocks a prompt |
|---|---|
| `precise` (default) | ProtectAI's open-source prompt-injection detector. Fewest normal requests blocked; never blocked code or math in testing |
| `broad` | The detector, or Laya when its jailbreak and injection scores are both very high. Catches more role-play jailbreaks, blocks some coding requests |
| `laya` | Laya's jailbreak and injection scores only |
| `off` | Nothing |

In local mode the detector runs inside your app (downloaded on first use); with `api_url` it runs on the server.

**Fail-open by default:** if the decision layer errors (server down, model not loaded), requests go to your strong model so your app keeps working. Set `fail_open=False` to raise instead.

## Honest limits

Laya decides; it doesn't write. It reads short text (about a page), works best with a handful of options per question, and should be **fine-tuned and calibrated on your own data** before you trust its thresholds in production. Start conservative and check the routes in `lr.stats`.

## License

Apache 2.0. Built on Laya © Convai Innovations (Apache 2.0).
