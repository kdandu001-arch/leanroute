# leanroute

**Put a fast, calibrated decision layer in front of any LLM.** Powered by [Laya](https://huggingface.co/convaiinnovations/laya) (Apache 2.0).

Before your app pays an LLM, Leanroute asks Laya (~35 ms on GPU, $0 per call) three things:

1. **Is this an attack?** Jailbreak / prompt injection → blocked, no LLM call.
2. **How hard is it?** Easy → your cheap model. Hard, sensitive or unsure → your strong model.
3. **Anything else you ask**, e.g. "is this spam?", "which team?", "hot lead?" → answered by Laya with a probability, no LLM at all.

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

## Tuning

```python
from leanroute import Policy
lr = Leanroute(policy=Policy(block_threshold=0.9, easy_max=1.0, min_confidence=0.0))
```

**Fail-open by default:** if the decision layer errors (server down, model not loaded), requests go to your strong model so your app keeps working. Set `fail_open=False` to raise instead.

## Honest limits

Laya decides; it doesn't write. It reads short text (about a page), works best with a handful of options per question, and should be **fine-tuned and calibrated on your own data** before you trust its thresholds in production. Start conservative and check the routes in `lr.stats`.

## License

Apache 2.0. Built on Laya © Convai Innovations (Apache 2.0).
