"""Any LLM / framework: ask Leanroute which model to use, then call it yourself."""
from leanroute import Leanroute

lr = Leanroute(api_url="http://localhost:8000")     # a running Leanroute server (or Leanroute() for local)

messages = [{"role": "user", "content": "Summarize this contract's termination clause in 2 lines."}]
d = lr.route(messages, cheap="llama3.2:3b", strong="llama3.1:70b")

if d.blocked:
    print("Blocked:", d.reason)
else:
    print(f"Use {d.model} ({d.route}): {d.reason}")
    # reply = your_llm_call(model=d.model, messages=messages)   # LangChain, LiteLLM, Ollama, raw HTTP...
