"""Drop-in wrappers so existing LLM code gets Leanroute with one line.

OpenAI-style clients (OpenAI, AsyncOpenAI, and anything that copies the SDK
shape: Groq, Together, OpenRouter via base_url, Ollama's OpenAI endpoint):
    client = lr.wrap(OpenAI(), cheap="...", strong="...")
    client.chat.completions.create(model="auto", messages=[...])

Anthropic:
    client = lr.wrap(Anthropic(), cheap="...", strong="...")
    client.messages.create(model="auto", max_tokens=500, messages=[...])

model="auto"  -> guard + route to cheap/strong
any other     -> guard only, your model is used as-is ("pinned")
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any

from .core import Blocked, Decision


def _usage(resp) -> tuple[int, int]:
    u = getattr(resp, "usage", None)
    if u is None and isinstance(resp, dict):
        u = resp.get("usage")
    if u is None:
        return 0, 0
    g = (lambda k: u.get(k, 0)) if isinstance(u, dict) else (lambda k: getattr(u, k, 0) or 0)
    return int(g("prompt_tokens") or g("input_tokens") or 0), int(g("completion_tokens") or g("output_tokens") or 0)


def _chunk_info(chunk) -> tuple[str, int, int]:
    """(text, prompt tokens, completion tokens) carried by one streamed chunk or event, OpenAI or Anthropic."""
    get = (lambda o, k: o.get(k) if isinstance(o, dict) else getattr(o, k, None))
    text = ""
    for choice in get(chunk, "choices") or []:                       # OpenAI-style chunk
        delta = get(choice, "delta")
        text += (get(delta, "content") if delta is not None else None) or ""
    delta = get(chunk, "delta")                                      # Anthropic content_block_delta
    if delta is not None and get(chunk, "type") == "content_block_delta":
        text += get(delta, "text") or ""
    usage = get(chunk, "usage")
    message = get(chunk, "message")                                  # Anthropic message_start
    if usage is None and message is not None:
        usage = get(message, "usage")
    pin, pout = _usage({"usage": usage}) if usage is not None else (0, 0)
    return text, pin, pout


class _Recorder:
    """Passes a stream through unchanged and records its real cost when it ends."""

    def __init__(self, stream, create, d, messages):
        self._s, self._create, self._d, self._messages = stream, create, d, messages
        self._text, self._pin, self._pout, self._done = [], 0, 0, False

    def _see(self, chunk):
        text, pin, pout = _chunk_info(chunk)
        self._text.append(text)
        self._pin, self._pout = max(self._pin, pin), max(self._pout, pout)
        return chunk

    def _finish(self):
        if self._done:
            return
        self._done = True
        pin = self._pin or max(1, sum(len(str(m.get("content", ""))) for m in self._messages or []) // 4)
        pout = self._pout or max(1, len("".join(self._text)) // 4)   # no usage in the stream: estimate it
        self._create.lr.stats.record(self._d, self._create.strong, pin, pout)

    def __iter__(self):
        try:
            for chunk in self._s:
                yield self._see(chunk)
        finally:
            self._finish()

    async def __aiter__(self):
        try:
            async for chunk in self._s:
                yield self._see(chunk)
        finally:
            self._finish()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._finish()
        if hasattr(self._s, "close"):
            self._s.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self._finish()
        if hasattr(self._s, "close"):
            r = self._s.close()
            if inspect.isawaitable(r):
                await r

    def __getattr__(self, name):
        return getattr(self._s, name)


def _attach(resp, d: Decision):
    try:
        setattr(resp, "leanroute", d)
    except Exception:
        pass  # some response objects are frozen; the decision is still on lr.last
    return resp


class _Create:
    def __init__(self, lr, original, cheap, strong, on_block):
        self.lr, self.original, self.cheap, self.strong, self.on_block = lr, original, cheap, strong, on_block
        self.is_async = inspect.iscoroutinefunction(original)

    def _plan(self, kwargs):
        requested = kwargs.get("model", "auto")
        d = self.lr.route(kwargs.get("messages") or [], self.cheap, self.strong)
        if d.blocked:
            self.lr.stats.record(d, self.strong)
            if self.on_block == "raise":
                raise Blocked(d)
            return d, None
        if requested not in (None, "", "auto"):
            d = Decision("pinned", requested, d.reason, d.scores, d.decision_ms)
            self.lr.last = d
        kwargs = dict(kwargs, model=d.model or self.strong)
        return d, kwargs

    def _done(self, d, resp, kwargs=None):
        if kwargs and kwargs.get("stream"):
            return _attach(_Recorder(resp, self, d, kwargs.get("messages")), d)
        pin, pout = _usage(resp)
        self.lr.stats.record(d, self.strong, pin, pout)
        return _attach(resp, d)

    def __call__(self, *args, **kwargs):
        if self.is_async:
            return self._acall(*args, **kwargs)
        d, kw = self._plan(kwargs)
        if kw is None:
            return None
        return self._done(d, self.original(*args, **kw), kw)

    async def _acall(self, *args, **kwargs):
        d, kw = await asyncio.to_thread(self._plan, kwargs)
        if kw is None:
            return None
        return self._done(d, await self.original(*args, **kw), kw)


class _Proxy:
    """Forwards everything to the wrapped object, except the attributes we override."""

    def __init__(self, target, overrides):
        object.__setattr__(self, "_t", target)
        object.__setattr__(self, "_o", overrides)

    def __getattr__(self, name):
        o = object.__getattribute__(self, "_o")
        if name in o:
            return o[name]
        return getattr(object.__getattribute__(self, "_t"), name)


def wrap_client(lr, client: Any, cheap: str, strong: str, on_block: str = "raise"):
    if on_block not in ("raise", "none"):
        raise ValueError("on_block must be 'raise' or 'none'")
    chat = getattr(client, "chat", None)
    if chat is not None and hasattr(getattr(chat, "completions", None), "create"):
        comp = chat.completions
        new_comp = _Proxy(comp, {"create": _Create(lr, comp.create, cheap, strong, on_block)})
        return _Proxy(client, {"chat": _Proxy(chat, {"completions": new_comp})})
    msgs = getattr(client, "messages", None)
    if msgs is not None and hasattr(msgs, "create"):
        return _Proxy(client, {"messages": _Proxy(msgs, {"create": _Create(lr, msgs.create, cheap, strong, on_block)})})
    raise TypeError("Unsupported client: expected client.chat.completions.create (OpenAI-style) or client.messages.create (Anthropic)")
