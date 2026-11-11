"""Run provenance: record everything a CV_editor session did, so the run can be reconstructed later.

A RunRecorder writes into one run folder:
    events.jsonl     every graph step (node and its state update), router decision, interrupt and reply
    llm_calls.jsonl  every chat-model call: node, model, prompt messages, response, tokens, cost, time
    summary.json     totals per model and per node (tokens, cost, calls), timings, errors

Used by testing mode (evals/run_case.py) and by `python langgraph_agent.py --trace-dir DIR`.
API keys never reach these files: the recorder writes no environment, no client objects and no
serialized model settings, and every text it writes is passed through `scrub()` as a last guard.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings

# OpenAI list prices in USD per 1M tokens (input, cached input, output), read 2026-11-10 from
# https://developers.openai.com/api/docs/pricing. Cached-input prices not listed there are assumed
# at 10% of the input price. Longest matching prefix wins (dated model names match their family).
PRICES_PER_1M = {
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6.1-terra": (1.00, 0.10, 5.00),
    "gpt-6.1-sol": (2.00, 0.20, 10.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "text-embedding-3-small": (0.02, 0.02, 0.0),
    # Anthropic list prices (input, cache hit, output), read 2026-11-11 from
    # https://platform.claude.com/docs/en/about-claude/pricing. Haiku 5.5's apply to prompts up to 100k tokens.
    # Anthropic reports uncached input, cache writes and cache reads apart: evals/grader.py prices them.
    "claude-opus-5-5": (4.00, 0.20, 20.00),
    "claude-sonnet-5-5": (2.00, 0.10, 10.00),
    "claude-haiku-5-5": (0.10, 0.01, 0.50),
}
# Anthropic 5-minute cache-write prices, USD per 1M tokens (1.25x input). Batch halves every price.
CACHE_WRITE_PER_1M = {
    "claude-opus-5-5": 5.00,
    "claude-sonnet-5-5": 2.50,
    "claude-haiku-5-5": 0.125,
}
# The Batch APIs (OpenAI /v1/batches, Anthropic Message Batches) charge half of every price above.
BATCH_DISCOUNT = 0.5

logger = logging.getLogger(__name__)

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"tvly-[A-Za-z0-9_\-]{8,}"),
]
_SECRET_KEYS = re.compile(r"api[_-]?key|secret|password|authorization", re.IGNORECASE)


def scrub(value: Any) -> Any:
    """Copy of `value` with anything that looks like an API key replaced, and secret-named keys dropped."""
    if isinstance(value, str):
        for pattern in _SECRET_PATTERNS:
            value = pattern.sub("[REDACTED]", value)
        return value
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if _SECRET_KEYS.search(str(k)) else scrub(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    return value


def price_for(model: Optional[str]) -> Optional[tuple]:
    """(input, cached input, output) USD per 1M tokens for a model name, or None if unknown."""
    if not model:
        return None
    name = model.split(":", 1)[-1]
    best = max((k for k in PRICES_PER_1M if name.startswith(k)), key=len, default=None)
    return PRICES_PER_1M.get(best) if best else None


def cost_usd(model: Optional[str], input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> Optional[float]:
    """Cost of one call at list price, or None for a model without a known price."""
    prices = price_for(model)
    if prices is None:
        return None
    p_in, p_cached, p_out = prices
    cached = min(cached_tokens or 0, input_tokens or 0)
    return ((input_tokens - cached) * p_in + cached * p_cached + output_tokens * p_out) / 1e6


def _json_default(obj: Any) -> Any:
    if isinstance(obj, type):  # e.g. a pydantic schema class passed as response_format
        schema = getattr(obj, "model_json_schema", None)
        return {"schema_class": obj.__name__, "json_schema": schema()} if schema else obj.__name__
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def to_jsonable(value: Any) -> Any:
    """`value` as plain JSON data (pydantic models dumped, other objects as strings), scrubbed."""
    return scrub(json.loads(json.dumps(value, default=_json_default, ensure_ascii=False)))


def _message_dict(message: Any) -> dict:
    """A LangChain message as {"role", "content", ...extra} for the log."""
    out = {"role": getattr(message, "type", type(message).__name__), "content": getattr(message, "content", str(message))}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = tool_calls
    extra = getattr(message, "additional_kwargs", None)
    if extra:
        out["additional_kwargs"] = extra
    return out


class _LLMCallRecorder(BaseCallbackHandler):
    """Callback handler that logs every chat-model call made inside the graph."""

    raise_error = False  # a logging problem must never end the session

    def __init__(self, recorder: "RunRecorder"):
        self.recorder = recorder
        self._open: dict = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        params = kwargs.get("invocation_params") or {}
        metadata = metadata or {}
        self._open[run_id] = {
            "started_at": time.time(),
            "node": metadata.get("langgraph_node"),
            "step": metadata.get("langgraph_step"),
            "model": params.get("model") or params.get("model_name") or metadata.get("ls_model_name"),
            "provider": metadata.get("ls_provider"),
            # a call whose reply came from a Batch API (evals/batch_sp.py replays it): priced at the Batch rate
            "batch_id": metadata.get("batch_id"),
            "invocation_params": {k: v for k, v in params.items() if k not in ("tools", "response_format")},
            "response_format": params.get("response_format"),
            "tools": params.get("tools"),
            "prompt": [[_message_dict(m) for m in batch] for batch in messages],
        }

    def on_llm_end(self, response, *, run_id, **kwargs):
        start = self._open.pop(run_id, None) or {"started_at": time.time()}
        generations = []
        usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
        response_model = None
        metadata = []
        for batch in response.generations or []:
            for gen in batch:
                message = getattr(gen, "message", None)
                generations.append(_message_dict(message) if message is not None else {"text": gen.text})
                um = getattr(message, "usage_metadata", None) or {}
                usage["input_tokens"] += um.get("input_tokens", 0) or 0
                usage["output_tokens"] += um.get("output_tokens", 0) or 0
                usage["cached_tokens"] += (um.get("input_token_details") or {}).get("cache_read", 0) or 0
                usage["reasoning_tokens"] += (um.get("output_token_details") or {}).get("reasoning", 0) or 0
                meta = getattr(message, "response_metadata", None) or {}
                metadata.append(meta)
                response_model = response_model or meta.get("model_name") or meta.get("model")
        model = response_model or start.get("model")
        list_cost = cost_usd(model, usage["input_tokens"], usage["output_tokens"], usage["cached_tokens"])
        batched = bool(start.get("batch_id"))
        cost = list_cost * BATCH_DISCOUNT if list_cost is not None and batched else list_cost
        self.recorder._llm_call({
            **{k: v for k, v in start.items() if k != "started_at" and not (k == "batch_id" and v is None)},
            "model": model,
            "seconds": round(time.time() - start["started_at"], 2),
            "usage": usage,
            "cost_usd": cost,
            **({"cost_list_usd": list_cost} if batched else {}),
            # finish_reason, service_tier, system_fingerprint, response id and model as the provider returned them
            "response_metadata": metadata,
            "response": generations,
        })

    def on_llm_error(self, error, *, run_id, **kwargs):
        start = self._open.pop(run_id, None) or {"started_at": time.time()}
        self.recorder._llm_call({
            **{k: v for k, v in start.items() if k != "started_at"},
            "seconds": round(time.time() - start["started_at"], 2),
            "error": f"{type(error).__name__}: {error}",
        })


class CountingEmbeddings(Embeddings):
    """Wraps an embeddings object and records how much text it embedded (tokens estimated as chars/4)."""

    def __init__(self, inner: Any, recorder: "RunRecorder", model: str):
        self._inner = inner
        self._recorder = recorder
        self._model = model

    def _count(self, kind: str, texts: list) -> None:
        chars = sum(len(t) for t in texts)
        tokens = chars // 4
        self._recorder.event("embedding", call=kind, model=self._model, texts=len(texts), chars=chars,
                             estimated_tokens=tokens, cost_usd=cost_usd(self._model, tokens, 0))
        self._recorder._add_usage(self._model, "embeddings", tokens, 0, 0, cost_usd(self._model, tokens, 0), estimated=True)

    def embed_documents(self, texts):
        self._count("documents", list(texts))
        return self._inner.embed_documents(texts)

    def embed_query(self, text):
        self._count("query", [text])
        return self._inner.embed_query(text)

    def __call__(self, text):  # FAISS may call the embedding object directly
        return self.embed_query(text)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class RunRecorder:
    """Writes the provenance files for one run folder. Thread-safe; never raises into the session."""

    def __init__(self, run_dir: str | os.PathLike, cost_cap_usd: Optional[float] = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cost_cap_usd = cost_cap_usd
        self.started = time.time()
        self._lock = threading.Lock()
        self._seq = 0
        self.totals: dict = {"by_model": {}, "by_node": {}, "llm_calls": 0, "llm_errors": 0, "cost_usd": 0.0}
        self.errors: list = []
        self.callback = _LLMCallRecorder(self)

    # -- writing -----------------------------------------------------------------------------
    def _append(self, filename: str, record: dict) -> None:
        line = json.dumps(to_jsonable(record), ensure_ascii=False)
        with self._lock:
            with open(self.run_dir / filename, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _stamp(self) -> dict:
        with self._lock:
            self._seq += 1
            seq = self._seq
        return {"seq": seq, "t": round(time.time() - self.started, 3), "at": datetime.now(timezone.utc).isoformat(timespec="milliseconds")}

    def event(self, kind: str, **data: Any) -> None:
        """Append one event (graph step, interrupt, reply, error, note) to events.jsonl."""
        try:
            self._append("events.jsonl", {**self._stamp(), "kind": kind, **data})
        except Exception as e:  # noqa: BLE001 - provenance must not break the session
            logger.warning(f"provenance: could not record a {kind} event ({type(e).__name__}: {e})")

    def _add_usage(self, model, node, input_tokens, output_tokens, cached_tokens, cost, estimated=False):
        with self._lock:
            for bucket, key in (("by_model", model or "unknown"), ("by_node", node or "unknown")):
                t = self.totals[bucket].setdefault(key, {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                                         "cached_tokens": 0, "cost_usd": 0.0})
                t["calls"] += 1
                t["input_tokens"] += input_tokens
                t["output_tokens"] += output_tokens
                t["cached_tokens"] += cached_tokens
                t["cost_usd"] += cost or 0.0
                if estimated:
                    t["estimated"] = True
            self.totals["cost_usd"] += cost or 0.0

    def _llm_call(self, record: dict) -> None:
        try:
            stamp = self._stamp()
            self._append("llm_calls.jsonl", {**stamp, **record})
            if record.get("error"):
                with self._lock:
                    self.totals["llm_errors"] += 1
                self.errors.append({"seq": stamp["seq"], "node": record.get("node"), "error": record["error"]})
                return
            u = record.get("usage") or {}
            with self._lock:
                self.totals["llm_calls"] += 1
            self._add_usage(record.get("model"), record.get("node"), u.get("input_tokens", 0),
                            u.get("output_tokens", 0), u.get("cached_tokens", 0), record.get("cost_usd"))
        except Exception as e:  # noqa: BLE001 - provenance must not break the session
            logger.warning(f"provenance: could not record a model call ({type(e).__name__}: {e})")

    # -- helpers for the session loop ----------------------------------------------------------
    @property
    def cost(self) -> float:
        return self.totals["cost_usd"]

    def over_cap(self) -> bool:
        return self.cost_cap_usd is not None and self.cost > self.cost_cap_usd

    def record_error(self, where: str, exc: BaseException) -> None:
        err = {"where": where, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]}
        self.errors.append(err)
        self.event("error", **err)

    def write_json(self, relative_path: str, data: Any) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def write_summary(self, **extra: Any) -> Path:
        return self.write_json("summary.json", {
            "seconds": round(time.time() - self.started, 1),
            **self.totals,
            "cost_usd": round(self.totals["cost_usd"], 6),
            "errors": self.errors,
            **extra,
        })

    def graph_config(self, thread_id: str) -> dict:
        """LangGraph config that routes every model call through this recorder."""
        return {"configurable": {"thread_id": thread_id}, "callbacks": [self.callback]}


def summarise_update(update: Any) -> Any:
    """A node's state update for events.jsonl: full values, but only the new chat messages."""
    if not isinstance(update, dict):
        return update
    out = dict(update)
    msgs = out.get("messages")
    if isinstance(msgs, list) and msgs:
        out["messages"] = msgs[-1:]
        out["messages_total"] = len(msgs)
    return out


def stream_turn(graph, graph_input, config, recorder: Optional[RunRecorder] = None):
    """
    Run the graph until it finishes or pauses for user input, logging every node update.

    Returns:
        The interrupt payload (a dict with "message") if the graph paused, else None.
    """
    interrupt_value = None
    for chunk in graph.stream(graph_input, config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                interrupt_value = update[0].value if update else {}
                if recorder:
                    recorder.event("interrupt", message=(interrupt_value or {}).get("message"))
                continue
            if recorder:
                data = {"node": node, "update": summarise_update(update)}
                if isinstance(update, dict) and "next" in update and node == "router":
                    data["router_decision"] = update.get("next")
                recorder.event("node", **data)
        if recorder and recorder.over_cap():
            raise RuntimeError(f"cost cap reached: ${recorder.cost:.4f} > ${recorder.cost_cap_usd}")
    return interrupt_value
