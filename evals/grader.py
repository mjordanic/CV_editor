"""Provider layer for the model graders (evaluate.py, score_truth.py) and for Batch grading.

One grading is one model call that answers a pydantic schema. This module keeps the three steps apart so
that a batch submitter can reuse the first and the last without the second:

    request = build_request(grader, system, shared_text, document_text, images, Schema)   # no network
    result = send(request, recorder)                                                     # one paid call
    result = result_from_message(request, message, seconds, batch=True)                 # parse a reply

`GradingRequest.anthropic_params()` is exactly the `params` of one Message Batches request (the
synchronous path streams the same dict); `custom_id` and submission belong to the batch code.

Graders are named "provider:model":

    anthropic:claude-sonnet-5-5   the primary grader      anthropic:claude-opus-5-5   the second grader
    anthropic:claude-haiku-5-5    cheap re-grader / trap check
    openai:<model>                the LangChain path used before (e.g. openai:gpt-6.1-sol, openai:gpt-6-luna)

Anthropic calls go through the official `anthropic` SDK:
- the model id is pinned (only the ids in ANTHROPIC_MODELS are accepted) and the id the API returns is
  recorded with every call;
- structured output is native (`output_config.format` with a JSON schema made from the pydantic model;
  forced tool use returns a 400 on Sonnet 5.5 and Opus 5.5). Ranges the API's schema subset cannot
  express (0-10 scores) are checked client-side by validating the answer against the pydantic model;
- thinking is adaptive (it cannot be turned off on these models) and the effort is set explicitly
  (default "medium"; Opus 5.5 and Haiku 5.5 default to medium, Sonnet 5.5 to high);
- temperature cannot be set: Opus 5.5 rejects any `temperature`, Sonnet 5.5 and Haiku 5.5 reject
  non-default values. Nothing is sent, and every record says "temperature: not settable";
- page images are Anthropic `image` blocks (base64 PNG);
- a cache breakpoint follows the system prompt and another the shared block (job ad, company text,
  candidate material, the user's instructions), which is identical for every arm of a case;
- no server-side fallback model: a refusal is recorded (stop_reason, stop_details) and the grading is
  flagged, never silently answered by another model;
- a reply that does not parse (invalid JSON, out-of-range score, max_tokens cut-off) is re-asked at most
  MAX_ATTEMPTS times in all; every attempt is logged with its tokens and cost.

OpenAI graders keep the LangChain `with_structured_output` path (temperature: provider default, not set).

Costs use provenance.PRICES_PER_1M plus the cache-write prices in provenance.CACHE_WRITE_PER_1M;
Batch results cost half (BATCH_DISCOUNT).
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

ANTHROPIC_MODELS = ("claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5")
PROVIDERS = ("anthropic", "openai")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORT = "medium"
TEMPERATURE_NOT_SETTABLE = "not settable"
TEMPERATURE_PROVIDER_DEFAULT = "provider default (not set)"
MAX_TOKENS = 64000  # the evaluation answer alone is ~10k tokens; thinking shares this budget
MAX_ATTEMPTS = 2  # one re-ask after an unparseable answer; never after a refusal or an API error
BATCH_DISCOUNT = 0.5
HAIKU_LONG_PROMPT_TOKENS = 100_000  # Haiku 5.5 list prices apply up to here; longer prompts cost more

OK, PARSE_ERROR, REFUSED = "ok", "parse_error", "refused"


# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GraderSpec:
    provider: str
    model: str
    effort: str | None = None

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def temperature(self) -> str:
        return TEMPERATURE_NOT_SETTABLE if self.provider == "anthropic" else TEMPERATURE_PROVIDER_DEFAULT

    def describe(self) -> dict:
        return {"provider": self.provider, "model_requested": self.model, "effort": self.effort,
                "temperature": self.temperature}


def parse_grader(spec: str, effort: str | None = None) -> GraderSpec:
    """'anthropic:claude-sonnet-5-5' / 'openai:gpt-6.1-sol' -> GraderSpec. A bare 'claude-...' id is
    Anthropic, any other bare id OpenAI. Anthropic ids must be one of ANTHROPIC_MODELS (pinned, no aliases
    or date suffixes); `effort` defaults to DEFAULT_EFFORT there and is not used for OpenAI."""
    provider, sep, model = spec.partition(":")
    if not sep:
        provider, model = ("anthropic" if spec.startswith("claude-") else "openai"), spec
    if provider not in PROVIDERS or not model:
        raise ValueError(f"grader '{spec}': use provider:model with provider one of {', '.join(PROVIDERS)}")
    if provider == "anthropic":
        if model not in ANTHROPIC_MODELS:
            raise ValueError(f"grader '{spec}': Anthropic grader must be one of {', '.join(ANTHROPIC_MODELS)}")
        effort = effort or DEFAULT_EFFORT
        if effort not in EFFORTS:
            raise ValueError(f"effort '{effort}': one of {', '.join(EFFORTS)}")
        return GraderSpec(provider, model, effort)
    if effort:
        raise ValueError("--effort applies to Anthropic graders only")
    return GraderSpec(provider, model, None)


# ---------------------------------------------------------------------------------------------
_UNSUPPORTED_KEYS = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
                     "minLength", "maxLength", "minItems", "maxItems", "uniqueItems", "pattern", "default", "title")


def json_schema_for(schema: type[BaseModel]) -> dict:
    """The pydantic model's JSON schema in the subset structured outputs accept: numeric, string and array
    constraints dropped (validated client-side), every object closed (additionalProperties false) and
    every property required."""
    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        out = {}
        for k, v in node.items():
            if k in ("properties", "$defs"):
                out[k] = {name: walk(sub) for name, sub in v.items()}
            elif k in _UNSUPPORTED_KEYS:
                continue
            else:
                out[k] = walk(v)
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out

    # property names may themselves be "title" etc.: property maps are walked by value, never filtered
    return walk(schema.model_json_schema())


def sha256_text(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def rubric_hash(system: str, templates: list[str], schema: type[BaseModel]) -> str:
    """Hash of everything that defines the rubric: system prompt, prompt template(s) and answer schema."""
    return sha256_text(system, *templates, json.dumps(json_schema_for(schema), sort_keys=True))


# ---------------------------------------------------------------------------------------------
@dataclass
class GradingRequest:
    """Everything one grading sends, before anything is sent."""
    grader: GraderSpec
    system: str
    shared_text: str  # identical across the arms of a case: cached
    document_text: str  # this session's documents
    schema: type[BaseModel]
    images: list[Path] = field(default_factory=list)
    max_tokens: int = MAX_TOKENS
    rubric_sha256: str | None = None

    @property
    def user_text(self) -> str:
        return self.shared_text + self.document_text

    def anthropic_params(self) -> dict:
        """Messages API params (no `stream`): the synchronous path streams them, Batch submits them as one
        request's `params`."""
        if self.grader.provider != "anthropic":
            raise ValueError("anthropic_params() is for Anthropic graders")
        content: list[dict] = [{"type": "text", "text": self.shared_text, "cache_control": {"type": "ephemeral"}},
                               {"type": "text", "text": self.document_text}]
        for path in self.images:
            data = base64.standard_b64encode(Path(path).read_bytes()).decode("ascii")
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
        return {
            "model": self.grader.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": content}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.grader.effort,
                              "format": {"type": "json_schema", "schema": json_schema_for(self.schema)}},
        }

    def langchain_messages(self) -> list:
        """System and human messages for a LangChain chat model; images as base64 image_url parts."""
        from langchain_core.messages import HumanMessage, SystemMessage
        if not self.images:
            return [SystemMessage(content=self.system), HumanMessage(content=self.user_text)]
        content: list[dict] = [{"type": "text", "text": self.user_text}]
        for path in self.images:
            data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}", "detail": "high"}})
        return [SystemMessage(content=self.system), HumanMessage(content=content)]

    def loggable(self) -> dict:
        """The request for prompt.json: image data replaced by the page file names."""
        names = [Path(p).name for p in self.images]
        out = {"grader": self.grader.describe(), "rubric_sha256": self.rubric_sha256,
               "images": names, "prompt_sha256": self.prompt_hash()}
        if self.grader.provider == "anthropic":
            params = copy.deepcopy(self.anthropic_params())
            blocks = params["messages"][0]["content"]
            for block, name in zip([b for b in blocks if b["type"] == "image"], names):
                block["source"] = {"omitted": f"png image, see pages/{name}"}
            out["params"] = params
        else:
            msgs = self.langchain_messages()
            out["messages"] = [{"role": m.type, "content": _loggable_parts(m.content, names)} for m in msgs]
        return out

    def prompt_hash(self) -> str:
        """Hash of the exact prompt: texts and image bytes."""
        images = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in self.images]
        return sha256_text(self.grader.name, self.system, self.shared_text, self.document_text, *images)


def _loggable_parts(content, names: list[str]):
    if not isinstance(content, list):
        return content
    it = iter(names)
    return [p if p.get("type") != "image_url" else {"type": "image_url", "image_url": f"<png, see pages/{next(it, '?')}>"}
            for p in content]


def build_request(grader: GraderSpec, system: str, shared_text: str, document_text: str,
                  schema: type[BaseModel], images: list[Path] | None = None, max_tokens: int = MAX_TOKENS,
                  rubric_sha256: str | None = None) -> GradingRequest:
    return GradingRequest(grader, system, shared_text, document_text, schema, list(images or []), max_tokens,
                          rubric_sha256)


# ---------------------------------------------------------------------------------------------
def anthropic_cost(model: str, usage: dict, batch: bool = False) -> float | None:
    """USD for one Anthropic call: uncached input, 5-minute cache writes, cache reads and output (thinking
    included) at list price; half for Batch. None for an unknown model."""
    from provenance import CACHE_WRITE_PER_1M, price_for
    prices = price_for(model)
    if prices is None:
        return None
    p_in, p_hit, p_out = prices
    p_write = CACHE_WRITE_PER_1M.get(model.split(":", 1)[-1], p_in * 1.25)
    usd = (usage.get("input_tokens", 0) * p_in + usage.get("cache_creation_input_tokens", 0) * p_write
           + usage.get("cache_read_input_tokens", 0) * p_hit + usage.get("output_tokens", 0) * p_out) / 1e6
    return usd * (BATCH_DISCOUNT if batch else 1.0)


def _usage_dict(usage: Any) -> dict:
    if usage is None:
        return {}
    data = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
    return {k: (data.get(k) or 0) for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                                             "cache_read_input_tokens")}


def parse_answer(text: str | None, schema: type[BaseModel]) -> tuple[dict | None, str | None]:
    """(answer, None) when `text` is JSON that validates against the schema (ranges included), else
    (None, why)."""
    if not text:
        return None, "empty answer"
    try:
        return schema.model_validate(json.loads(text)).model_dump(mode="json"), None
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    except ValidationError as e:
        return None, f"schema validation failed: {str(e)[:2000]}"


@dataclass
class GradingResult:
    status: str  # ok | parse_error | refused
    answer: dict | None
    attempts: list[dict]
    grader: dict

    @property
    def cost_usd(self) -> float:
        return round(sum(a.get("cost_usd") or 0.0 for a in self.attempts), 6)

    @property
    def model_returned(self) -> str | None:
        return next((a.get("model_returned") for a in reversed(self.attempts) if a.get("model_returned")), None)

    def record(self) -> dict:
        """The grading's identity and totals, for evaluation.json / truth_check.json."""
        totals: dict = {}
        for a in self.attempts:
            for k, v in (a.get("usage") or {}).items():
                totals[k] = totals.get(k, 0) + (v or 0)
        last = self.attempts[-1] if self.attempts else {}
        return {**self.grader, "model_returned": self.model_returned, "status": self.status,
                "attempts": len(self.attempts), "usage": totals, "cost_usd": self.cost_usd,
                "stop_reason": last.get("stop_reason"), "stop_details": last.get("stop_details"),
                "request_ids": [a.get("request_id") for a in self.attempts if a.get("request_id")],
                "batch_id": last.get("batch_id")}


def attempt_from_message(request: GradingRequest, message: Any, seconds: float, batch_id: str | None = None,
                         request_id: str | None = None) -> tuple[dict, dict | None, str | None]:
    """One Anthropic reply (streamed or from a batch) -> (attempt record, answer or None, error or None)."""
    usage = _usage_dict(getattr(message, "usage", None))
    model = getattr(message, "model", None) or request.grader.model
    stop_reason = getattr(message, "stop_reason", None)
    stop_details = getattr(message, "stop_details", None)
    if hasattr(stop_details, "model_dump"):
        stop_details = stop_details.model_dump(mode="json")
    text = "".join(getattr(b, "text", "") for b in getattr(message, "content", None) or [] if getattr(b, "type", None) == "text")
    if stop_reason == "refusal":
        answer, error = None, "refused"
    elif stop_reason == "max_tokens":
        answer, error = None, f"answer cut off at max_tokens={request.max_tokens}"
    else:
        answer, error = parse_answer(text, request.schema)
    attempt = {
        "provider": "anthropic", "model_requested": request.grader.model, "model_returned": model,
        "message_id": getattr(message, "id", None), "request_id": request_id or getattr(message, "_request_id", None),
        "batch_id": batch_id, "effort": request.grader.effort, "temperature": request.grader.temperature,
        "seconds": round(seconds, 2), "stop_reason": stop_reason, "stop_details": stop_details, "usage": usage,
        "cost_usd": anthropic_cost(model, usage, batch=batch_id is not None),
        "price_note": ("prompt over 100k tokens: Haiku 5.5 long-prompt prices apply, cost understated"
                       if model.startswith("claude-haiku") and sum(usage.get(k, 0) for k in (
                           "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
                       > HAIKU_LONG_PROMPT_TOKENS else None),
        "response_text": text, "parse_error": error,
    }
    return attempt, answer, error


def result_from_message(request: GradingRequest, message: Any, seconds: float = 0.0,
                        batch_id: str | None = None) -> GradingResult:
    """A GradingResult from one already-received Anthropic message (Batch results use this)."""
    attempt, answer, error = attempt_from_message(request, message, seconds, batch_id=batch_id)
    status = OK if answer is not None else REFUSED if error == "refused" else PARSE_ERROR
    return GradingResult(status, answer, [attempt], request.grader.describe())


# ---------------------------------------------------------------------------------------------
def _log_attempt(recorder, node: str, attempt: dict, request: GradingRequest) -> None:
    if recorder is None:
        return
    u = attempt.get("usage") or {}
    recorder._llm_call({
        "node": node, "model": attempt.get("model_returned") or request.grader.model, "provider": "anthropic",
        "batch_id": attempt.get("batch_id"),
        "invocation_params": {"model": request.grader.model, "max_tokens": request.max_tokens,
                              "effort": request.grader.effort, "thinking": "adaptive",
                              "temperature": request.grader.temperature},
        "seconds": attempt.get("seconds"),
        # input_tokens here is the whole prompt (as for OpenAI); cached_tokens the part read from cache
        "usage": {"input_tokens": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                  + u.get("cache_read_input_tokens", 0),
                  "output_tokens": u.get("output_tokens", 0), "cached_tokens": u.get("cache_read_input_tokens", 0),
                  "cache_write_tokens": u.get("cache_creation_input_tokens", 0), "reasoning_tokens": None},
        "cost_usd": attempt.get("cost_usd"),
        **{k: attempt.get(k) for k in ("model_requested", "model_returned", "message_id", "request_id",
                                       "stop_reason", "stop_details", "parse_error")},
        "response": [{"type": "text", "text": attempt.get("response_text")}],
    })


def log_attempts(recorder, result: GradingResult, request: GradingRequest, node: str = "grader") -> None:
    """Log a result's Anthropic attempts to the recorder (llm_calls.jsonl and the summary totals). send()
    does this as it goes; Batch grading calls it for a reply read with result_from_message()."""
    for attempt in result.attempts:
        _log_attempt(recorder, node, attempt, request)


def anthropic_client():
    """The Anthropic client (credentials from ANTHROPIC_API_KEY, loaded from .env by the caller). An
    out-of-credit or spend-limit error is not retried by evals.credit's policy (see send)."""
    import anthropic
    return anthropic.Anthropic()


def _send_anthropic(request: GradingRequest, recorder, node: str, client=None) -> GradingResult:
    from evals.credit import OutOfCredit, is_out_of_credit
    client = client or anthropic_client()
    params = request.anthropic_params()
    attempts: list[dict] = []
    answer = None
    status = PARSE_ERROR
    for _ in range(MAX_ATTEMPTS):
        started = time.time()
        try:
            with client.messages.stream(**params) as stream:
                message = stream.get_final_message()
                response = getattr(stream, "response", None)
                request_id = response.headers.get("request-id") if response is not None else None
        except BaseException as e:
            if recorder is not None:
                recorder._llm_call({"node": node, "model": request.grader.model, "provider": "anthropic",
                                    "seconds": round(time.time() - started, 2), "error": f"{type(e).__name__}: {e}"})
            if is_out_of_credit(e):
                raise OutOfCredit(f"{type(e).__name__}: {e}"[:500]) from e
            raise
        attempt, answer, error = attempt_from_message(request, message, time.time() - started, request_id=request_id)
        attempts.append(attempt)
        _log_attempt(recorder, node, attempt, request)
        if answer is not None:
            status = OK
            break
        if error == "refused":
            status = REFUSED
            break
    return GradingResult(status, answer, attempts, request.grader.describe())


def _send_langchain(request: GradingRequest, recorder, llm=None) -> GradingResult:
    from evals.credit import CreditGuard, OutOfCredit, install_no_retry_patch, is_out_of_credit
    install_no_retry_patch()
    if llm is None:
        from langchain.chat_models import init_chat_model
        llm = init_chat_model(request.grader.name)
    chain = llm.with_structured_output(request.schema, include_raw=True)
    callbacks = ([recorder.callback] if recorder is not None else []) + [CreditGuard()]
    attempts: list[dict] = []
    answer = None
    for _ in range(MAX_ATTEMPTS):
        started = time.time()
        cost_before = recorder.cost if recorder is not None else 0.0
        try:
            result = chain.invoke(request.langchain_messages(), config={"callbacks": callbacks})
        except BaseException as e:
            if isinstance(e, OutOfCredit) or is_out_of_credit(e):
                raise OutOfCredit(str(e)) from e
            raise
        raw = result.get("raw")
        meta = getattr(raw, "response_metadata", None) or {}
        parsed = result.get("parsed")
        error = None
        if parsed is not None:
            try:
                answer = request.schema.model_validate(parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
                                                       ).model_dump(mode="json")
            except ValidationError as e:
                error = f"schema validation failed: {str(e)[:2000]}"
        else:
            error = str(result.get("parsing_error") or "no parsed answer")
        attempts.append({
            "provider": request.grader.provider, "model_requested": request.grader.model,
            "model_returned": meta.get("model_name") or meta.get("model"),
            "request_id": meta.get("id"), "temperature": request.grader.temperature, "effort": None,
            "seconds": round(time.time() - started, 2), "stop_reason": meta.get("finish_reason"),
            "usage": dict(getattr(raw, "usage_metadata", None) or {}),
            "raw": {"content": getattr(raw, "content", None), "additional_kwargs": getattr(raw, "additional_kwargs", None),
                    "response_metadata": meta, "usage_metadata": getattr(raw, "usage_metadata", None)},
            "parse_error": error,
            # priced by the recorder's LangChain callback (provenance.cost_usd)
            "cost_usd": (recorder.cost - cost_before) if recorder is not None else None,
        })
        if answer is not None:
            break
    return GradingResult(OK if answer is not None else PARSE_ERROR, answer, attempts, request.grader.describe())


def send(request: GradingRequest, recorder=None, *, node: str = "grader", client=None, llm=None) -> GradingResult:
    """Make the grading call(s). `recorder` (provenance.RunRecorder) logs every attempt to llm_calls.jsonl
    and adds its cost. `client` replaces the Anthropic client and `llm` the LangChain model (tests).
    Raises evals.credit.OutOfCredit when the provider reports no credit or a reached spend limit."""
    if request.grader.provider == "anthropic" and llm is None:
        return _send_anthropic(request, recorder, node, client)
    return _send_langchain(request, recorder, llm)
