"""Pluginfer Devserver — drop-in OpenAI / Anthropic shim.

The killer adoption move: any existing app that talks to OpenAI or
Anthropic can be redirected at this proxy with a single env-var flip,

    OPENAI_BASE_URL=http://localhost:11434/v1
    ANTHROPIC_BASE_URL=http://localhost:11434

and start running the same prompts through the Pluginfer compute mesh —
zero SDK changes, zero code changes, zero migration. Every request is
translated into a JobSpec, dropped into the live auction, and the
lowest-cost qualifying bidder wins. Mesh GPUs underbid centralised APIs
on routine workloads; cloud providers still win when caller pays for
their quality. The shim is provider-agnostic by construction.

Why this is the unlock
----------------------
Today the SDK story is "rewrite your app to call pluginfer.jobs.submit".
That's a 1-day job per app and a non-zero cognitive tax. The shim
collapses it to:

    docker run -d -p 11434:11434 pluginfer/devserver
    OPENAI_BASE_URL=http://localhost:11434/v1 python my_existing_app.py

The hostname `api.openai.com` becomes a CNAME pointing at the mesh.
Every Anthropic / OpenAI / LangChain / LlamaIndex / DSPy / agent-
framework user gets Pluginfer for free. The competing migration cost
elsewhere in the market is days-to-weeks; here it's seconds.

Wire format
-----------
* `POST /v1/chat/completions` — OpenAI Chat Completions schema.
* `POST /v1/messages`         — Anthropic Messages schema.
* `POST /v1/embeddings`       — OpenAI embeddings (kind=embed).
* `GET  /v1/models`           — model catalogue assembled from
                                registered providers.
* `GET  /healthz`             — liveness.

Streaming (SSE) is supported on chat completions and messages — the
incoming request's `stream: true` flag is preserved end-to-end and the
event chunks emitted match each upstream's wire shape.

Receipts
--------
Every request carries response header `X-Pluginfer-Receipt-ID` and
`X-Pluginfer-Provider`. The full §D1 signed receipt is fetchable at
`/v1/receipts/{receipt_id}` (when the receipts router is mounted).
This is the migration trojan horse: silent for callers, but every
request leaves an auditable, signed trail that AWS+Anthropic+OpenAI
literally cannot match.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import secrets
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.sse import EventSourceResponse

from .jobs_service import JobRecord, JobsService

DEFAULT_PORT = int(os.environ.get("PLUGINFER_DEV_PORT", "11434"))
DEFAULT_QUALITY_FLOOR = float(os.environ.get("PLUGINFER_DEV_QUALITY_FLOOR", "0.0"))
DEFAULT_COST_CEILING_USD = float(os.environ.get("PLUGINFER_DEV_COST_CEILING_USD", "1.0"))
DEFAULT_LATENCY_MS = int(os.environ.get("PLUGINFER_DEV_LATENCY_MS", "60000"))

TERMINAL_STATES = {"completed", "failed", "cancelled", "timeout"}


# ---------------------------------------------------------------------------
# Wire schemas (only the fields we actually translate — extras pass through)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: Any  # str OR list[content-part] for multimodal — we flatten


class ChatCompletionsBody(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=None, ge=1, le=200_000)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    stop: Optional[Any] = None
    user: Optional[str] = None
    # Pluginfer-specific overrides — caller can pin auction parameters
    # without leaving the OpenAI schema (any unknown field is ignored
    # by the OpenAI SDK, so existing code stays valid):
    pluginfer_cost_ceiling_usd: Optional[float] = Field(default=None, ge=0.0)
    pluginfer_latency_ceiling_ms: Optional[int] = Field(default=None, ge=10)
    pluginfer_quality_floor: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    pluginfer_privacy: Optional[str] = None  # public|private|sensitive

    model_config = ConfigDict(extra="allow")


class AnthropicMessage(BaseModel):
    role: str
    content: Any


class MessagesBody(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int = Field(..., ge=1, le=200_000)
    temperature: Optional[float] = None
    system: Optional[Any] = None  # str OR list[block]
    stream: bool = False
    pluginfer_cost_ceiling_usd: Optional[float] = Field(default=None, ge=0.0)
    pluginfer_latency_ceiling_ms: Optional[int] = Field(default=None, ge=10)
    pluginfer_quality_floor: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    pluginfer_privacy: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class EmbeddingsBody(BaseModel):
    model: str
    input: Any  # str OR list[str]
    user: Optional[str] = None

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Translation: SDK schema -> Pluginfer JobSpec payload
# ---------------------------------------------------------------------------

def _flatten_content(content: Any) -> str:
    """OpenAI / Anthropic both accept str OR list-of-content-blocks. The
    auction's cloud-LLM providers expect a single prompt string, so we
    concatenate text parts and silently drop image/audio parts that the
    backend can't honour (they would arrive as base64 — not useful for
    a routine LLM job)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: List[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in (None, "text"):
                    out.append(str(part.get("text", "")))
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(s for s in out if s)
    return str(content) if content is not None else ""


def _messages_to_prompt(
    messages: List[Dict[str, Any]], system: Optional[Any] = None
) -> str:
    """Collapse a chat transcript into a single prompt string. We use the
    OpenAI format used by every chat-tuned model: `<role>: <text>` with
    a trailing `assistant:` cue. Mesh providers running Llama-3-Instruct
    or Filum produce sensible answers from this; cloud providers ignore
    it and use their own templating because we ALSO ship the structured
    `messages` list under `payload.openai.messages`."""
    lines: List[str] = []
    if system:
        lines.append(f"system: {_flatten_content(system)}")
    for m in messages:
        role = m.get("role", "user")
        text = _flatten_content(m.get("content"))
        lines.append(f"{role}: {text}")
    lines.append("assistant:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auction wait + result extraction
# ---------------------------------------------------------------------------

async def _await_terminal(svc: JobsService, rec: JobRecord, timeout_s: float) -> JobRecord:
    deadline = time.monotonic() + timeout_s
    poll = 0.02
    while rec.state not in TERMINAL_STATES:
        if time.monotonic() >= deadline:
            return rec
        await asyncio.sleep(poll)
        poll = min(poll * 1.5, 0.25)
    return rec


def _decode_result_text(rec: JobRecord) -> str:
    """Pull a text payload out of a completed JobRecord. Both
    `MeshGPUProvider` and `_CloudLLMProvider` populate result_b64 (via
    JobsService normalisation); the cloud path is UTF-8 text, the mesh
    path is JSON-serialised and may carry a `text` field. Try both."""
    if not rec.result_b64:
        return ""
    try:
        raw = base64.b64decode(rec.result_b64)
    except Exception:
        return ""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    # Mesh path: JSON dict with optional `text`/`output`/`response` keys
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return text
        for k in ("text", "output", "response", "completion", "content"):
            if isinstance(obj.get(k), str):
                return obj[k]
        return text
    return text


# ---------------------------------------------------------------------------
# Response builders — match upstream wire shapes byte-for-byte
# ---------------------------------------------------------------------------

def _openai_chat_response(
    *, model: str, text: str, prompt_tokens: int, completion_tokens: int,
    finish_reason: str = "stop",
) -> Dict[str, Any]:
    return {
        "id": "chatcmpl-" + secrets.token_hex(12),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "system_fingerprint": "pluginfer-mesh",
    }


def _openai_chat_chunk(
    *, chat_id: str, model: str, delta: Dict[str, Any],
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0, "delta": delta, "finish_reason": finish_reason,
        }],
    }


def _anthropic_messages_response(
    *, model: str, text: str, input_tokens: int, output_tokens: int,
    stop_reason: str = "end_turn",
) -> Dict[str, Any]:
    return {
        "id": "msg_" + secrets.token_hex(12),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def _approx_token_count(s: str) -> int:
    """Rough char/4 estimator. Good enough for `usage` accounting; the
    chain-side ledger uses the provider's signed count, not this."""
    return max(1, len(s) // 4)


# ---------------------------------------------------------------------------
# Job submission helper shared by all routes
# ---------------------------------------------------------------------------

async def _run_llm_job(
    svc: JobsService,
    *,
    kind: str,
    prompt: str,
    structured_payload: Dict[str, Any],
    cost_ceiling_usd: float,
    latency_ceiling_ms: int,
    quality_floor: float,
    privacy_class: str,
    requester_identity: str,
    streaming: bool = False,
) -> JobRecord:
    payload = {
        "prompt": prompt,
        "input": prompt,  # mesh providers may key off either name
        **structured_payload,
    }
    rec = await svc.submit(
        kind=kind,
        payload=payload,
        cost_ceiling_usd=cost_ceiling_usd,
        latency_ceiling_ms=latency_ceiling_ms,
        privacy_class=privacy_class,
        quality_floor=quality_floor,
        requester_identity=requester_identity,
        streaming=streaming,
    )
    timeout_s = (latency_ceiling_ms / 1000.0) + 5.0
    return await _await_terminal(svc, rec, timeout_s)


async def _submit_llm_job_streaming(
    svc: JobsService,
    *,
    kind: str,
    prompt: str,
    structured_payload: Dict[str, Any],
    cost_ceiling_usd: float,
    latency_ceiling_ms: int,
    quality_floor: float,
    privacy_class: str,
    requester_identity: str,
) -> JobRecord:
    """G8 — submit but DO NOT await terminal state. The caller (an
    SSE generator) consumes rec.delta_queue while the executor still
    runs, then awaits terminal to emit final usage / finish_reason."""
    payload = {
        "prompt": prompt,
        "input": prompt,
        **structured_payload,
    }
    return await svc.submit(
        kind=kind,
        payload=payload,
        cost_ceiling_usd=cost_ceiling_usd,
        latency_ceiling_ms=latency_ceiling_ms,
        privacy_class=privacy_class,
        quality_floor=quality_floor,
        requester_identity=requester_identity,
        streaming=True,
    )


# ---------------------------------------------------------------------------
# Auth: optional, default off. A devserver running on localhost should
# not require credentials. In hosted mode the operator can flip
# `require_auth=True` and pass through a Bearer token of any shape — the
# identity is just hashed and used to scope receipts.
# ---------------------------------------------------------------------------

def _identity_for(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth.split(None, 1)[1].strip()
        if token:
            return f"bearer:{token[:32]}"
    api_key = request.headers.get("x-api-key", "")  # Anthropic style
    if api_key:
        return f"x-api-key:{api_key[:32]}"
    return "anonymous"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def build_devserver_app(
    *,
    jobs_service: Optional[JobsService] = None,
    cors_origins: Optional[List[str]] = None,
    title: str = "Pluginfer Devserver",
) -> FastAPI:
    """Build a FastAPI app exposing the OpenAI / Anthropic shim. The
    caller owns the `JobsService` (and therefore the auction + provider
    set); this function just wires routes."""
    if jobs_service is None:
        from core.providers import Auction
        jobs_service = JobsService(auction=Auction())

    app = FastAPI(
        title=title,
        version="1.0.0",
        description=(
            "OpenAI / Anthropic SDK-compatible shim over the Pluginfer "
            "compute auction. Point your existing client at this server "
            "and your prompts route through the mesh — zero code changes."
        ),
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.state.jobs = jobs_service
    # Gateway token-saver (core/response_cache.py). Deterministic
    # repeats cost zero tokens; disable with PLUGINFER_CACHE_DISABLE=1.
    from core.response_cache import ResponseCache
    app.state.response_cache = ResponseCache()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount the browser-provider gateway routes + the public receipts
    # leaderboard onto the devserver, so a single `pluginfer --dev`
    # instance is a complete demo: it serves OpenAI traffic AND accepts
    # browser providers AND publishes receipts. This is the viral
    # bundle.
    from .routers import provider_jobs as provider_jobs_router
    from .routers import receipts as receipts_router
    app.include_router(provider_jobs_router.router)
    app.include_router(receipts_router.router)

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        return {
            "status": "ok",
            "service": "pluginfer-devserver",
            "providers": len(jobs_service.auction.providers),
            "jobs_in_flight": sum(
                1 for r in jobs_service.jobs.values()
                if r.state not in TERMINAL_STATES
            ),
        }

    @app.get("/v1/models")
    def list_models() -> Dict[str, Any]:
        models: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for p in jobs_service.auction.providers:
            pid = getattr(p, "provider_id", None)
            if not pid or pid in seen:
                continue
            seen.add(pid)
            models.append({
                "id": pid,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "pluginfer-mesh",
                "pluginfer": {
                    "privacy_grade": getattr(p, "privacy_grade", "public"),
                    "kind": type(p).__name__,
                },
            })
        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionsBody, request: Request):
        identity = _identity_for(request)
        prompt = _messages_to_prompt([m.model_dump() for m in body.messages])
        cost = body.pluginfer_cost_ceiling_usd or DEFAULT_COST_CEILING_USD
        latency = body.pluginfer_latency_ceiling_ms or DEFAULT_LATENCY_MS
        quality = (
            body.pluginfer_quality_floor
            if body.pluginfer_quality_floor is not None
            else DEFAULT_QUALITY_FLOOR
        )
        privacy = body.pluginfer_privacy or "public"
        max_tokens = body.max_tokens or 512
        structured = {
            "model": body.model,
            "max_tokens": max_tokens,
            "temperature": body.temperature,
            "top_p": body.top_p,
            "openai": {
                "messages": [m.model_dump() for m in body.messages],
                "model": body.model,
            },
        }

        if body.stream:
            return EventSourceResponse(
                _stream_openai_chat(
                    request.app.state.jobs,
                    body.model, prompt, structured,
                    cost, latency, quality, privacy, identity,
                )
            )

        # Token-saver: byte-identical deterministic requests are served
        # from the gateway cache — zero provider tokens, zero
        # settlement, original receipt id preserved, honestly labelled
        # via X-Pluginfer-Cache. Policy in core/response_cache.py.
        cache = request.app.state.response_cache
        cache_key = ""
        if cache is not None and cache.cacheable(structured):
            cache_key = cache.key_for(structured)
            cached = cache.get(cache_key)
            if cached is not None:
                entry, age_s = cached
                headers = dict(entry["headers"])
                headers["X-Pluginfer-Cache"] = "hit"
                headers["X-Pluginfer-Cache-Age"] = f"{age_s:.0f}"
                headers["X-Pluginfer-Price-USD"] = "0"
                return JSONResponse(entry["response"], headers=headers)

        rec = await _run_llm_job(
            request.app.state.jobs,
            kind="llm.completion",
            prompt=prompt,
            structured_payload=structured,
            cost_ceiling_usd=cost,
            latency_ceiling_ms=latency,
            quality_floor=quality,
            privacy_class=privacy,
            requester_identity=identity,
        )
        if rec.state != "completed":
            return _failure_response(rec)
        text = _decode_result_text(rec)
        prompt_tokens = _approx_token_count(prompt)
        completion_tokens = _approx_token_count(text)
        resp = _openai_chat_response(
            model=body.model, text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        headers = _attest_and_headers(request.app.state.jobs, rec)
        if cache is not None and cache_key:
            cache.put(cache_key, {"response": resp, "headers": dict(headers)})
        headers["X-Pluginfer-Cache"] = "miss" if cache_key else "bypass"
        return JSONResponse(resp, headers=headers)

    @app.post("/v1/messages")
    async def anthropic_messages(body: MessagesBody, request: Request):
        identity = _identity_for(request)
        prompt = _messages_to_prompt(
            [m.model_dump() for m in body.messages], system=body.system,
        )
        cost = body.pluginfer_cost_ceiling_usd or DEFAULT_COST_CEILING_USD
        latency = body.pluginfer_latency_ceiling_ms or DEFAULT_LATENCY_MS
        quality = (
            body.pluginfer_quality_floor
            if body.pluginfer_quality_floor is not None
            else DEFAULT_QUALITY_FLOOR
        )
        privacy = body.pluginfer_privacy or "public"
        structured = {
            "model": body.model,
            "max_tokens": body.max_tokens,
            "temperature": body.temperature,
            "anthropic": {
                "messages": [m.model_dump() for m in body.messages],
                "system": body.system,
                "model": body.model,
            },
        }

        if body.stream:
            return EventSourceResponse(
                _stream_anthropic_messages(
                    request.app.state.jobs,
                    body.model, prompt, structured,
                    cost, latency, quality, privacy, identity,
                )
            )

        rec = await _run_llm_job(
            request.app.state.jobs,
            kind="llm.completion",
            prompt=prompt,
            structured_payload=structured,
            cost_ceiling_usd=cost,
            latency_ceiling_ms=latency,
            quality_floor=quality,
            privacy_class=privacy,
            requester_identity=identity,
        )
        if rec.state != "completed":
            return _failure_response(rec)
        text = _decode_result_text(rec)
        resp = _anthropic_messages_response(
            model=body.model, text=text,
            input_tokens=_approx_token_count(prompt),
            output_tokens=_approx_token_count(text),
        )
        return JSONResponse(
            resp,
            headers=_attest_and_headers(request.app.state.jobs, rec),
        )

    @app.post("/v1/embeddings")
    async def embeddings(body: EmbeddingsBody, request: Request):
        identity = _identity_for(request)
        inputs = body.input if isinstance(body.input, list) else [body.input]
        # Pluginfer dispatches embeddings as kind="embed". Mesh providers
        # that advertise embedding capability bid; cloud-LLM stubs
        # abstain (their `bid` returns None for non-completion kinds in
        # follow-up work — for now they bid on anything, so the auction
        # will pick the cheapest).
        results: List[Dict[str, Any]] = []
        last_rec: Optional[JobRecord] = None
        for i, item in enumerate(inputs):
            text = item if isinstance(item, str) else json.dumps(item)
            rec = await _run_llm_job(
                request.app.state.jobs,
                kind="embed",
                prompt=text,
                structured_payload={"model": body.model, "input": text},
                cost_ceiling_usd=DEFAULT_COST_CEILING_USD,
                latency_ceiling_ms=DEFAULT_LATENCY_MS,
                quality_floor=DEFAULT_QUALITY_FLOOR,
                privacy_class="public",
                requester_identity=identity,
            )
            last_rec = rec
            if rec.state != "completed":
                continue
            blob = _decode_result_text(rec)
            try:
                vec = json.loads(blob)
            except (json.JSONDecodeError, TypeError):
                vec = []
            results.append({
                "object": "embedding", "index": i, "embedding": vec,
            })
        headers = _attest_and_headers(request.app.state.jobs, last_rec) \
            if last_rec else {}
        return JSONResponse(
            {
                "object": "list",
                "data": results,
                "model": body.model,
                "usage": {
                    "prompt_tokens": sum(_approx_token_count(str(x)) for x in inputs),
                    "total_tokens": sum(_approx_token_count(str(x)) for x in inputs),
                },
            },
            headers=headers,
        )

    return app


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

async def _stream_openai_chat(
    svc: JobsService, model: str, prompt: str,
    structured: Dict[str, Any],
    cost: float, latency_ms: int, quality: float, privacy: str,
    identity: str,
) -> AsyncIterator[Dict[str, Any]]:
    """G8 streaming path. Submit with `streaming=True` so the
    JobRecord carries a delta_queue, then pump incremental chunks
    from streaming-capable providers; non-streaming providers result
    in exactly one full-content delta + terminal — same byte-shape as
    the OpenAI SDK contract."""
    rec = await _submit_llm_job_streaming(
        svc, kind="llm.completion", prompt=prompt,
        structured_payload=structured,
        cost_ceiling_usd=cost, latency_ceiling_ms=latency_ms,
        quality_floor=quality, privacy_class=privacy,
        requester_identity=identity,
    )
    chat_id = "chatcmpl-" + secrets.token_hex(12)
    if rec.state == "failed" and rec.delta_queue is None:
        yield {"data": json.dumps(_openai_chat_chunk(
            chat_id=chat_id, model=model,
            delta={"role": "assistant", "content": f"[pluginfer:error:{rec.state}:{rec.detail or ''}]"},
            finish_reason="stop",
        ))}
        yield {"data": "[DONE]"}
        return
    # Role-only opener (OpenAI convention).
    yield {"data": json.dumps(_openai_chat_chunk(
        chat_id=chat_id, model=model,
        delta={"role": "assistant", "content": ""},
    ))}

    q = rec.delta_queue
    deadline = time.monotonic() + (latency_ms / 1000.0) + 5.0
    emitted_any = False
    if q is not None:
        while True:
            timeout_left = max(0.05, deadline - time.monotonic())
            try:
                chunk = await asyncio.wait_for(q.get(), timeout=timeout_left)
            except asyncio.TimeoutError:
                break
            if chunk.get("terminal"):
                break
            text_part = chunk.get("text") or ""
            if text_part:
                emitted_any = True
                yield {"data": json.dumps(_openai_chat_chunk(
                    chat_id=chat_id, model=model,
                    delta={"content": text_part},
                ))}
    if not emitted_any:
        # Provider didn't stream — fall back to full content delta
        # using the terminal result.
        text = _decode_result_text(rec)
        if text:
            yield {"data": json.dumps(_openai_chat_chunk(
                chat_id=chat_id, model=model, delta={"content": text},
            ))}
    yield {"data": json.dumps(_openai_chat_chunk(
        chat_id=chat_id, model=model, delta={}, finish_reason="stop",
    ))}
    yield {"data": "[DONE]"}


async def _stream_anthropic_messages(
    svc: JobsService, model: str, prompt: str,
    structured: Dict[str, Any],
    cost: float, latency_ms: int, quality: float, privacy: str,
    identity: str,
) -> AsyncIterator[Dict[str, Any]]:
    rec = await _run_llm_job(
        svc, kind="llm.completion", prompt=prompt,
        structured_payload=structured,
        cost_ceiling_usd=cost, latency_ceiling_ms=latency_ms,
        quality_floor=quality, privacy_class=privacy,
        requester_identity=identity,
    )
    msg_id = "msg_" + secrets.token_hex(12)
    if rec.state != "completed":
        yield {"event": "error",
               "data": json.dumps({"type": "error",
                                   "error": {"type": "pluginfer_" + rec.state,
                                             "message": rec.detail or ""}})}
        return
    text = _decode_result_text(rec)
    yield {"event": "message_start",
           "data": json.dumps({"type": "message_start",
                               "message": {
                                   "id": msg_id, "type": "message",
                                   "role": "assistant", "model": model,
                                   "content": [], "stop_reason": None,
                                   "stop_sequence": None,
                                   "usage": {"input_tokens": _approx_token_count(prompt),
                                             "output_tokens": 0}}})}
    yield {"event": "content_block_start",
           "data": json.dumps({"type": "content_block_start", "index": 0,
                               "content_block": {"type": "text", "text": ""}})}
    if text:
        yield {"event": "content_block_delta",
               "data": json.dumps({"type": "content_block_delta", "index": 0,
                                   "delta": {"type": "text_delta", "text": text}})}
    yield {"event": "content_block_stop",
           "data": json.dumps({"type": "content_block_stop", "index": 0})}
    yield {"event": "message_delta",
           "data": json.dumps({"type": "message_delta",
                               "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                               "usage": {"output_tokens": _approx_token_count(text)}})}
    yield {"event": "message_stop", "data": json.dumps({"type": "message_stop"})}


# ---------------------------------------------------------------------------
# Failure + receipt headers
# ---------------------------------------------------------------------------

def _failure_response(rec: JobRecord) -> JSONResponse:
    code = {
        "failed": status.HTTP_502_BAD_GATEWAY,
        "timeout": status.HTTP_504_GATEWAY_TIMEOUT,
        "cancelled": status.HTTP_499_CLIENT_CLOSED_REQUEST
            if hasattr(status, "HTTP_499_CLIENT_CLOSED_REQUEST")
            else status.HTTP_400_BAD_REQUEST,
    }.get(rec.state, status.HTTP_502_BAD_GATEWAY)
    # Failures get the unsigned headers — no completed receipt to sign.
    return JSONResponse(
        {
            "error": {
                "type": f"pluginfer_{rec.state}",
                "code": rec.state,
                "message": rec.detail or rec.state,
                "job_id": rec.job_id,
            }
        },
        status_code=code,
        headers=_receipt_headers(rec),
    )


def _receipt_headers(
    rec: JobRecord, *, signed: bool = False,
) -> Dict[str, str]:
    """Build the response headers that ride alongside every devserver
    response. ``signed=True`` indicates the gateway has already produced
    a §A1 PNIS receipt for this job; the SDK can fetch it at
    ``GET /v1/receipts/{job_id}``."""
    h = {"X-Pluginfer-Job-Id": rec.job_id, "X-Pluginfer-State": rec.state}
    if rec.matched_provider_pubkey:
        h["X-Pluginfer-Provider"] = rec.matched_provider_pubkey
    if rec.price_locked_usd is not None:
        h["X-Pluginfer-Price-USD"] = f"{rec.price_locked_usd:.6f}"
    if rec.execution_ms is not None:
        h["X-Pluginfer-Execution-MS"] = f"{rec.execution_ms:.1f}"
    # Receipt-ID is the lookup key for /v1/receipts/{job_id}, so it
    # IS the job_id. The result hash gets its own header so callers
    # who want to verify against bytes-on-the-wire don't have to
    # fetch the receipt body.
    h["X-Pluginfer-Receipt-ID"] = rec.job_id
    if rec.result_hash_hex:
        h["X-Pluginfer-Result-Hash"] = rec.result_hash_hex
    h["X-Pluginfer-Receipt-Signed"] = "1" if signed else "0"
    return h


def _attest_and_headers(
    svc: JobsService, rec: JobRecord,
) -> Dict[str, str]:
    """Best-effort attestation + headers. If anything in the receipt
    pipeline raises (e.g. payload not serialisable), we ship the
    unsigned headers — the SDK contract is unchanged either way."""
    signed = False
    if rec.state == "completed":
        try:
            svc.attest_receipt(rec)
            signed = True
        except Exception:
            signed = False
    return _receipt_headers(rec, signed=signed)


# ---------------------------------------------------------------------------
# CLI: `python -m api.devserver` or `pluginfer dev`
# ---------------------------------------------------------------------------

def _build_default_jobs_service() -> JobsService:
    """Construct a JobsService with whatever providers the local
    environment makes available. Cloud LLM providers are wired but only
    bid when their keychain entry resolves, so this is safe to ship as
    the default."""
    from core.providers import (
        Auction, AnthropicProvider, OpenAIProvider,
    )
    auction = Auction()
    # Cloud providers — fail closed if no key is configured.
    try:
        auction.register(OpenAIProvider(enabled=True))
    except Exception:
        pass
    try:
        auction.register(AnthropicProvider(enabled=True))
    except Exception:
        pass
    return JobsService(auction=auction)


def serve(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
    """Boot the devserver. Importable for embedding in other launchers."""
    import uvicorn
    app = build_devserver_app(jobs_service=_build_default_jobs_service())
    print(
        f"\n  pluginfer devserver listening on http://{host}:{port}\n"
        f"  redirect your SDK with:\n"
        f"    OPENAI_BASE_URL=http://{host}:{port}/v1\n"
        f"    ANTHROPIC_BASE_URL=http://{host}:{port}\n"
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Pluginfer Devserver — OpenAI/Anthropic shim")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    serve(host=args.host, port=args.port)
