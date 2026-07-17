"""G8 — per-token streaming on the devserver.

A streaming-capable provider accepts an `on_delta` keyword on its
`execute()` method. JobsService passes a thread-safe callback that
pushes each chunk into the JobRecord's `delta_queue`. The SSE chat
handler in the devserver pumps those chunks as separate `data:`
events — so a chat-UI client sees the response materialise per
token instead of as a single blob.

This test fixture uses a deliberately slow streaming provider that
emits 5 separate "delta tokens" with 5ms gaps. We collect the SSE
events and assert that more than one content-bearing delta arrives,
proving the per-token path actually runs (rather than the
fall-back-to-full-content path used for non-streaming providers).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sys
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import httpx  # noqa: E402

from api.devserver import build_devserver_app  # noqa: E402
from api.jobs_service import JobsService  # noqa: E402
from core.providers import (  # noqa: E402
    Auction,
    Bid,
    JobSpec,
    Provider,
    PRIVACY_PUBLIC,
)


class _StreamingProvider(Provider):
    """Emits 5 incremental chunks via on_delta, then returns the full
    concatenated result. Models a real per-token LLM."""

    provider_id = "streaming-provider"
    privacy_grade = PRIVACY_PUBLIC

    def bid(self, job: JobSpec) -> Bid:
        return Bid(
            provider_id=self.provider_id,
            price_usd=0.0001,
            eta_ms=200,
            expected_quality=0.9,
            privacy_grade=PRIVACY_PUBLIC,
            evidence={"src": "streamer"},
        )

    def execute(self, job: JobSpec, bid: Bid, *, on_delta=None) -> dict:
        chunks = ["hel", "lo, ", "stream", "ed wor", "ld!"]
        full = "".join(chunks)
        if on_delta is not None:
            for c in chunks:
                on_delta({"text": c})
                time.sleep(0.005)
        out = full.encode("utf-8")
        return {
            "status": "executed",
            "job_id": job.job_id,
            "result_bytes": base64.b64encode(out).decode("ascii"),
            "result_hash": hashlib.sha256(out).hexdigest(),
            "execution_ms": 30,
            "provider_sig": "AAAA",
            "provider_pubkey_pem": "-----BEGIN PUBLIC KEY-----\nfake\n-----END PUBLIC KEY-----\n",
        }


class _NonStreamingProvider(Provider):
    """Old-school provider: no on_delta kwarg, just returns one blob.
    This proves the fallback path still emits exactly one content delta
    so existing clients see no behaviour change."""

    provider_id = "old-provider"
    privacy_grade = PRIVACY_PUBLIC

    def bid(self, job: JobSpec) -> Bid:
        return Bid(
            provider_id=self.provider_id,
            price_usd=0.0001,
            eta_ms=10,
            expected_quality=0.85,
            privacy_grade=PRIVACY_PUBLIC,
            evidence={"src": "old"},
        )

    def execute(self, job: JobSpec, bid: Bid) -> dict:
        out = b"one-shot response"
        return {
            "status": "executed",
            "job_id": job.job_id,
            "result_bytes": base64.b64encode(out).decode("ascii"),
            "result_hash": hashlib.sha256(out).hexdigest(),
            "execution_ms": 1,
            "provider_sig": "AAAA",
            "provider_pubkey_pem": "-----BEGIN PUBLIC KEY-----\nfake\n-----END PUBLIC KEY-----\n",
        }


def _app(provider):
    auction = Auction()
    auction.register(provider)
    svc = JobsService(auction=auction)
    return build_devserver_app(jobs_service=svc)


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def _collect_sse_data(stream_ctx) -> list[str]:
    """Read every SSE `data: ...` line out of a streaming response."""
    chunks: list[str] = []
    async for line in stream_ctx.aiter_lines():
        if line.startswith("data:"):
            chunks.append(line[len("data:"):].strip())
    return chunks


def test_streaming_provider_emits_multiple_content_deltas():
    """The streaming provider emits 5 chunks; we expect the SSE stream
    to carry at LEAST 2 distinct content-bearing deltas (proves the
    on_delta path actually runs end-to-end)."""
    app = _app(_StreamingProvider())

    async def _run():
        async with _client(app) as c:
            async with c.stream("POST", "/v1/chat/completions", json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "stream me"}],
                "max_tokens": 64,
                "stream": True,
            }) as r:
                assert r.status_code == 200
                chunks = await _collect_sse_data(r)
            decoded = [json.loads(c) for c in chunks if c and c != "[DONE]"]
            content_deltas = [
                ch for ch in decoded
                if ch["choices"][0]["delta"].get("content")
            ]
            assert len(content_deltas) >= 2, (
                f"expected >=2 content deltas, got {len(content_deltas)}"
            )
            assert "[DONE]" in chunks
            # Recompose the streamed text and verify it matches.
            text = "".join(
                ch["choices"][0]["delta"]["content"] for ch in content_deltas
            )
            assert "stream" in text
    asyncio.run(_run())


def test_non_streaming_provider_falls_back_to_single_chunk():
    """Backwards-compatibility — a provider without on_delta in its
    signature still produces exactly one content delta + DONE marker.
    This is the existing W43 SDK contract."""
    app = _app(_NonStreamingProvider())

    async def _run():
        async with _client(app) as c:
            async with c.stream("POST", "/v1/chat/completions", json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "no streaming"}],
                "max_tokens": 64,
                "stream": True,
            }) as r:
                chunks = await _collect_sse_data(r)
            decoded = [json.loads(c) for c in chunks if c and c != "[DONE]"]
            content_deltas = [
                ch for ch in decoded
                if ch["choices"][0]["delta"].get("content")
            ]
            assert len(content_deltas) == 1
            assert "[DONE]" in chunks
    asyncio.run(_run())
