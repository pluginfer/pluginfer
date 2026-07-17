"""End-to-end tests for the Pluginfer Devserver — the OpenAI/Anthropic
SDK shim.

We drive the FastAPI app via httpx.ASGITransport so the full pipeline
(schema parsing, auction, dispatch, response shaping) runs in-process,
with a deterministic FakeProvider that always wins.

The asserts deliberately check that the response shape is BYTE-COMPATIBLE
with the upstream SDK contract — that's the entire point of the shim.

Test style follows the project's ``asyncio.run(_run())`` pattern (see
test_api.py) so we don't take a pytest-asyncio dependency.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sys
from pathlib import Path

import pytest

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


class _EchoProvider(Provider):
    """Always wins the auction; returns the prompt UTF-8-encoded as the
    result blob. The devserver decodes that as the assistant's text."""
    provider_id = "echo-provider"
    privacy_grade = PRIVACY_PUBLIC

    def bid(self, job: JobSpec) -> Bid:
        return Bid(
            provider_id=self.provider_id,
            price_usd=0.0001,
            eta_ms=20,
            expected_quality=0.99,
            privacy_grade=PRIVACY_PUBLIC,
            evidence={"src": "echo"},
        )

    def execute(self, job: JobSpec, bid: Bid) -> dict:
        prompt = (job.payload.get("prompt") or "")
        # The shim collapses messages into "<role>: <text>" lines and
        # appends "assistant:" as a generation cue. Pull the most recent
        # user line back out for the echo response.
        last_user = ""
        for line in prompt.splitlines():
            if line.startswith("user:"):
                last_user = line.split(":", 1)[-1].strip()
        text = f"echo: {last_user}"
        out = text.encode("utf-8")
        return {
            "status": "executed",
            "job_id": job.job_id,
            "result_text": text,
            "result_bytes": base64.b64encode(out).decode("ascii"),
            "result_hash": hashlib.sha256(out).hexdigest(),
            "execution_ms": 5,
            "provider_sig": "AAAA",
            "provider_pubkey_pem": "-----BEGIN PUBLIC KEY-----\nfake\n-----END PUBLIC KEY-----\n",
        }


@pytest.fixture
def app():
    auction = Auction()
    auction.register(_EchoProvider())
    svc = JobsService(auction=auction)
    return build_devserver_app(jobs_service=svc)


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def test_healthz_reports_provider_count(app):
    async def _run():
        async with _client(app) as c:
            r = await c.get("/healthz")
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "ok"
            assert body["providers"] == 1
    asyncio.run(_run())


def test_models_lists_registered_providers(app):
    async def _run():
        async with _client(app) as c:
            r = await c.get("/v1/models")
            assert r.status_code == 200
            body = r.json()
            assert body["object"] == "list"
            ids = [m["id"] for m in body["data"]]
            assert "echo-provider" in ids
    asyncio.run(_run())


def test_chat_completions_returns_openai_shape(app):
    async def _run():
        async with _client(app) as c:
            r = await c.post("/v1/chat/completions", json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello world"}],
                "max_tokens": 64,
            })
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["object"] == "chat.completion"
            assert body["model"] == "gpt-4o-mini"
            assert isinstance(body["choices"], list) and len(body["choices"]) == 1
            msg = body["choices"][0]["message"]
            assert msg["role"] == "assistant"
            assert "echo" in msg["content"]
            assert "hello world" in msg["content"]
            assert "usage" in body and "prompt_tokens" in body["usage"]
            assert r.headers.get("x-pluginfer-job-id")
            assert r.headers.get("x-pluginfer-provider") == "echo-provider"
    asyncio.run(_run())


def test_chat_completions_handles_multimodal_content_blocks(app):
    """OpenAI lets `content` be a list of typed blocks. The shim must
    flatten text blocks and ignore image blocks gracefully."""
    async def _run():
        async with _client(app) as c:
            r = await c.post("/v1/chat/completions", json={
                "model": "gpt-4o",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this:"},
                        {"type": "image_url", "image_url": {"url": "data:..."}},
                        {"type": "text", "text": "the image"},
                    ],
                }],
                "max_tokens": 32,
            })
            assert r.status_code == 200
            text = r.json()["choices"][0]["message"]["content"]
            # Image block dropped, text blocks combined.
            assert "image" in text or "describe" in text
    asyncio.run(_run())


def test_chat_completions_pluginfer_overrides_pin_auction(app):
    """Caller can pin auction parameters without leaving the OpenAI
    schema by using `pluginfer_*` extension fields."""
    async def _run():
        async with _client(app) as c:
            r = await c.post("/v1/chat/completions", json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 16,
                "pluginfer_cost_ceiling_usd": 0.001,
                "pluginfer_latency_ceiling_ms": 5_000,
                "pluginfer_quality_floor": 0.5,
                "pluginfer_privacy": "public",
            })
            assert r.status_code == 200
    asyncio.run(_run())


def test_anthropic_messages_returns_anthropic_shape(app):
    async def _run():
        async with _client(app) as c:
            r = await c.post("/v1/messages", json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi claude"}],
            })
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["type"] == "message"
            assert body["role"] == "assistant"
            assert body["model"] == "claude-haiku-4-5-20251001"
            assert isinstance(body["content"], list)
            block = body["content"][0]
            assert block["type"] == "text"
            assert "hi claude" in block["text"]
            assert "input_tokens" in body["usage"]
            assert "output_tokens" in body["usage"]
    asyncio.run(_run())


def test_anthropic_messages_handles_system_string(app):
    async def _run():
        async with _client(app) as c:
            r = await c.post("/v1/messages", json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 32,
                "system": "You are a pirate.",
                "messages": [{"role": "user", "content": "hello"}],
            })
            assert r.status_code == 200
    asyncio.run(_run())


def test_chat_completions_streaming_emits_done_marker(app):
    async def _run():
        async with _client(app) as c:
            async with c.stream("POST", "/v1/chat/completions", json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "stream me"}],
                "max_tokens": 32,
                "stream": True,
            }) as r:
                assert r.status_code == 200
                chunks = []
                async for line in r.aiter_lines():
                    if line.startswith("data:"):
                        chunks.append(line[len("data:"):].strip())
            assert "[DONE]" in chunks
            decoded = [json.loads(c) for c in chunks if c and c != "[DONE]"]
            assert decoded[0]["choices"][0]["delta"].get("role") == "assistant"
            assert any(
                c["choices"][0]["delta"].get("content")
                for c in decoded
            )
            assert decoded[-1]["choices"][0]["finish_reason"] == "stop"
    asyncio.run(_run())


def test_no_provider_returns_5xx(app):
    """Drop the provider, hit the shim — must fail loudly, not return
    an empty 200."""
    async def _run():
        app.state.jobs.auction.providers.clear()
        async with _client(app) as c:
            r = await c.post("/v1/chat/completions", json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "x"}],
                "max_tokens": 16,
            })
        assert r.status_code in (502, 504)
        body = r.json()
        assert body["error"]["type"].startswith("pluginfer_")
    asyncio.run(_run())
