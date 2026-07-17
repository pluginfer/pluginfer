"""G13 — waitlist capture endpoint."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402


def test_waitlist_appends_to_jsonl_log(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PLUGINFER_WAITLIST_LOG_PATH",
        str(tmp_path / "waitlist.jsonl"),
    )
    from fastapi import FastAPI
    # Late import so the env var sticks.
    from growth.waitlist_capture import router
    app = FastAPI()
    app.include_router(router)

    async def _run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            r = await c.post("/api/waitlist", json={
                "email": "user@example.com",
                "hardware_class": "consumer-gpu-mid",
                "use_case": "langchain-agent",
            })
            assert r.status_code == 201, r.text
            assert r.json()["queued"] is True
    asyncio.run(_run())

    log = Path(tmp_path / "waitlist.jsonl").read_text()
    line = json.loads(log.splitlines()[0])
    assert line["email"] == "user@example.com"
    assert line["hardware_class"] == "consumer-gpu-mid"
    assert line["use_case"] == "langchain-agent"
    assert line["submitted_at_unix"] > 0


def test_waitlist_rejects_malformed_email(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PLUGINFER_WAITLIST_LOG_PATH",
        str(tmp_path / "waitlist.jsonl"),
    )
    from fastapi import FastAPI
    from growth.waitlist_capture import router
    app = FastAPI()
    app.include_router(router)

    async def _run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            r = await c.post("/api/waitlist", json={"email": "not-an-email"})
            assert r.status_code == 422
    asyncio.run(_run())
