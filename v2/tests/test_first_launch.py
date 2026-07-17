"""G10 — first-launch profile builder + onboarding router."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import httpx  # noqa: E402

from core.first_launch import (  # noqa: E402
    FirstLaunchProfile,
    TIER_TO_EARNINGS_ID,
    build_profile,
)


def test_build_profile_returns_complete_shape(tmp_path):
    prof = build_profile(
        idle_hours_per_day=8.0,
        power_cost_usd_per_kwh=0.12,
        wallet_path=str(tmp_path / "wallet.pem"),
    )
    d = prof.to_dict()
    assert d["schema"] == "pluginfer-first-launch/v1"
    for key in (
        "hardware", "earnings", "game_detection", "wallet",
        "compliance", "next_steps",
    ):
        assert key in d, key
    assert "tier" in d["hardware"]
    assert "expected_usd_per_day" in d["earnings"]


def test_build_profile_flags_sanctioned_region(tmp_path):
    prof = build_profile(
        country_code_hint="IR",
        wallet_path=str(tmp_path / "wallet.pem"),
    )
    d = prof.to_dict()
    assert d["compliance"]["allowed"] is False
    # The first step explains why.
    assert any("sanctioned" in s.lower() for s in d["next_steps"])


def test_build_profile_allows_clean_region(tmp_path):
    prof = build_profile(
        country_code_hint="US",
        wallet_path=str(tmp_path / "wallet.pem"),
    )
    d = prof.to_dict()
    assert d["compliance"]["allowed"] is True
    # When wallet doesn't exist, the first step should be wallet-creation.
    assert any("wallet" in s.lower() for s in d["next_steps"])


def test_tier_to_earnings_id_covers_every_known_tier():
    for tier in ("high-end", "mid-range", "entry-level", "no-gpu", "unknown"):
        assert tier in TIER_TO_EARNINGS_ID


# ---------------------------------------------------------------------------
# Router integration
# ---------------------------------------------------------------------------

def test_router_profile_endpoint_returns_shape():
    from fastapi import FastAPI
    from api.routers import first_launch as fl
    app = FastAPI()
    app.include_router(fl.router)

    async def _run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            r = await c.get(
                "/v1/first_launch/profile",
                headers={"CF-IPCountry": "US"},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["compliance"]["allowed"] is True
            assert "hardware" in body
            assert "earnings" in body
    asyncio.run(_run())


def test_router_respects_country_header_for_blocked_region():
    from fastapi import FastAPI
    from api.routers import first_launch as fl
    app = FastAPI()
    app.include_router(fl.router)

    async def _run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            r = await c.get(
                "/v1/first_launch/profile",
                headers={"CF-IPCountry": "KP"},
            )
            assert r.status_code == 200
            body = r.json()
            assert body["compliance"]["allowed"] is False
    asyncio.run(_run())
