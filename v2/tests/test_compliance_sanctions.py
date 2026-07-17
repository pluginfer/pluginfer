"""G5 — OFAC + EU + UN sanctions screening + region screening.

The legal blocker is real: every USD that flows through Pluginfer
between pseudonymous wallets is the operator's strict-liability
exposure under the OFAC SDN regime. These tests pin the screen.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import httpx  # noqa: E402

from api.main import build_app  # noqa: E402
from core.compliance import (  # noqa: E402
    BLOCKED_COUNTRY_CODES,
    SanctionsRegistry,
    is_sanctioned_address,
    is_sanctioned_region,
    screen_auction_participants,
)
from core.compliance.sanctions import _to_address  # noqa: E402
from core.providers import Auction  # noqa: E402


SAMPLE_PUBKEY = (
    "-----BEGIN PUBLIC KEY-----\n"
    "MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE" + ("A" * 80) + "\n"
    "-----END PUBLIC KEY-----\n"
)
SAMPLE_ADDR = _to_address(SAMPLE_PUBKEY)


# ---------------------------------------------------------------------------
# Address screening
# ---------------------------------------------------------------------------

def test_address_screen_clean_when_list_empty():
    """Default registry path resolves a populated-but-comments-only
    file. A clean address must be allowed."""
    r = is_sanctioned_address(SAMPLE_PUBKEY)
    assert r.allowed is True


def test_address_screen_blocks_when_listed(tmp_path):
    """Pre-populated registry with the sample address → screen denies
    with the matched list label."""
    data_dir = tmp_path / "data" / "sanctions"
    data_dir.mkdir(parents=True)
    sdn = data_dir / "ofac_sdn_addresses.txt"
    sdn.write_text(f"{SAMPLE_ADDR}\n", encoding="utf-8")

    reg = SanctionsRegistry(data_dir=data_dir)
    assert reg.total == 1

    r = is_sanctioned_address(SAMPLE_PUBKEY, registry=reg)
    assert r.allowed is False
    assert r.matched_list == "OFAC-SDN"
    assert r.matched_address == SAMPLE_ADDR


def test_address_screen_blocks_via_raw_address_form(tmp_path):
    """The caller can pass the already-derived address (lowercase hex)
    rather than the PEM; we still match."""
    data_dir = tmp_path / "data" / "sanctions"
    data_dir.mkdir(parents=True)
    (data_dir / "ofac_sdn_addresses.txt").write_text(
        f"{SAMPLE_ADDR}\n", encoding="utf-8"
    )
    reg = SanctionsRegistry(data_dir=data_dir)
    r = is_sanctioned_address(SAMPLE_ADDR, registry=reg)
    assert r.allowed is False


# ---------------------------------------------------------------------------
# Region screening
# ---------------------------------------------------------------------------

def test_region_screen_blocks_comprehensively_sanctioned():
    for cc in ("IR", "KP", "CU", "SY", "RU", "BY"):
        r = is_sanctioned_region(cc)
        assert r.allowed is False, cc
        assert r.matched_list == "REGION"
        assert r.matched_country == cc
        assert cc in BLOCKED_COUNTRY_CODES


def test_region_screen_allows_safe_countries():
    for cc in ("US", "GB", "IN", "DE", "JP"):
        r = is_sanctioned_region(cc)
        assert r.allowed is True, cc


def test_region_screen_blocks_subdivision_only():
    """Crimea is in Ukraine (UA — not country-blocked) but the
    subdivision must be rejected."""
    r = is_sanctioned_region("UA", subdivision="UA-43")
    assert r.allowed is False
    assert r.matched_country == "UA-43"


# ---------------------------------------------------------------------------
# Auction-participant screen
# ---------------------------------------------------------------------------

def test_screen_auction_blocks_listed_buyer(tmp_path):
    data_dir = tmp_path / "data" / "sanctions"
    data_dir.mkdir(parents=True)
    (data_dir / "ofac_sdn_addresses.txt").write_text(
        f"{SAMPLE_ADDR}\n", encoding="utf-8"
    )
    reg = SanctionsRegistry(data_dir=data_dir)
    r = screen_auction_participants(
        buyer_pubkey_pem=SAMPLE_PUBKEY,
        provider_pubkey_pem=None,
        registry=reg,
    )
    assert r.allowed is False
    assert "buyer" in (r.reason or "")


def test_screen_auction_blocks_sanctioned_provider_country():
    r = screen_auction_participants(
        buyer_pubkey_pem=None,
        provider_pubkey_pem=None,
        buyer_country_code="US",
        provider_country_code="IR",
    )
    assert r.allowed is False
    assert "provider" in (r.reason or "")
    assert r.matched_country == "IR"


def test_screen_auction_allows_clean_participants():
    r = screen_auction_participants(
        buyer_pubkey_pem=SAMPLE_PUBKEY,
        provider_pubkey_pem=SAMPLE_PUBKEY,
        buyer_country_code="US",
        provider_country_code="IN",
    )
    assert r.allowed is True


# ---------------------------------------------------------------------------
# Gateway integration — sanctioned register POST returns 451
# ---------------------------------------------------------------------------

def test_register_endpoint_returns_451_for_sanctioned_region():
    auction = Auction()
    app = build_app(
        auction=auction,
        rate_limit_capacity=5_000.0,
        rate_limit_refill_per_sec=10_000.0,
    )

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test",
        ) as c:
            r = await c.post(
                "/v1/providers/register",
                headers={"CF-IPCountry": "IR"},
                json={
                    "provider_pubkey_pem": SAMPLE_PUBKEY,
                    "hardware_class": "browser-webgpu",
                    "price_per_1k_tok_usd": 0.0001,
                    "base_eta_ms": 100,
                    "base_quality": 0.9,
                    "privacy_grade": "public",
                },
            )
            assert r.status_code == 451, r.text
            assert "sanctioned" in (r.json().get("detail") or "")
    asyncio.run(_run())


def test_register_endpoint_succeeds_for_clean_country():
    auction = Auction()
    app = build_app(
        auction=auction,
        rate_limit_capacity=5_000.0,
        rate_limit_refill_per_sec=10_000.0,
    )

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test",
        ) as c:
            r = await c.post(
                "/v1/providers/register",
                headers={"CF-IPCountry": "US"},
                json={
                    "provider_pubkey_pem": SAMPLE_PUBKEY,
                    "hardware_class": "browser-webgpu",
                    "price_per_1k_tok_usd": 0.0001,
                    "base_eta_ms": 100,
                    "base_quality": 0.9,
                    "privacy_grade": "public",
                },
            )
            assert r.status_code == 201
    asyncio.run(_run())
