"""G6 — Sybil resistance for browser-tab providers.

Three layered defences:
  1. Per-/24 token-bucket rate limit on register/heartbeat ops.
  2. WebGPU fingerprint Sybil detector (same fingerprint across many
     subnets in a short window -> block).
  3. Stake-to-register tier promotion (untrusted / staked / verified).

These tests exercise each defence in isolation + the gateway
integration that wires them up.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import httpx  # noqa: E402

from api.main import build_app  # noqa: E402
from core.providers import Auction  # noqa: E402
from core.sybil_guard import (  # noqa: E402
    FingerprintSybilDetector,
    PerSubnetRateLimiter,
    TierResult,
    fingerprint_hash,
    resolve_tier,
    subnet_of,
)


# ---------------------------------------------------------------------------
# Pure unit tests
# ---------------------------------------------------------------------------

def test_subnet_of_ipv4():
    assert subnet_of("203.0.113.42") == "203.0.113.0/24"
    assert subnet_of("198.51.100.5") == "198.51.100.0/24"


def test_subnet_of_ipv6_collapses_to_48():
    assert subnet_of("2001:db8:1234:5678::1") == "2001:db8:1234::/48"


def test_subnet_of_empty():
    assert subnet_of("") == ""


def test_fingerprint_hash_is_stable():
    h1 = fingerprint_hash("NVIDIA", "Ampere", "RTX 4090", "550.40")
    h2 = fingerprint_hash("NVIDIA", "Ampere", "RTX 4090", "550.40")
    assert h1 == h2
    h3 = fingerprint_hash("NVIDIA", "Ampere", "RTX 3090", "550.40")
    assert h3 != h1


def test_fingerprint_hash_with_missing_parts():
    assert fingerprint_hash(None, None, None, None) == \
        fingerprint_hash("", "", "", "")


def test_resolve_tier_thresholds():
    assert resolve_tier(stake_plg=0.0, turnstile_ok=False).tier == "untrusted"
    assert resolve_tier(stake_plg=0.5, turnstile_ok=False).tier == "untrusted"
    assert resolve_tier(stake_plg=2.0, turnstile_ok=False).tier == "staked"
    assert resolve_tier(stake_plg=2.0, turnstile_ok=True).tier == "verified"
    assert resolve_tier(stake_plg=0.5, turnstile_ok=True).tier == "untrusted"


def test_rate_limiter_caps_per_subnet():
    rl = PerSubnetRateLimiter(capacity=3, refill_per_sec=0.0)
    # First 3 from one subnet succeed.
    for _ in range(3):
        assert rl.allow("203.0.113.5") is True
    # 4th from the same subnet -> blocked.
    assert rl.allow("203.0.113.5") is False
    # Different subnet has its own bucket.
    assert rl.allow("198.51.100.5") is True


def test_rate_limiter_refills_over_time():
    rl = PerSubnetRateLimiter(capacity=2, refill_per_sec=1.0)
    # Burn the bucket at t=0.
    assert rl.allow("203.0.113.5", now=0.0) is True
    assert rl.allow("203.0.113.5", now=0.0) is True
    assert rl.allow("203.0.113.5", now=0.0) is False
    # After 1 second a token is back.
    assert rl.allow("203.0.113.5", now=1.0) is True


def test_fingerprint_sybil_detector_under_threshold_clean():
    d = FingerprintSybilDetector(window_s=60, max_subnets=3)
    fp = "abc123"
    assert d.record_and_check(fp, "203.0.113.5") is True
    assert d.record_and_check(fp, "198.51.100.5") is True
    assert d.record_and_check(fp, "192.0.2.5") is True


def test_fingerprint_sybil_detector_trips_over_threshold():
    d = FingerprintSybilDetector(window_s=60, max_subnets=2)
    fp = "abc123"
    assert d.record_and_check(fp, "203.0.113.5") is True
    assert d.record_and_check(fp, "198.51.100.5") is True
    # 3rd subnet -> trips.
    assert d.record_and_check(fp, "192.0.2.5") is False


def test_fingerprint_sybil_detector_forgets_after_window():
    d = FingerprintSybilDetector(window_s=10, max_subnets=2)
    fp = "abc123"
    d.record_and_check(fp, "203.0.113.5", now=0.0)
    d.record_and_check(fp, "198.51.100.5", now=0.0)
    # Past the window the old entries age out.
    assert d.record_and_check(fp, "192.0.2.5", now=11.0) is True


# ---------------------------------------------------------------------------
# Gateway integration: register endpoint enforces the guards
# ---------------------------------------------------------------------------

SAMPLE_PEM = (
    "-----BEGIN PUBLIC KEY-----\n"
    "MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE" + ("A" * 80) + "\n"
    "-----END PUBLIC KEY-----\n"
)


def _make_app():
    auction = Auction()
    app = build_app(
        auction=auction,
        rate_limit_capacity=5_000.0,
        rate_limit_refill_per_sec=10_000.0,
    )
    return app


def test_register_default_tier_is_untrusted():
    app = _make_app()

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test",
        ) as c:
            r = await c.post("/v1/providers/register", json={
                "provider_pubkey_pem": SAMPLE_PEM,
                "hardware_class": "browser-webgpu",
                "price_per_1k_tok_usd": 0.0001,
                "base_eta_ms": 100,
                "base_quality": 0.9,
                "privacy_grade": "public",
            })
            assert r.status_code == 201
            body = r.json()
            assert body["tier"] == "untrusted"
            assert body["stake_plg"] == 0.0
    asyncio.run(_run())


def test_register_with_turnstile_promotes_when_staked():
    """A staking-only path can't promote to verified — needs both
    stake + turnstile token."""
    app = _make_app()
    # Pre-populate a fake ledger with a stake-tx that resolves to 5 PLG.
    class _Ledger:
        @staticmethod
        def get_transaction_by_id(_tx_id):
            return type("Tx", (), {"amount": 5.0})()
    app.state.ledger = _Ledger()

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test",
        ) as c:
            r = await c.post("/v1/providers/register", json={
                "provider_pubkey_pem": SAMPLE_PEM,
                "stake_tx_id": "tx-deadbeef",
                "turnstile_token": "TURNSTILE-OK",
            })
            assert r.status_code == 201, r.text
            body = r.json()
            assert body["tier"] == "verified"
            assert body["stake_plg"] == 5.0
    asyncio.run(_run())


def test_register_rate_limit_kicks_in():
    """Hammer the register endpoint with the same source IP and watch
    it return 429 once the per-/24 bucket empties."""
    import os
    os.environ["PLUGINFER_PER_SUBNET_OPS_PER_MIN"] = "3"
    # Force re-import to pick up the env var.
    import importlib
    import core.sybil_guard
    importlib.reload(core.sybil_guard)

    app = _make_app()

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test",
        ) as c:
            ok_count = 0
            block_count = 0
            for i in range(8):
                # Each pubkey unique so we don't trip dedup; same IP
                # so the /24 bucket counts them all together.
                pem = SAMPLE_PEM.replace("AAA", f"A{i:02d}")
                r = await c.post(
                    "/v1/providers/register",
                    headers={"CF-Connecting-IP": "203.0.113.99"},
                    json={
                        "provider_pubkey_pem": pem,
                    },
                )
                if r.status_code == 201:
                    ok_count += 1
                elif r.status_code == 429:
                    block_count += 1
            assert ok_count >= 1
            assert block_count >= 1
    asyncio.run(_run())
    # Reset for clean state.
    os.environ.pop("PLUGINFER_PER_SUBNET_OPS_PER_MIN", None)
    importlib.reload(core.sybil_guard)


def test_register_fingerprint_sybil_block():
    """Same fingerprint registering from 5+ different subnets in a
    short window -> Sybil block."""
    import os
    os.environ["PLUGINFER_FP_SYBIL_MAX_SUBNETS"] = "2"
    os.environ["PLUGINFER_PER_SUBNET_OPS_PER_MIN"] = "1000"
    import importlib
    import core.sybil_guard
    importlib.reload(core.sybil_guard)

    app = _make_app()
    fp_fields = {
        "webgpu_vendor": "NVIDIA",
        "webgpu_architecture": "Ada",
        "webgpu_device": "RTX 4090",
        "webgpu_driver": "550.40.07",
    }

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test",
        ) as c:
            results = []
            for i, ip in enumerate([
                "203.0.113.1",
                "198.51.100.1",
                "192.0.2.1",        # 3rd subnet should trip
                "10.0.0.1",
            ]):
                pem = SAMPLE_PEM.replace("AAA", f"A{i:02d}")
                r = await c.post(
                    "/v1/providers/register",
                    headers={"CF-Connecting-IP": ip},
                    json={
                        "provider_pubkey_pem": pem,
                        **fp_fields,
                    },
                )
                results.append(r.status_code)
            # First 2 should succeed (201), the rest 429.
            assert results[0] == 201
            assert results[1] == 201
            assert 429 in results[2:]
    asyncio.run(_run())

    os.environ.pop("PLUGINFER_FP_SYBIL_MAX_SUBNETS", None)
    os.environ.pop("PLUGINFER_PER_SUBNET_OPS_PER_MIN", None)
    importlib.reload(core.sybil_guard)
