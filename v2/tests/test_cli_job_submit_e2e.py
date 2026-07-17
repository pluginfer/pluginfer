"""W37 — submit-job ↔ JobsService end-to-end (no HTTP, no daemon).

Cases:
  1. LocalFederationProvider bids on inference when a backend is up.
  2. With a stub backend, JobsService.submit transitions queued ->
     matched -> running -> completed and the result hash matches the
     provider output.
  3. Sign path: when wallet is provided, executed result carries
     provider_sig + provider_pubkey_pem.
  4. No backend available -> bid is None -> auction yields no winner ->
     job state goes to 'failed' with detail 'no_provider_matched'.
  5. Privacy mapping: privacy_class='sensitive' propagates to LOCAL_ONLY
     federation privacy_mode.

These cases are the load-bearing assertions for the CLI's new path —
`ai/filum/cli.py:_job_submit_through_jobs_service` runs the same
JobsService used by the FastAPI router, so once these green-light, the
CLI is on the same execution path as the REST API.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest  # noqa: E402

from api.jobs_service import JobsService  # noqa: E402
from core.providers import Auction  # noqa: E402

from ai.filum.hpa.federation_provider import LocalFederationProvider  # noqa: E402


# ---------------------------------------------------------------------------
# Stub federation — no Ollama, no torch, no network. The point of these
# tests is to exercise the auction + JobsService wiring, not the LLM.
# ---------------------------------------------------------------------------


@dataclass
class _StubResp:
    text: str = "stub-result-from-test"
    model_id: str = "stub-model"
    backend_name: str = "stub_backend"
    elapsed_s: float = 0.001
    receipt_id: str = "rcpt-stub"
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {"tokens_generated": 1}


class _StubFederation:
    """Minimal stand-in for ModelFederation. list_available() returns
    truthy so the provider bids; generate() returns a deterministic
    response so the test asserts on hash."""

    def __init__(self, available: bool = True, payload_text: str = "stub-result-from-test"):
        self._available = available
        self._payload = payload_text
        self.last_request = None

    def list_available(self):
        if not self._available:
            return []
        return [{"backend": "stub", "is_local": True, "models": ["stub-model"]}]

    def generate(self, req):
        self.last_request = req
        return _StubResp(text=self._payload)


def _build_service(*, backend_available: bool = True,
                    payload_text: str = "stub-result-from-test",
                    wallet=None):
    fed = _StubFederation(available=backend_available, payload_text=payload_text)
    prov = LocalFederationProvider(
        provider_id="local-fed-test",
        wallet=wallet,
        federation_factory=lambda: fed,
        base_eta_ms=10,
        base_quality=0.9,
    )
    auction = Auction()
    auction.register(prov)
    return JobsService(auction=auction), fed, prov


# ---------------------------------------------------------------------------
# 1. Bid path
# ---------------------------------------------------------------------------


def test_local_federation_provider_bids_on_inference():
    _svc, _fed, prov = _build_service()
    from core.providers import JobSpec
    job = JobSpec(
        job_id="j-bid-1", kind="inference", payload={"prompt": "hi", "max_tokens": 16},
        cost_ceiling_usd=0.01, latency_ceiling_ms=2_000, quality_floor=0.5,
    )
    bid = prov.bid(job)
    assert bid is not None
    assert bid.provider_id == "local-fed-test"
    assert bid.eta_ms == 10
    assert bid.expected_quality >= 0.5


def test_local_federation_provider_abstains_when_no_backend():
    _svc, _fed, prov = _build_service(backend_available=False)
    from core.providers import JobSpec
    job = JobSpec(
        job_id="j-bid-no", kind="inference", payload={"prompt": "hi"},
        cost_ceiling_usd=0.01, latency_ceiling_ms=2_000,
    )
    assert prov.bid(job) is None


def test_local_federation_provider_abstains_on_training_kind():
    _svc, _fed, prov = _build_service()
    from core.providers import JobSpec
    job = JobSpec(
        job_id="j-bid-train", kind="training", payload={"data": "x"},
        cost_ceiling_usd=1.0, latency_ceiling_ms=60_000,
    )
    assert prov.bid(job) is None


# ---------------------------------------------------------------------------
# 2. Submit -> matched -> completed
# ---------------------------------------------------------------------------


async def _wait_terminal(svc: JobsService, job_id: str, timeout_s: float = 5.0):
    import time
    terminal = {"completed", "failed", "cancelled", "timeout"}
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rec = svc.get(job_id)
        if rec is not None and rec.state in terminal:
            return rec
        await asyncio.sleep(0.01)
    return svc.get(job_id)


def test_jobs_service_e2e_completes_with_local_federation():
    svc, _fed, _prov = _build_service(payload_text="hello-world-output")

    async def _run():
        rec = await svc.submit(
            kind="inference",
            payload={"prompt": "hi", "max_tokens": 16},
            cost_ceiling_usd=0.10,
            latency_ceiling_ms=10_000,
            privacy_class="public",
            quality_floor=0.5,
            requester_identity="test-user",
        )
        # After submit returns, state should already be matched (auction
        # is sync) -- the execution task runs on the loop.
        assert rec.state in ("matched", "running", "completed")
        rec = await _wait_terminal(svc, rec.job_id)
        assert rec is not None
        assert rec.state == "completed", f"unexpected state: {rec.state} ({rec.detail})"
        # Hash matches the deterministic stub output.
        import hashlib
        expected_hash = hashlib.sha256(b"hello-world-output").hexdigest()
        assert rec.result_hash_hex == expected_hash
        assert rec.matched_provider_pubkey == "local-fed-test"
        assert rec.price_locked_usd is not None
        assert rec.execution_ms is not None and rec.execution_ms >= 0

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3. Sign path
# ---------------------------------------------------------------------------


def test_jobs_service_e2e_signs_when_wallet_present():
    from core.tokenomics import Wallet
    wallet = Wallet()
    svc, _fed, _prov = _build_service(wallet=wallet)

    async def _run():
        rec = await svc.submit(
            kind="inference",
            payload={"prompt": "sign me", "max_tokens": 8},
            cost_ceiling_usd=0.10,
            latency_ceiling_ms=10_000,
            privacy_class="public",
            quality_floor=0.5,
            requester_identity="test-user",
        )
        rec = await _wait_terminal(svc, rec.job_id)
        assert rec.state == "completed"
        assert rec.provider_signature_b64 is not None
        # The signature must verify against the wallet's pubkey on the
        # signing-message contract used by Wallet.sign (the result hash).
        assert Wallet.verify(
            wallet.public_key_pem, rec.result_hash_hex,
            rec.provider_signature_b64,
        )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 4. No backend -> failed
# ---------------------------------------------------------------------------


def test_jobs_service_e2e_no_backend_fails_cleanly():
    svc, _fed, _prov = _build_service(backend_available=False)

    async def _run():
        rec = await svc.submit(
            kind="inference",
            payload={"prompt": "x", "max_tokens": 4},
            cost_ceiling_usd=0.10,
            latency_ceiling_ms=2_000,
            privacy_class="public",
            quality_floor=0.5,
            requester_identity="test-user",
        )
        # Submit returns rec already failed (auction had no winning bid).
        assert rec.state == "failed"
        assert rec.detail == "no_provider_matched"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 5. Privacy mapping
# ---------------------------------------------------------------------------


def test_local_federation_provider_maps_privacy_to_local_only():
    fed = _StubFederation()
    prov = LocalFederationProvider(
        provider_id="p", federation_factory=lambda: fed,
    )
    auction = Auction()
    auction.register(prov)
    svc = JobsService(auction=auction)

    async def _run():
        rec = await svc.submit(
            kind="inference",
            payload={"prompt": "secret", "max_tokens": 4},
            cost_ceiling_usd=0.10,
            latency_ceiling_ms=5_000,
            privacy_class="private",
            quality_floor=0.4,
            requester_identity="test-user",
        )
        rec = await _wait_terminal(svc, rec.job_id)
        assert rec.state == "completed"
        assert fed.last_request is not None
        assert fed.last_request.privacy_mode == "LOCAL_ONLY"

    asyncio.run(_run())
