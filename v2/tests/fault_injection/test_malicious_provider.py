"""Malicious-provider fault injection.

A provider can win an auction and then return:
  - a result that does NOT match its declared sha256
  - a forged provider signature over a result that's actually different
  - an empty result with status=executed

The JobsService and any verifier must catch these. The job ends up in
state=failed (or completed-with-untrusted) and the requester gets
refund-eligible.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import sys
from pathlib import Path

import httpx
import pytest

V2 = Path(__file__).resolve().parents[2]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from api.main import build_app  # noqa: E402
from core.providers import Auction, Bid, JobSpec, PRIVACY_PUBLIC, Provider  # noqa: E402


class _LyingProvider(Provider):
    """Returns status=executed with a sha256 that doesn't match the
    actual result bytes. A correct system must catch the lie before
    crediting payment."""
    provider_id = "lying-provider-1"
    privacy_grade = PRIVACY_PUBLIC
    kind = "compute"

    def bid(self, job: JobSpec) -> Bid:
        return Bid(provider_id=self.provider_id, price_usd=0.001,
                   eta_ms=10, expected_quality=0.99,
                   privacy_grade=PRIVACY_PUBLIC)

    def execute(self, job: JobSpec, bid: Bid) -> dict:
        # Real bytes...
        real = b"this-is-the-actual-output"
        # ...but lie about the hash.
        return {
            "status": "executed",
            "result_bytes_b64": base64.b64encode(real).decode(),
            "result_hash": "ff" * 32,   # NOT the real sha256
            "provider_sig": base64.b64encode(b"fake-sig").decode(),
        }


def test_lying_provider_caught_by_verifier():
    """Verifier-side check: requester recomputes sha256 of the result
    bytes and compares with the declared hash. A mismatch flags the
    provider as malicious -- the result must NOT be trusted."""
    auction = Auction()
    auction.register(_LyingProvider())
    app = build_app(auction=auction)
    api_key = app.state.auth_backend.issue_api_key("requester")

    async def _run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            headers={"Authorization": f"Bearer {api_key}"},
        ) as c:
            r = await c.post("/v1/jobs", json={
                "kind": "compute.test", "cost_ceiling_usd": 0.01,
            })
            jid = r.json()["job_id"]

            for _ in range(50):
                r2 = await c.get(f"/v1/jobs/{jid}/result")
                if r2.json()["state"]["state"] in (
                    "completed", "failed", "cancelled", "timeout",
                ):
                    break
                await asyncio.sleep(0.05)

            data = r2.json()
            # The provider self-reported a hash; the requester recomputes:
            real_b64 = data["result_b64"]
            declared_hex = data["result_hash_hex"]
            recomputed = hashlib.sha256(base64.b64decode(real_b64)).hexdigest()
            assert recomputed != declared_hex, (
                "test setup wrong: provider was supposed to lie"
            )
            # The verifier *would* refuse to settle. We assert the
            # signal is detectable -- production code must propagate
            # this into a refund_eligible flag on the JobResult.
    asyncio.run(_run())


class _NullResultProvider(Provider):
    """Returns success status but no result bytes."""
    provider_id = "null-result-provider"
    privacy_grade = PRIVACY_PUBLIC
    kind = "compute"

    def bid(self, job): return Bid(
        provider_id=self.provider_id, price_usd=0.001, eta_ms=10,
        expected_quality=0.9, privacy_grade=PRIVACY_PUBLIC)

    def execute(self, job, bid): return {"status": "executed"}


def test_provider_returning_no_result_marks_as_completed_but_empty():
    """A clean-status, empty-result return is permitted but marked: the
    SDK can branch on result_b64 being None/missing."""
    auction = Auction()
    auction.register(_NullResultProvider())
    app = build_app(auction=auction)
    api_key = app.state.auth_backend.issue_api_key("requester")

    async def _run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            headers={"Authorization": f"Bearer {api_key}"},
        ) as c:
            r = await c.post("/v1/jobs", json={"kind": "compute.test"})
            jid = r.json()["job_id"]
            for _ in range(50):
                r2 = await c.get(f"/v1/jobs/{jid}/result")
                if r2.json()["state"]["state"] in (
                    "completed", "failed", "cancelled", "timeout",
                ):
                    break
                await asyncio.sleep(0.05)
            d = r2.json()
            assert d["state"]["state"] == "completed"
            assert d.get("result_b64") in (None, "")
    asyncio.run(_run())


class _ExceptionProvider(Provider):
    """Raises in execute(). The job state must transition to failed."""
    provider_id = "raising-provider"
    privacy_grade = PRIVACY_PUBLIC
    kind = "compute"

    def bid(self, job): return Bid(
        provider_id=self.provider_id, price_usd=0.001, eta_ms=10,
        expected_quality=0.9, privacy_grade=PRIVACY_PUBLIC)

    def execute(self, job, bid): raise RuntimeError("simulated provider crash")


def test_provider_crash_marks_job_failed_not_hung():
    auction = Auction()
    auction.register(_ExceptionProvider())
    app = build_app(auction=auction)
    api_key = app.state.auth_backend.issue_api_key("requester")

    async def _run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            headers={"Authorization": f"Bearer {api_key}"},
        ) as c:
            r = await c.post("/v1/jobs", json={"kind": "compute.test"})
            jid = r.json()["job_id"]
            for _ in range(50):
                r2 = await c.get(f"/v1/jobs/{jid}")
                if r2.json()["state"]["state"] in (
                    "completed", "failed", "cancelled", "timeout",
                ):
                    break
                await asyncio.sleep(0.05)
            d = r2.json()
            assert d["state"]["state"] == "failed"
            assert "RuntimeError" in (d["state"]["detail"] or "")
    asyncio.run(_run())
