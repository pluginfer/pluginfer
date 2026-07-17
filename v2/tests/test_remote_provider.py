"""End-to-end test: a job submitted to the auction lands on a REMOTE
peer via MeshConnector, the peer signs the result, the requester
verifies the signature.

This is the key smoke test for "the auction can use peer GPUs over the
internet". Two MeshConnectors over loopback UDP + a real seed UDP
server stand in for the two-ISP case; the wire format and code paths
are identical to what the production seed-on-VPS deployment runs.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import pytest

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from core.mesh_connector import MeshConnector  # noqa: E402
from core.providers import (  # noqa: E402
    Auction,
    Bid,
    JobSpec,
    PRIVACY_PUBLIC,
    Provider,
)
from core.remote_provider import JobServer, RemoteProvider  # noqa: E402
from core.tokenomics import Wallet  # noqa: E402
from infrastructure.seed_node.punch_server import (  # noqa: E402
    PunchRelayState,
    _PunchProtocol,
)


async def _start_seed(loop) -> Tuple:
    state = PunchRelayState()
    transport, proto = await loop.create_datagram_endpoint(
        lambda: _PunchProtocol(state),
        local_addr=("127.0.0.1", 0),
    )
    proto.state = state
    return transport, proto, transport.get_extra_info("sockname")


# ---------------------------------------------------------------------------
# A minimal "real" inner provider that signs its result with a wallet
# ---------------------------------------------------------------------------


class _SigningProvider(Provider):
    """The compute backend the JobServer dispatches to. NOT a mock --
    real bytes, real sha256, real ECDSA signature."""

    privacy_grade = PRIVACY_PUBLIC
    kind = "compute"

    def __init__(self, wallet: Wallet) -> None:
        self.wallet = wallet
        self.provider_id = wallet.address

    def bid(self, job: JobSpec) -> Bid:
        return Bid(
            provider_id=self.provider_id, price_usd=0.001, eta_ms=10,
            expected_quality=0.99, privacy_grade=PRIVACY_PUBLIC,
        )

    def execute(self, job: JobSpec, bid: Bid) -> dict:
        out = json.dumps({"kind": job.kind, "echo": job.payload}).encode("utf-8")
        digest = hashlib.sha256(out).hexdigest()
        return {
            "status": "executed",
            "result_bytes_b64": base64.b64encode(out).decode(),
            "result_hash": digest,
            "provider_sig": self.wallet.sign(digest),
        }


# ---------------------------------------------------------------------------
# the test
# ---------------------------------------------------------------------------


def test_job_runs_on_remote_peer_over_mesh():
    """Two nodes over loopback. Node A is the requester. Node B runs
    a JobServer wrapping a SigningProvider. A's auction uses a
    RemoteProvider pointing at B; auction.run picks B; A.execute()
    round-trips over the mesh; result + ECDSA sig come back; A
    verifies the sig and the hash matches."""

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        seed_t, seed_p, seed_addr = await _start_seed(loop)
        wa, wb = Wallet(), Wallet()

        # --- requester side (A) ---
        a_conn = await MeshConnector.start(
            seed_addr=seed_addr, wallet=wa, bind_host="127.0.0.1",
        )
        # --- provider side (B) ---
        b_conn = await MeshConnector.start(
            seed_addr=seed_addr, wallet=wb, bind_host="127.0.0.1",
        )
        # Wait for both peers to register with the seed.
        for _ in range(40):
            if (wa.public_key_pem in seed_p.state.regs
                    and wb.public_key_pem in seed_p.state.regs):
                break
            await asyncio.sleep(0.02)

        signing = _SigningProvider(wb)
        job_server = JobServer(connector=b_conn, inner_provider=signing)
        job_server.attach()

        # The advertised bid that A's auction sees for B. In production
        # this comes from a discovery protocol; here we inject directly.
        advertised = signing.bid(JobSpec(
            job_id="dummy", kind="compute.test", payload={},
            cost_ceiling_usd=1.0, latency_ceiling_ms=10_000,
        ))
        remote = RemoteProvider(
            provider_id=wb.public_key_pem,        # we use pubkey as the connect target
            connector=a_conn,
            advertised_bid=Bid(
                provider_id=wb.public_key_pem,    # rebind so the bid carries the same id
                price_usd=advertised.price_usd,
                eta_ms=advertised.eta_ms,
                expected_quality=advertised.expected_quality,
                privacy_grade=advertised.privacy_grade,
            ),
        )

        auction = Auction()
        auction.register(remote)

        spec = JobSpec(
            job_id="job-1",
            kind="compute.echo",
            payload={"prompt": "hello over mesh"},
            cost_ceiling_usd=0.01,
            latency_ceiling_ms=10_000,
            quality_floor=0.7,
            privacy_class="public",
        )

        # 1. Auction picks the (only) RemoteProvider.
        result = auction.run(spec)
        assert result.is_won(), (
            f"auction had no winner -- bids={result.bids}, "
            f"rejected={result.rejected}"
        )
        assert result.winner.provider_id == wb.public_key_pem

        # 2. Execute via the REMOTE path. Run remote.execute on a
        # separate thread because it round-trips on the loop we're
        # currently driving. asyncio.to_thread does this cleanly.
        out = await asyncio.to_thread(remote.execute, spec, result.winner)

        try:
            assert out["status"] == "executed", f"remote returned: {out}"
            assert out["result_bytes_b64"]
            # Decode and reverify the hash + signature locally.
            decoded = base64.b64decode(out["result_bytes_b64"])
            recomputed = hashlib.sha256(decoded).hexdigest()
            assert recomputed == out["result_hash"], "result hash mismatch"
            assert Wallet.verify(
                wb.public_key_pem, recomputed, out["provider_sig"],
            ), "provider signature failed verify"
        finally:
            a_conn.close()
            b_conn.close()
            seed_t.close()
            await asyncio.sleep(0)

    asyncio.run(_run())


def test_job_error_propagates_when_remote_provider_raises():
    """If the inner provider on the remote side raises, the requester
    sees `status=error` with the type+message in `reason`."""

    class _BoomProvider(Provider):
        privacy_grade = PRIVACY_PUBLIC
        kind = "compute"
        provider_id = "boom"
        def bid(self, job): return Bid(
            provider_id="boom", price_usd=0.001, eta_ms=10,
            expected_quality=0.99, privacy_grade=PRIVACY_PUBLIC,
        )
        def execute(self, job, bid): raise RuntimeError("simulated remote failure")

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        seed_t, seed_p, seed_addr = await _start_seed(loop)
        wa, wb = Wallet(), Wallet()
        a_conn = await MeshConnector.start(
            seed_addr=seed_addr, wallet=wa, bind_host="127.0.0.1",
        )
        b_conn = await MeshConnector.start(
            seed_addr=seed_addr, wallet=wb, bind_host="127.0.0.1",
        )
        for _ in range(40):
            if (wa.public_key_pem in seed_p.state.regs
                    and wb.public_key_pem in seed_p.state.regs):
                break
            await asyncio.sleep(0.02)

        srv = JobServer(connector=b_conn, inner_provider=_BoomProvider())
        srv.attach()

        remote = RemoteProvider(
            provider_id=wb.public_key_pem,
            connector=a_conn,
            advertised_bid=Bid(
                provider_id=wb.public_key_pem,
                price_usd=0.001, eta_ms=10, expected_quality=0.9,
                privacy_grade=PRIVACY_PUBLIC,
            ),
        )

        spec = JobSpec(
            job_id="job-boom", kind="compute.test", payload={},
            cost_ceiling_usd=0.01, latency_ceiling_ms=5_000,
        )

        try:
            out = await asyncio.to_thread(remote.execute, spec,
                                          remote.advertised_bid)
            assert out["status"] == "error"
            assert "RuntimeError" in (out.get("reason") or "")
            assert "simulated remote failure" in (out.get("reason") or "")
        finally:
            a_conn.close()
            b_conn.close()
            seed_t.close()
            await asyncio.sleep(0)

    asyncio.run(_run())
