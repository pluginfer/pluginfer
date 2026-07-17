"""RemoteProvider + JobServer: jobs run on remote peer nodes.

The two pieces that close the "in-process auction" loop into a real
peer-to-peer compute marketplace:

  * `RemoteProvider` -- a `Provider` subclass that wraps a
    `MeshConnector` channel to a peer. Its `execute(job, bid)`
    serialises the job over the channel, waits for the peer's signed
    result, and returns it. From the auction's perspective this is
    indistinguishable from a local `MeshGPUProvider`; the broker
    doesn't know or care.
  * `JobServer` -- the listener that runs on the OTHER end. Wraps any
    inner `Provider` (typically `MeshGPUProvider`, but could be a
    test fake or a special-purpose provider). Subscribes to inbound
    JOB_REQUEST envelopes, dispatches to the inner provider, signs the
    result hash with the operator wallet, ships JOB_RESULT back.

Both ends speak the same JSON-line wire format on top of MeshConnector
bytes:

  Requester -> Provider (JOB_REQUEST):
    {"op":"JOB_REQUEST","request_id":"<hex>","job":{job-spec dict},
     "bid":{bid dict}}

  Provider -> Requester (JOB_RESULT):
    {"op":"JOB_RESULT","request_id":"<hex>","status":"executed",
     "result_b64":"...","result_hash":"...","provider_sig":"..."}

  Provider -> Requester (JOB_ERROR):
    {"op":"JOB_ERROR","request_id":"<hex>","reason":"..."}

`request_id` is a fresh uuid4-hex per job so a single channel can carry
multiple in-flight jobs concurrently (the auction doesn't strictly
need this for v1, but it costs nothing and saves a refactor later).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import secrets
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

from .mesh_connector import MeshChannel, MeshConnector
from .providers import Bid, JobSpec, Provider

logger = logging.getLogger(__name__)

JOB_REQUEST_TIMEOUT_S: float = 30.0


# ---------------------------------------------------------------------------
# RemoteProvider (lives on the requester / broker side)
# ---------------------------------------------------------------------------


@dataclass
class RemoteProvider(Provider):
    """A peer that has advertised they will run jobs of this kind.

    The auction sees this exactly like a local provider: same `bid()`,
    same `execute()`. Internally `execute()` round-trips over the mesh.
    """
    provider_id: str
    connector: MeshConnector
    # The bid the peer pre-advertised when it registered with our
    # broker. In production this comes from a discovery/auction
    # protocol; for v1 we let the caller hand us the bid directly.
    advertised_bid: Bid
    privacy_grade: str = "public"
    kind: str = "compute"
    # Outstanding requests: request_id -> Future[dict]
    _pending: Dict[str, asyncio.Future] = field(default_factory=dict, repr=False)

    def bid(self, job: JobSpec) -> Optional[Bid]:
        # Delegate constraint checks to the advertised bid; if it
        # violates the job, return None to abstain.
        violation = self.advertised_bid.violates(job)
        if violation is not None:
            return None
        return self.advertised_bid

    def execute(self, job: JobSpec, bid: Bid) -> Dict[str, Any]:
        """Sync wrapper over the async mesh round-trip. The Auction's
        existing executor (run_in_executor) already calls this from a
        thread, so we use asyncio.run_coroutine_threadsafe to dispatch
        onto the connector's loop."""
        loop = self._loop_for_connector()
        coro = self._execute_async(job, bid)
        if loop is not None and loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
            return fut.result(timeout=JOB_REQUEST_TIMEOUT_S + 5.0)
        # If we're on a fresh thread with no active loop, run the
        # whole coroutine inline (test convenience).
        return asyncio.run(coro)

    async def _execute_async(self, job: JobSpec, bid: Bid) -> Dict[str, Any]:
        ch = await self.connector.connect(self.provider_id)
        request_id = secrets.token_hex(16)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = fut

        # Install/extend on_message to demux JOB_RESULT / JOB_ERROR
        # frames by request_id. We must be careful not to clobber any
        # existing on_message the caller may have set.
        prior = ch.on_message
        def _route(payload: bytes) -> None:
            if prior is not None:
                try:
                    prior(payload)
                except Exception as e:                              # pragma: no cover
                    logger.warning("prior on_message raised: %s", e)
            try:
                msg = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return
            if not isinstance(msg, dict):
                return
            rid = msg.get("request_id")
            if not isinstance(rid, str):
                return
            target_fut = self._pending.get(rid)
            if target_fut is None or target_fut.done():
                return
            target_fut.set_result(msg)
        ch.on_message = _route

        # Send the request.
        body = {
            "op": "JOB_REQUEST",
            "request_id": request_id,
            "job": _spec_to_dict(job),
            "bid": _bid_to_dict(bid),
        }
        await ch.send(json.dumps(body).encode("utf-8"))

        try:
            reply = await asyncio.wait_for(fut, timeout=JOB_REQUEST_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            return {"status": "timeout", "reason": "remote_no_reply"}
        finally:
            # Restore prior on_message if our caller had one.
            ch.on_message = prior
            self._pending.pop(request_id, None)

        if reply.get("op") == "JOB_ERROR":
            return {"status": "error", "reason": reply.get("reason", "?")}
        # JOB_RESULT -- pass through the provider's payload as-is.
        return {
            "status": reply.get("status", "executed"),
            "result_bytes_b64": reply.get("result_b64"),
            "result_hash": reply.get("result_hash"),
            "provider_sig": reply.get("provider_sig"),
        }

    def _loop_for_connector(self) -> Optional[asyncio.AbstractEventLoop]:
        # MeshConnector is bound to whatever loop started its punch
        # transport. In production the broker runs on the same loop;
        # for tests we discover it via the punch transport.
        try:
            tr = self.connector.punch.transport
            if tr is None:
                return None
            return tr._loop  # type: ignore[attr-defined]  # private but stable in CPython
        except Exception:
            return None


def _spec_to_dict(spec: JobSpec) -> dict:
    """Make a JobSpec round-trippable over the wire."""
    out = {
        "job_id": spec.job_id,
        "kind": spec.kind,
        "payload": spec.payload,
        "cost_ceiling_usd": spec.cost_ceiling_usd,
        "latency_ceiling_ms": spec.latency_ceiling_ms,
        "privacy_class": spec.privacy_class,
        "quality_floor": spec.quality_floor,
        "submitted_at": spec.submitted_at,
    }
    return out


def _spec_from_dict(d: dict) -> JobSpec:
    return JobSpec(
        job_id=str(d["job_id"]),
        kind=str(d["kind"]),
        payload=d.get("payload") or {},
        cost_ceiling_usd=float(d.get("cost_ceiling_usd", 0.0)),
        latency_ceiling_ms=int(d.get("latency_ceiling_ms", 30000)),
        privacy_class=str(d.get("privacy_class", "public")),
        quality_floor=float(d.get("quality_floor", 0.7)),
        submitted_at=float(d.get("submitted_at", time.time())),
    )


def _bid_to_dict(bid: Bid) -> dict:
    return {
        "provider_id": bid.provider_id,
        "price_usd": bid.price_usd,
        "eta_ms": bid.eta_ms,
        "expected_quality": bid.expected_quality,
        "privacy_grade": bid.privacy_grade,
        "evidence": bid.evidence,
    }


def _bid_from_dict(d: dict) -> Bid:
    return Bid(
        provider_id=str(d["provider_id"]),
        price_usd=float(d["price_usd"]),
        eta_ms=int(d["eta_ms"]),
        expected_quality=float(d["expected_quality"]),
        privacy_grade=str(d.get("privacy_grade", "public")),
        evidence=d.get("evidence") or {},
    )


# ---------------------------------------------------------------------------
# JobServer (lives on the provider / peer side)
# ---------------------------------------------------------------------------


@dataclass
class JobServer:
    """Listens for JOB_REQUEST envelopes on every channel of a
    MeshConnector and dispatches each one to `inner_provider.execute`.

    `inner_provider` can be a real MeshGPUProvider, a wrapped cloud
    LLM provider, or a test fake. Any callable that takes
    (JobSpec, Bid) -> dict works.
    """
    connector: MeshConnector
    inner_provider: Provider
    on_request: Optional[Callable[[JobSpec, Bid], None]] = None

    def attach(self) -> None:
        """Install the request handler on the connector. Calling this
        is sufficient -- subsequent inbound channels (from peers
        INTRODUCE'ing to us) will inherit the dispatcher."""
        # Patch the connector's _on_punch_invite + RELAY_DELIVER paths
        # so every newly-created channel gets our on_message wired.
        self.connector._mesh_data_dispatch = self._handle_request   # type: ignore[attr-defined]
        # Existing channels (if any) get retro-fitted.
        for ch in self.connector.channels.values():
            self._wire_channel(ch)
        # Wrap the connector's hooks so future channels get wired.
        original_make_direct = self.connector._make_direct_channel
        original_make_relay = self.connector._make_relay_channel

        def _patched_direct(peer_pubkey, peer_addr):
            ch = original_make_direct(peer_pubkey, peer_addr)
            self._wire_channel(ch)
            return ch

        def _patched_relay(peer_pubkey, session):
            ch = original_make_relay(peer_pubkey, session)
            self._wire_channel(ch)
            return ch

        self.connector._make_direct_channel = _patched_direct  # type: ignore[method-assign]
        self.connector._make_relay_channel = _patched_relay    # type: ignore[method-assign]

    # ------------------------------------------------------------------

    def _wire_channel(self, ch: MeshChannel) -> None:
        prior = ch.on_message
        def _on(payload: bytes) -> None:
            if prior is not None:
                try:
                    prior(payload)
                except Exception as e:                          # pragma: no cover
                    logger.warning("prior on_message raised: %s", e)
            asyncio.create_task(self._handle_request(ch, payload))
        ch.on_message = _on

    async def _handle_request(self, ch: MeshChannel, payload: bytes) -> None:
        try:
            msg = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(msg, dict):
            return
        if msg.get("op") != "JOB_REQUEST":
            return
        try:
            request_id = str(msg["request_id"])
            spec = _spec_from_dict(msg["job"])
            bid = _bid_from_dict(msg["bid"])
        except (KeyError, ValueError, TypeError) as e:
            await ch.send(json.dumps({
                "op": "JOB_ERROR",
                "request_id": str(msg.get("request_id", "?")),
                "reason": f"bad_request: {e}",
            }).encode("utf-8"))
            return

        if self.on_request is not None:
            try:
                self.on_request(spec, bid)
            except Exception as e:                              # pragma: no cover
                logger.warning("on_request hook raised: %s", e)

        # Run the inner provider. Note: inner_provider.execute is sync;
        # we run it in a thread so the asyncio loop stays responsive.
        loop = asyncio.get_running_loop()
        try:
            out = await loop.run_in_executor(
                None, self.inner_provider.execute, spec, bid,
            )
        except Exception as e:
            await ch.send(json.dumps({
                "op": "JOB_ERROR",
                "request_id": request_id,
                "reason": f"{type(e).__name__}: {e}",
            }).encode("utf-8"))
            return

        # Forward the result back. The provider's execute() output
        # is already in the right shape (status, result_bytes_b64,
        # result_hash, provider_sig); we just wrap it.
        if not isinstance(out, dict):
            await ch.send(json.dumps({
                "op": "JOB_ERROR",
                "request_id": request_id,
                "reason": "inner_provider_returned_non_dict",
            }).encode("utf-8"))
            return

        reply = {
            "op": "JOB_RESULT",
            "request_id": request_id,
            "status": out.get("status", "executed"),
            "result_b64": out.get("result_bytes_b64"),
            "result_hash": out.get("result_hash"),
            "provider_sig": out.get("provider_sig"),
        }
        await ch.send(json.dumps(reply).encode("utf-8"))
