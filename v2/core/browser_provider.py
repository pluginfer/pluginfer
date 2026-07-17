"""HttpBrowserProvider — the server-side mirror of the in-tab provider.

The standalone binary mesh has a `RemoteProvider` that talks to a peer
over `MeshConnector` (a TCP/QUIC channel). Browsers can't open arbitrary
sockets, so we add a second `Provider` shape that backs onto the
gateway's HTTP poll protocol:

  1. browser tab POSTs ``/v1/providers/register`` with its bid template
     (pubkey, hardware class, price floor, ETA, privacy grade).
  2. server creates an ``HttpBrowserProvider`` with that template and
     adds it to the auction. Heartbeat freshness gates `bid()`.
  3. when the auction picks this provider, `execute()` blocks the
     server thread waiting on a ``threading.Event``. The job_id is
     pushed onto a per-provider FIFO of open pickups.
  4. browser tab polls ``/v1/providers/open_jobs`` and sees the
     queued job, executes it client-side, then POSTs
     ``/v1/providers/deliver`` with the signed result.
  5. server applies ``deliver()`` which sets the event and unblocks
     ``execute()``, which returns the result to ``JobsService``.

The whole protocol is HTTP/1.1 long-poll (no WebSocket, no
WebTransport) so it works from any browser tab on any network without
infrastructure assumptions. A v2 will use Server-Sent Events for the
pickup notifications to drop polling latency.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .providers import (
    PRIVACY_PUBLIC,
    Bid,
    JobSpec,
    Provider,
)


# A registered tab is considered "fresh" for this many seconds since
# its last heartbeat. Tabs that disappear (user closes them) stop
# bidding — the auction simply skips them.
HEARTBEAT_TTL_SEC = 90.0

# Maximum wall-clock the server-side execute() will block waiting for
# the browser tab to deliver. After this, the job is timed out and the
# auction layer can release escrow.
DEFAULT_DELIVER_TIMEOUT_SEC = 30.0


@dataclass
class _PendingJob:
    job: JobSpec
    bid: Bid
    event: threading.Event
    result_holder: Dict[str, Any]
    issued_at_unix: float


@dataclass
class HttpBrowserProvider(Provider):
    """A remote provider whose body executes inside a browser tab.

    The auction sees a normal `Provider`; the routing layer is hidden.
    """
    provider_id: str
    pubkey_pem: str
    hardware_class: str = "browser-webgpu"
    base_price_per_1k_tok_usd: float = 0.0001
    base_eta_ms: int = 1500
    base_quality: float = 0.7
    privacy_grade: str = PRIVACY_PUBLIC
    enabled: bool = True
    last_seen_unix: float = field(default_factory=time.time)
    # G6: tier-derived hard cap on the cost_ceiling of jobs we bid on.
    # 0 (default) means uncapped — production callers stamp the cap
    # at registration via core.sybil_guard.resolve_tier. The bid()
    # path treats <=0 as "no cap" so test fixtures continue to work
    # without setting it.
    max_job_cost_usd: float = 0.0
    tier: str = "untrusted"

    _pending: Dict[str, _PendingJob] = field(default_factory=dict, repr=False)
    _open_pickups: List[str] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------
    # Provider interface
    # ------------------------------------------------------------------
    def bid(self, job: JobSpec) -> Optional[Bid]:
        if not self.enabled:
            return None
        if time.time() - self.last_seen_unix > HEARTBEAT_TTL_SEC:
            return None
        # G6: refuse to bid on jobs above the tier's cost cap. An
        # untrusted browser tab bidding $0.0001 on a $50 job is the
        # exact attack we want to bound: even if the tab wins and
        # then refuses to deliver, the buyer's escrow is unaffected
        # because the bid never registered.
        if (self.max_job_cost_usd > 0
                and job.cost_ceiling_usd > self.max_job_cost_usd):
            return None
        approx_tokens = float(job.payload.get("max_tokens", 200))
        price = self.base_price_per_1k_tok_usd * (approx_tokens / 1000.0)
        return Bid(
            provider_id=self.provider_id,
            price_usd=price,
            eta_ms=self.base_eta_ms,
            expected_quality=self.base_quality,
            privacy_grade=self.privacy_grade,
            evidence={
                "hardware_class": self.hardware_class,
                "remote": "browser-http",
            },
        )

    def execute(self, job: JobSpec, bid: Bid) -> Dict[str, Any]:
        evt = threading.Event()
        holder: Dict[str, Any] = {}
        pending = _PendingJob(
            job=job, bid=bid, event=evt, result_holder=holder,
            issued_at_unix=time.time(),
        )
        with self._lock:
            self._pending[job.job_id] = pending
            self._open_pickups.append(job.job_id)

        # Block this thread until the browser delivers (or we time out).
        # Cap by job's own latency ceiling — the auction layer would
        # consider this a SLA violation past that point anyway.
        timeout_s = min(
            DEFAULT_DELIVER_TIMEOUT_SEC,
            max(1.0, job.latency_ceiling_ms / 1000.0 + 5.0),
        )
        delivered = evt.wait(timeout=timeout_s)

        with self._lock:
            self._pending.pop(job.job_id, None)
            if job.job_id in self._open_pickups:
                self._open_pickups.remove(job.job_id)

        if not delivered:
            return {
                "status": "timeout",
                "provider_id": self.provider_id,
                "job_id": job.job_id,
                "deadline_ms": job.latency_ceiling_ms,
                "refund_eligible": True,
            }

        return holder.get("result") or {
            "status": "error",
            "code": "empty_delivery",
            "provider_id": self.provider_id,
            "job_id": job.job_id,
            "refund_eligible": True,
        }

    # ------------------------------------------------------------------
    # Gateway-side hooks
    # ------------------------------------------------------------------
    def heartbeat(self) -> None:
        self.last_seen_unix = time.time()

    def open_pickups(self, max_n: int = 8) -> List[Dict[str, Any]]:
        """Return jobs waiting for this provider to deliver. Each entry
        is a JSON-able dict — exactly the shape browser tabs need."""
        with self._lock:
            ids = list(self._open_pickups)[:max_n]
            payloads = []
            for jid in ids:
                p = self._pending.get(jid)
                if p is None:
                    continue
                payloads.append({
                    "job_id": p.job.job_id,
                    "kind": p.job.kind,
                    "payload": p.job.payload,
                    "cost_ceiling_usd": p.job.cost_ceiling_usd,
                    "latency_ceiling_ms": p.job.latency_ceiling_ms,
                    "privacy_class": p.job.privacy_class,
                    "price_locked_usd": p.bid.price_usd,
                    "issued_at_unix": p.issued_at_unix,
                })
        return payloads

    def deliver(self, job_id: str, result: Dict[str, Any]) -> bool:
        """Apply a result delivered by the browser tab. Returns True if
        the job was waiting for us; False if it had already timed out
        (so the browser knows not to bother)."""
        with self._lock:
            entry = self._pending.get(job_id)
            if entry is None:
                return False
            entry.result_holder["result"] = result
            entry.event.set()
        self.heartbeat()
        return True


@dataclass
class HttpBrowserRegistry:
    """Tracks registered browser providers keyed by pubkey fingerprint.
    The router uses this to register / look up / heartbeat them."""
    by_id: Dict[str, HttpBrowserProvider] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def register(
        self, *,
        provider_id: str,
        pubkey_pem: str,
        hardware_class: str,
        base_price_per_1k_tok_usd: float,
        base_eta_ms: int,
        base_quality: float,
        privacy_grade: str,
    ) -> HttpBrowserProvider:
        with self._lock:
            existing = self.by_id.get(provider_id)
            if existing is not None:
                # Re-register (browser tab refresh) — refresh template +
                # heartbeat. Keep any pending jobs intact.
                existing.pubkey_pem = pubkey_pem
                existing.hardware_class = hardware_class
                existing.base_price_per_1k_tok_usd = base_price_per_1k_tok_usd
                existing.base_eta_ms = base_eta_ms
                existing.base_quality = base_quality
                existing.privacy_grade = privacy_grade
                existing.enabled = True
                existing.heartbeat()
                return existing
            p = HttpBrowserProvider(
                provider_id=provider_id,
                pubkey_pem=pubkey_pem,
                hardware_class=hardware_class,
                base_price_per_1k_tok_usd=base_price_per_1k_tok_usd,
                base_eta_ms=base_eta_ms,
                base_quality=base_quality,
                privacy_grade=privacy_grade,
            )
            self.by_id[provider_id] = p
            return p

    def get(self, provider_id: str) -> Optional[HttpBrowserProvider]:
        return self.by_id.get(provider_id)

    def all(self) -> List[HttpBrowserProvider]:
        return list(self.by_id.values())

    def reap_stale(self) -> int:
        """Remove providers whose heartbeats have aged out. Returns the
        number reaped."""
        now = time.time()
        reaped = 0
        with self._lock:
            for pid in list(self.by_id):
                if now - self.by_id[pid].last_seen_unix > HEARTBEAT_TTL_SEC * 4:
                    del self.by_id[pid]
                    reaped += 1
        return reaped
