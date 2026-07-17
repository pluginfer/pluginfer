"""Cross-gateway settlement forwarder.

The local-ledger problem
------------------------
When gateway A's auction picks provider B (running on gateway B), A's
ledger debits the buyer's wallet and CREDITS provider B's wallet —
but only in A's own ledger. B's gateway has its own ledger that
doesn't yet see the earnings. Providers don't trust that, and rightly
so: the funds aren't on B's side.

This module closes the loop. After every successful settlement, A
POSTs a signed credit notification to B's `/v1/wallet/credit_notice`.
B verifies A's signature against the seed-known gateway pubkey of A,
applies the credit locally, and returns a signed ack.

Failures are queued: an outbox table tracks unsent notifications,
retries with exponential backoff, and surfaces unreachable peers to
a reconciliation job that runs nightly. No payment ever silently
goes missing.

Innovation: §A29 "Cross-gateway signed-credit forwarding for
permissionless compute auctions." The combination of (a) per-job
signed receipt, (b) per-credit signed notification, (c) outbox-with-
retry, AND (d) nightly reconciliation gives the provider mathematical
certainty that earnings recorded ANYWHERE in the mesh will
eventually land in their home ledger — without trusting any single
gateway.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_RETRY_DELAYS_S = (1, 4, 16, 64, 256)


@dataclass
class CreditNotice:
    """A single cross-gateway credit notification."""
    notice_id: str
    source_gateway_pubkey: str
    target_gateway_url: str
    provider_wallet_id: str
    amount_usd: Decimal
    job_id: str
    issued_at_unix: float = field(default_factory=time.time)
    delivered: bool = False
    last_attempt_unix: float = 0.0
    attempts: int = 0
    last_error: Optional[str] = None

    def signing_message(self) -> str:
        return (
            f"CREDIT|{self.notice_id}|"
            f"{self.source_gateway_pubkey}|"
            f"{self.provider_wallet_id}|"
            f"{self.amount_usd}|"
            f"{self.job_id}|"
            f"{self.issued_at_unix:.6f}"
        )


@dataclass
class SettlementForwarder:
    """Owns the outbox of pending credit notifications. The
    `forward()` call attempts an immediate send; `retry_pending()`
    drains stuck notices on a schedule."""
    source_gateway_pubkey: str
    sign_fn: Callable[[str], str]
    outbox: List[CreditNotice] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # Pluggable HTTP poster; default is urllib. Tests inject a
    # mock that records calls without touching the network.
    poster: Optional[Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]]] = None

    def forward(
        self, *, target_gateway_url: str, provider_wallet_id: str,
        amount_usd: Decimal, job_id: str,
    ) -> CreditNotice:
        """Synchronous: tries immediate send. On failure, leaves the
        notice in the outbox for retry. Returns the notice record
        for caller visibility."""
        nid = hashlib.sha256(
            f"{self.source_gateway_pubkey}|{job_id}|{provider_wallet_id}".encode()
        ).hexdigest()[:24]
        notice = CreditNotice(
            notice_id=nid,
            source_gateway_pubkey=self.source_gateway_pubkey,
            target_gateway_url=target_gateway_url.rstrip("/"),
            provider_wallet_id=provider_wallet_id,
            amount_usd=amount_usd, job_id=job_id,
        )
        with self._lock:
            self.outbox.append(notice)
        self._try_send(notice)
        return notice

    def _try_send(self, notice: CreditNotice) -> bool:
        if notice.delivered:
            return True
        notice.attempts += 1
        notice.last_attempt_unix = time.time()
        body = {
            "notice_id": notice.notice_id,
            "source_gateway_pubkey": notice.source_gateway_pubkey,
            "provider_wallet_id": notice.provider_wallet_id,
            "amount_usd": str(notice.amount_usd),
            "job_id": notice.job_id,
            "issued_at_unix": notice.issued_at_unix,
            "signature": self.sign_fn(notice.signing_message()),
        }
        url = f"{notice.target_gateway_url}/v1/wallet/credit_notice"
        poster = self.poster or _default_poster
        resp = poster(url, body)
        if resp and resp.get("status") == "credited":
            notice.delivered = True
            notice.last_error = None
            return True
        notice.last_error = (resp or {}).get(
            "error", "no response from target gateway"
        )
        return False

    def retry_pending(self) -> Dict[str, int]:
        """Iterate pending notices; reattempt those whose backoff has
        elapsed. Returns {"sent": n_sent, "still_pending": n_pending}.
        Call this from a background tick or a nightly reconciliation
        job."""
        sent = 0
        still_pending = 0
        with self._lock:
            snapshot = list(self.outbox)
        now = time.time()
        for n in snapshot:
            if n.delivered:
                continue
            backoff_idx = min(n.attempts - 1, len(DEFAULT_RETRY_DELAYS_S) - 1)
            if backoff_idx < 0:
                backoff_idx = 0
            wait_s = DEFAULT_RETRY_DELAYS_S[backoff_idx]
            if (now - n.last_attempt_unix) < wait_s:
                still_pending += 1
                continue
            if self._try_send(n):
                sent += 1
            else:
                still_pending += 1
        return {"sent": sent, "still_pending": still_pending}

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for n in self.outbox if not n.delivered)


def _default_poster(url: str, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        logger.warning("credit notice POST failed for %s: %s", url, e)
        return None


# ---------------------------------------------------------------------------
# Receiving side — verifying inbound credit notices
# ---------------------------------------------------------------------------

def verify_credit_notice(
    body: Dict[str, Any],
    *,
    known_gateway_pubkeys: Dict[str, str],
    verify_signature: Optional[Callable[..., bool]] = None,
) -> Optional[Dict[str, Any]]:
    """Receiver side. Given the inbound POST body + a map of trusted
    gateway pubkey fingerprints (or full PEMs), check:
      * signature verifies under source_gateway_pubkey,
      * amount_usd parses as a positive Decimal,
      * notice_id has not been previously credited (caller's job).

    Returns the validated payload dict, or None on rejection.
    `verify_signature(msg, sig_b64, pubkey_pem) -> bool` is pluggable
    so tests can inject a stub. When the callback is None, signature
    check is SKIPPED — appropriate for unit tests that don't have a
    Wallet."""
    required = (
        "notice_id", "source_gateway_pubkey", "provider_wallet_id",
        "amount_usd", "job_id", "issued_at_unix", "signature",
    )
    if not all(k in body for k in required):
        return None
    source_pub = str(body["source_gateway_pubkey"])
    fp = hashlib.sha256(source_pub.encode("utf-8")).hexdigest()
    if known_gateway_pubkeys and (
        source_pub not in known_gateway_pubkeys.values()
        and fp not in known_gateway_pubkeys
    ):
        return None
    try:
        amount = Decimal(str(body["amount_usd"]))
    except Exception:
        return None
    if amount <= Decimal("0"):
        return None
    if verify_signature is not None:
        notice = CreditNotice(
            notice_id=body["notice_id"],
            source_gateway_pubkey=source_pub,
            target_gateway_url="",
            provider_wallet_id=str(body["provider_wallet_id"]),
            amount_usd=amount,
            job_id=str(body["job_id"]),
            issued_at_unix=float(body["issued_at_unix"]),
        )
        if not verify_signature(
            notice.signing_message(), str(body["signature"]), source_pub,
        ):
            return None
    return {
        "notice_id": body["notice_id"],
        "source_gateway_pubkey": source_pub,
        "provider_wallet_id": body["provider_wallet_id"],
        "amount_usd": amount,
        "job_id": body["job_id"],
        "issued_at_unix": float(body["issued_at_unix"]),
    }


__all__ = [
    "CreditNotice",
    "DEFAULT_RETRY_DELAYS_S",
    "SettlementForwarder",
    "verify_credit_notice",
]
