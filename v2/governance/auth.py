"""Access control for the governance gateway.

The audit flagged this as the worst immature bug: the gateway forwards
an org's PAID upstream key, yet every endpoint was open — anyone who
reached the port could spend the key and read every team's spend. This
module closes that.

Three roles, each a bearer token:

  * **client keys** — issued per team/app; required to call
    ``/v1/chat/completions`` and ``/v1/messages``. The key's fingerprint
    is what appears on receipts (never the raw key), and a key may be
    pinned to a default budget envelope so a caller can't spend from
    another team's envelope by setting a header.
  * **read key(s)** — required for the reporting/observability surface
    (report, savings, receipts, dashboard). Read access to spend data
    is sensitive; it is no longer public.
  * **admin key** — required to mint/revoke keys and set envelopes.

Enforcement is fail-closed ONLY when auth is configured. A gateway
built with no keys at all stays open (the historical dev/test default);
the moment any admin key is set, the surface locks down. Standalone
``main()`` refuses to bind a public interface without auth.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _fingerprint(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class ClientKey:
    fingerprint: str
    label: str = ""
    envelope: Optional[str] = None      # pinned default envelope
    revoked: bool = False


@dataclass
class AuthConfig:
    """Central auth state. Thread-safe. All comparisons are constant
    time. Raw keys are never stored — only their sha256 fingerprints."""

    admin_key: Optional[str] = None
    read_keys: List[str] = field(default_factory=list)
    _client_by_fp: Dict[str, ClientKey] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # -- configured? --------------------------------------------------
    @property
    def enforced(self) -> bool:
        """Auth binds only once configured — else the dev/test default
        stays open (backwards compatible)."""
        with self._lock:
            return bool(self.admin_key or self.read_keys
                        or self._client_by_fp)

    # -- checks (constant time) ---------------------------------------
    def _eq(self, a: Optional[str], b: Optional[str]) -> bool:
        if not a or not b:
            return False
        return hmac.compare_digest(a, b)

    def is_admin(self, key: Optional[str]) -> bool:
        return self._eq(key, self.admin_key)

    def is_reader(self, key: Optional[str]) -> bool:
        if self.is_admin(key):
            return True
        with self._lock:
            return any(self._eq(key, rk) for rk in self.read_keys)

    def client_for(self, key: Optional[str]) -> Optional[ClientKey]:
        if not key:
            return None
        with self._lock:
            ck = self._client_by_fp.get(_fingerprint(key))
            if ck is None or ck.revoked:
                return None
            return ck

    # -- admin ops ----------------------------------------------------
    def issue_client_key(self, *, label: str = "",
                         envelope: Optional[str] = None) -> str:
        raw = "plg-" + secrets.token_urlsafe(24)
        with self._lock:
            self._client_by_fp[_fingerprint(raw)] = ClientKey(
                fingerprint=_fingerprint(raw), label=label,
                envelope=envelope)
        return raw

    def revoke_client_key(self, raw_or_fp: str) -> bool:
        fp = raw_or_fp if len(raw_or_fp) == 16 else _fingerprint(raw_or_fp)
        with self._lock:
            ck = self._client_by_fp.get(fp)
            if ck is None:
                return False
            ck.revoked = True
            return True

    def list_client_keys(self) -> List[Dict[str, object]]:
        with self._lock:
            return [{"fingerprint": ck.fingerprint, "label": ck.label,
                     "envelope": ck.envelope, "revoked": ck.revoked}
                    for ck in self._client_by_fp.values()]


def bearer(headers) -> Optional[str]:
    """Extract a bearer token from an Authorization header (or the
    X-Api-Key header). Returns None if absent."""
    auth = headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return headers.get("x-api-key") or None
