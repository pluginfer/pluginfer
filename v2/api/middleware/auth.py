"""Auth backend.

Two paths:

1. **API key** — `Authorization: Bearer pf_live_<...>`. The backend
   stores `sha256(key)` only, never the raw key. Convenient for SDK
   users.

2. **Wallet signature** — challenge / response over a server-issued
   nonce. The client signs `nonce|audience|expires_at` with their wallet
   private key (ECDSA secp256k1) and posts `pubkey_pem` + `signature_b64`
   to `POST /v1/auth/verify`. The backend verifies and mints an
   API-key-equivalent session.

Both paths land at `require_auth` which yields the authenticated
identity (api_key id or wallet pubkey) for downstream handlers.

The verifier reuses the same ECDSA chain as `core/updater.py` /
`core/gossip.py` so we get the W31 / W24 hardening for free.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

CHALLENGE_TTL_SEC = 30.0
SESSION_TTL_SEC = 24 * 3600


@dataclass
class _Challenge:
    nonce: str
    issued_at: float
    expires_at: float
    audience: str = "pluginfer-api"


@dataclass
class AuthBackend:
    """In-memory auth backend. Production swaps `_keys` for a Redis or
    SQL-backed store without changing the handler signatures."""
    _keys: Dict[str, str] = field(default_factory=dict)            # sha256(key) -> identity
    _challenges: Dict[str, _Challenge] = field(default_factory=dict)
    _sessions: Dict[str, tuple[str, float]] = field(default_factory=dict)
                                                                  # session_id -> (identity, expires)
    audience: str = "pluginfer-api"

    # ------------------------------------------------------------------
    # API key issuance / lookup
    # ------------------------------------------------------------------
    def issue_api_key(self, identity: str, *, prefix: str = "pf_live_") -> str:
        """Return a freshly-minted API key string. The raw value is
        returned to the caller and never stored — only its hash."""
        raw = prefix + secrets.token_urlsafe(32)
        self._keys[hashlib.sha256(raw.encode()).hexdigest()] = identity
        return raw

    def revoke_api_key(self, raw: str) -> bool:
        return self._keys.pop(hashlib.sha256(raw.encode()).hexdigest(), None) is not None

    def identify_api_key(self, raw: str) -> Optional[str]:
        return self._keys.get(hashlib.sha256(raw.encode()).hexdigest())

    # ------------------------------------------------------------------
    # Wallet challenge / verify
    # ------------------------------------------------------------------
    def issue_challenge(self) -> _Challenge:
        now = time.time()
        c = _Challenge(
            nonce=secrets.token_hex(32),
            issued_at=now,
            expires_at=now + CHALLENGE_TTL_SEC,
            audience=self.audience,
        )
        self._challenges[c.nonce] = c
        # Garbage-collect any stale challenges so the dict stays bounded.
        for k in list(self._challenges):
            if self._challenges[k].expires_at < now:
                self._challenges.pop(k, None)
        return c

    def verify_challenge(
        self,
        *,
        nonce: str,
        pubkey_pem: str,
        signature_b64: str,
    ) -> str:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec

        c = self._challenges.pop(nonce, None)
        if c is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="unknown_or_expired_nonce",
            )
        if c.expires_at < time.time():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="challenge_expired",
            )

        body = f"{c.nonce}|{c.audience}|{c.expires_at}".encode()
        try:
            pub = serialization.load_pem_public_key(pubkey_pem.encode())
            if not isinstance(pub, ec.EllipticCurvePublicKey):
                raise ValueError("only EC public keys accepted")
            pub.verify(
                base64.b64decode(signature_b64),
                body,
                ec.ECDSA(hashes.SHA256()),
            )
        except (InvalidSignature, ValueError, Exception):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="signature_verification_failed",
            )

        # Identity == fingerprint(pubkey_pem) — short, stable, sharable.
        identity = "wallet:" + hashlib.sha256(pubkey_pem.encode()).hexdigest()[:16]
        session_id = secrets.token_urlsafe(32)
        self._sessions[session_id] = (identity, time.time() + SESSION_TTL_SEC)
        return session_id

    def identify_session(self, session_id: str) -> Optional[str]:
        rec = self._sessions.get(session_id)
        if not rec:
            return None
        identity, expires = rec
        if expires < time.time():
            self._sessions.pop(session_id, None)
            return None
        return identity

    # ------------------------------------------------------------------
    # Unified resolver
    # ------------------------------------------------------------------
    def identify(self, *, bearer: Optional[str], wallet_session: Optional[str]) -> Optional[str]:
        if bearer:
            who = self.identify_api_key(bearer)
            if who:
                return who
        if wallet_session:
            who = self.identify_session(wallet_session)
            if who:
                return who
        return None


_bearer_scheme = HTTPBearer(auto_error=False)


def require_auth(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> str:
    """FastAPI dependency. Returns the authenticated identity string,
    raises 401 otherwise."""
    backend: AuthBackend = request.app.state.auth_backend
    bearer = creds.credentials if creds else None
    session = request.headers.get("x-pluginfer-session")
    identity = backend.identify(bearer=bearer, wallet_session=session)
    if identity is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing_or_invalid_credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return identity
