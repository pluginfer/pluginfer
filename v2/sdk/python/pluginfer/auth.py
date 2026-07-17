"""Auth API surface — wallet challenge-response login."""

from __future__ import annotations

import base64
from typing import Any, Optional

from ._http import HttpSession


class AuthAPI:
    def __init__(self, session: HttpSession) -> None:
        self._s = session

    def issue_challenge(self) -> dict:
        return self._s.post("/v1/auth/challenge", json={})

    def verify(self, *, nonce: str, pubkey_pem: str, signature_b64: str) -> dict:
        return self._s.post("/v1/auth/verify", json={
            "nonce": nonce,
            "pubkey_pem": pubkey_pem,
            "signature_b64": signature_b64,
        })

    def login_with_wallet(self, wallet: Any) -> str:
        """Run the full challenge-response with a Pluginfer Wallet.

        `wallet` must expose `.public_key_pem` and `.sign(bytes) -> bytes`.
        On success the returned session id is also stashed in the
        underlying HTTP session so subsequent requests are authed.
        """
        ch = self.issue_challenge()
        nonce = ch["nonce"]
        body = f"{nonce}|{ch['audience']}|{ch['expires_at_unix']}".encode()
        sig = wallet.sign(body)
        sig_b64 = base64.b64encode(sig).decode()
        out = self.verify(
            nonce=nonce, pubkey_pem=wallet.public_key_pem, signature_b64=sig_b64,
        )
        sid = out["session_id"]
        self._s.set_session(sid)
        return sid
