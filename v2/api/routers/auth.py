"""Auth endpoints: challenge issue + signed-challenge verify."""

from __future__ import annotations

from fastapi import APIRouter, Request, status

from ..schemas import AuthChallenge, AuthVerify

router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.post("/challenge", response_model=AuthChallenge)
def issue_challenge(request: Request) -> AuthChallenge:
    backend = request.app.state.auth_backend
    c = backend.issue_challenge()
    return AuthChallenge(
        nonce=c.nonce,
        issued_at_unix=c.issued_at,
        expires_at_unix=c.expires_at,
        audience=c.audience,
    )


@router.post("/verify", status_code=status.HTTP_200_OK)
def verify_challenge(body: AuthVerify, request: Request) -> dict:
    backend = request.app.state.auth_backend
    session_id = backend.verify_challenge(
        nonce=body.nonce,
        pubkey_pem=body.pubkey_pem,
        signature_b64=body.signature_b64,
    )
    return {"session_id": session_id, "session_header": "X-Pluginfer-Session"}
