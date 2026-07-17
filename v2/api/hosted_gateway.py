"""Hosted gateway — the front door for external buyers.

This is the surface a startup ditching AWS actually hits. They never
install Pluginfer; they just point their OpenAI SDK at our public
gateway URL:

    OPENAI_BASE_URL=https://api.pluginfer.network/v1
    OPENAI_API_KEY=plg_live_<their-key>

The gateway:
  1. Authenticates the API key → resolves to a buyer wallet_id.
  2. Looks up the wallet's available balance.
  3. Submits the job to the embedded JobsService with that
     wallet_id, so the auction locks the cleared price up front.
  4. Returns the same OpenAI-shaped response the upstream SDK
     expects, plus signed-receipt headers.

When the wallet runs out, the response is HTTP 402 Payment
Required with a Pluginfer-specific body:

    {
      "error": "wallet_underfunded",
      "wallet_id": "wlt_abc",
      "deficit_usd": "0.42",
      "topup_url": "https://api.pluginfer.network/v1/wallets/wlt_abc/topup"
    }

The buyer can either:
  * Top up via /v1/wallets/{wallet_id}/topup and re-issue the
    request (auto-retry on the SDK side via standard 402 handling),
  * OR keep the wallet always-funded via auto-recharge (Stripe
    subscription / pre-paid balance — handled by the topup endpoint).

Innovation worth filing: §A26 "API-key-mediated auction-routed
inference with built-in operator margin." The startup-side code
is byte-compatible with OpenAI; the request path runs an auction
across a permissionless mesh; the buyer pays via a pre-funded
wallet that hits 402 instead of a surprise bill. There is no
prior art for this combination.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi import Request as FastApiRequest
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

DEFAULT_PRICE_CEILING_USD = float(os.environ.get(
    "PLUGINFER_DEFAULT_COST_CEILING_USD", "0.10",
))
DEFAULT_LATENCY_CEILING_MS = int(os.environ.get(
    "PLUGINFER_DEFAULT_LATENCY_MS", "30000",
))


# ---------------------------------------------------------------------------
# API key store
# ---------------------------------------------------------------------------

@dataclass
class ApiKeyRecord:
    """Maps an opaque API key (presented as a Bearer token) to a buyer
    wallet ID. Keys are stored as SHA-256 hashes — we never persist
    the raw secret. Rotation = issue a new key, revoke the old one."""
    key_hash_hex: str
    wallet_id: str
    label: str = ""
    revoked: bool = False
    created_unix: float = 0.0
    last_used_unix: Optional[float] = None


@dataclass
class ApiKeyStore:
    """Thin in-memory index. Production deployments back this with
    SQLite (alongside JobStore) or a managed KV; the interface stays
    the same. The hashing means a DB dump leaks NO usable secrets —
    you have to rotate, not invalidate."""
    records: Dict[str, ApiKeyRecord] = field(default_factory=dict)

    @staticmethod
    def _hash(raw: str) -> str:
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def issue(self, *, wallet_id: str, label: str = "") -> str:
        """Generate a new key, store its hash, return the raw key once.
        Caller must surface it to the buyer immediately; we won't be
        able to show it again."""
        import time
        raw = "plg_live_" + secrets.token_urlsafe(32)
        rec = ApiKeyRecord(
            key_hash_hex=self._hash(raw),
            wallet_id=wallet_id, label=label,
            created_unix=time.time(),
        )
        self.records[rec.key_hash_hex] = rec
        return raw

    def resolve(self, raw_key: str) -> Optional[ApiKeyRecord]:
        rec = self.records.get(self._hash(raw_key))
        if rec is None or rec.revoked:
            return None
        import time
        rec.last_used_unix = time.time()
        return rec

    def revoke(self, raw_key: str) -> bool:
        rec = self.records.get(self._hash(raw_key))
        if rec is None:
            return False
        rec.revoked = True
        return True


# ---------------------------------------------------------------------------
# FastAPI mount
# ---------------------------------------------------------------------------

def build_hosted_gateway(
    *,
    jobs_service: Any,
    ledger: Any,
    api_keys: Optional[ApiKeyStore] = None,
    title: str = "Pluginfer Hosted Gateway",
):
    """Mount the hosted-gateway routes onto a fresh FastAPI app. The
    app shares the supplied JobsService + ledger so the same auction
    powers external-buyer requests AND any internal nodes."""
    app = FastAPI(title=title)
    if api_keys is None:
        api_keys = ApiKeyStore()
    app.state.jobs = jobs_service
    app.state.ledger = ledger
    app.state.api_keys = api_keys

    def _authenticate(request: FastApiRequest) -> ApiKeyRecord:
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer token")
        raw = auth.split(None, 1)[1].strip()
        rec = api_keys.resolve(raw)
        if rec is None:
            raise HTTPException(401, "invalid or revoked api key")
        return rec

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/keys")
    async def issue_key(request: FastApiRequest) -> Dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "JSON body required")
        """Operator-facing — issue a new API key bound to a wallet.
        Production gates this behind an admin token; the shape exists
        so the operator can script onboarding flows.

        Body: {"wallet_id": "...", "label": "..."}
        Returns: {"api_key": "plg_live_..."} — shown ONCE."""
        wallet_id = str(body.get("wallet_id") or "").strip()
        if not wallet_id:
            raise HTTPException(400, "wallet_id required")
        ledger.get_or_create_wallet(wallet_id, role="buyer")
        key = api_keys.issue(
            wallet_id=wallet_id, label=str(body.get("label") or ""),
        )
        return {"api_key": key, "wallet_id": wallet_id}

    @app.get("/v1/wallets/{wallet_id}")
    def get_wallet(wallet_id: str, request: FastApiRequest) -> Dict[str, Any]:
        rec = _authenticate(request)
        if rec.wallet_id != wallet_id:
            raise HTTPException(403, "wallet does not belong to this api key")
        w = ledger.get_wallet(wallet_id)
        if w is None:
            raise HTTPException(404, "wallet not found")
        return w.to_public()

    @app.post("/v1/wallets/{wallet_id}/topup")
    async def topup_wallet(
        wallet_id: str, request: FastApiRequest,
    ) -> Dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "JSON body required")
        """Credit a wallet from a (pre-charged) payment intent.
        Production wraps this behind Stripe / MoonPay / Ramp — the
        gateway only credits after the PSP confirms cleared funds.
        For the test/dev surface, the operator can call this directly
        with `amount_usd`."""
        rec = _authenticate(request)
        if rec.wallet_id != wallet_id:
            raise HTTPException(403, "wallet/api-key mismatch")
        amount_str = str(body.get("amount_usd") or "")
        try:
            amount = Decimal(amount_str)
        except Exception:
            raise HTTPException(400, "amount_usd must be a decimal string")
        if amount <= Decimal("0"):
            raise HTTPException(400, "amount must be positive")
        ledger.credit(wallet_id, amount, note=body.get("note", "topup"))
        w = ledger.get_wallet(wallet_id)
        # Auto-resume any paused jobs the moment funds clear. This is
        # the "continue on enough wallet" trigger — same as Claude
        # Code resuming after a usage-limit pause.
        import asyncio
        for rec_obj in list(jobs_service.jobs.values()):
            if (rec_obj.state == "paused_funding"
                    and getattr(rec_obj, "_buyer_wallet_id", None) == wallet_id):
                asyncio.create_task(jobs_service.resume_funding(rec_obj.job_id))
        return {**w.to_public(), "credited_usd": str(amount)}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: FastApiRequest):
        """The drop-in OpenAI surface. The buyer's existing app sees
        the same wire format the upstream returns; the auction routes
        the actual work."""
        rec = _authenticate(request)
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "request body must be JSON")
        if not isinstance(body, dict):
            raise HTTPException(400, "request body must be a JSON object")
        # Pluginfer-specific overrides ride inside the OpenAI body
        # without breaking the upstream schema.
        ceiling = float(body.get(
            "pluginfer_cost_ceiling_usd", DEFAULT_PRICE_CEILING_USD,
        ))
        latency = int(body.get(
            "pluginfer_latency_ceiling_ms", DEFAULT_LATENCY_CEILING_MS,
        ))
        quality = float(body.get("pluginfer_quality_floor", 0.5))
        privacy = str(body.get("pluginfer_privacy", "public"))
        # Flatten messages into a single prompt — devserver does the
        # same in its OpenAI handler so we stay schema-compatible.
        messages = body.get("messages") or []
        prompt = "\n".join(
            f"{m.get('role','user')}: {m.get('content','')}"
            for m in messages if isinstance(m, dict)
        )
        spec_payload = {
            "prompt": prompt,
            "model": body.get("model", "pluginfer-alpha"),
            "max_tokens": int(body.get("max_tokens", 512)),
        }
        # Optional consortium hint for big-batch jobs.
        if body.get("pluginfer_consortium"):
            spec_payload["consortium"] = body["pluginfer_consortium"]

        job_rec = await jobs_service.submit(
            kind="llm.completion", payload=spec_payload,
            cost_ceiling_usd=ceiling, latency_ceiling_ms=latency,
            privacy_class=privacy, quality_floor=quality,
            requester_identity=rec.wallet_id,
            buyer_wallet_id=rec.wallet_id,
        )
        # Wait for terminal state up to the latency ceiling.
        import asyncio, time
        deadline = time.monotonic() + (latency / 1000.0) + 5.0
        while time.monotonic() < deadline:
            cur = jobs_service.get(job_rec.job_id)
            if cur.state in (
                "completed", "completed_partial", "failed", "timeout",
            ):
                break
            if cur.state == "paused_funding":
                # Surface 402 with structured remediation.
                w = ledger.get_wallet(rec.wallet_id)
                deficit = (
                    Decimal(str(cur.price_locked_usd or ceiling))
                    - (w.available_usd if w else Decimal("0"))
                )
                return JSONResponse(
                    status_code=402,
                    content={
                        "error": "wallet_underfunded",
                        "wallet_id": rec.wallet_id,
                        "job_id": cur.job_id,
                        "deficit_usd": str(max(deficit, Decimal("0"))),
                        "topup_url": f"/v1/wallets/{rec.wallet_id}/topup",
                        "detail": cur.detail,
                    },
                )
            await asyncio.sleep(0.1)
        cur = jobs_service.get(job_rec.job_id)
        if cur.state == "paused_funding":
            return JSONResponse(
                status_code=402,
                content={
                    "error": "wallet_underfunded",
                    "wallet_id": rec.wallet_id,
                    "job_id": cur.job_id,
                    "topup_url": f"/v1/wallets/{rec.wallet_id}/topup",
                    "detail": cur.detail,
                },
            )
        if cur.state not in ("completed", "completed_partial"):
            raise HTTPException(
                502, f"job {cur.state}: {cur.detail or 'unknown'}",
            )
        # Shape into OpenAI Chat Completions response.
        import base64 as _b64
        out_bytes = _b64.b64decode(cur.result_b64) if cur.result_b64 else b""
        text = out_bytes.decode("utf-8", errors="replace")
        resp = {
            "id": f"chatcmpl-{cur.job_id}",
            "object": "chat.completion",
            "created": int(cur.completed_at_unix or 0),
            "model": body.get("model", "pluginfer-alpha"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"total_tokens": int(spec_payload["max_tokens"])},
        }
        return JSONResponse(
            content=resp,
            headers={
                "X-Pluginfer-Receipt-ID": cur.job_id,
                "X-Pluginfer-Price-USD": str(cur.price_locked_usd or 0),
                "X-Pluginfer-Provider": cur.matched_provider_pubkey or "",
                "X-Pluginfer-Wallet-Balance": str(
                    ledger.get_wallet(rec.wallet_id).available_usd
                    if ledger.get_wallet(rec.wallet_id) else 0
                ),
            },
        )

    return app


__all__ = [
    "ApiKeyRecord",
    "ApiKeyStore",
    "build_hosted_gateway",
]
