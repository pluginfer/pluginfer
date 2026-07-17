"""Provider-side gateway endpoints.

These close the loop between the in-tab browser provider and the
in-process auction. Three handlers, four URLs:

  POST  /v1/providers/register   — register a remote (browser) provider
  POST  /v1/providers/heartbeat  — keep-alive for the bid template
  GET   /v1/providers/open_jobs  — long-poll waiting jobs
  POST  /v1/providers/bid        — register/refresh the bid template
                                   (alias for register; kept for SDK
                                   compatibility with the spec we
                                   document for browser tabs)
  POST  /v1/providers/deliver    — submit the result blob

Auth model
----------
This router runs deliberately auth-free at the application level — the
browser tab's pubkey IS its identity, and we want the on-ramp to be a
single fetch with no key provisioning. Sybil resistance lives at the
auction layer: a freshly-registered tab has zero history and a low
quality score, so cheap-but-untrusted bids only win on small jobs
where the cost of slashing a misbehaver is bounded.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from core.browser_provider import (
    HttpBrowserProvider,
    HttpBrowserRegistry,
)

router = APIRouter(prefix="/v1/providers", tags=["providers"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RegisterBody(BaseModel):
    provider_pubkey_pem: str = Field(..., min_length=64)
    hardware_class: str = Field("browser-webgpu", max_length=128)
    price_per_1k_tok_usd: float = Field(0.0001, ge=0.0, le=10.0)
    base_eta_ms: int = Field(2000, ge=10, le=600_000)
    base_quality: float = Field(0.7, ge=0.0, le=1.0)
    privacy_grade: str = Field("public")
    # Optional human-readable handle the provider wants on the
    # leaderboard. Sanitised before display.
    nickname: Optional[str] = Field(None, max_length=64)
    # G6 — Sybil guard fields. Defaults keep backward-compatibility
    # with existing browser-tab clients while the new fields drive
    # tier promotion.
    webgpu_vendor: Optional[str] = Field(None, max_length=128)
    webgpu_architecture: Optional[str] = Field(None, max_length=128)
    webgpu_device: Optional[str] = Field(None, max_length=128)
    webgpu_driver: Optional[str] = Field(None, max_length=128)
    # Stake tx id from the chain side — when present + the deposit
    # ≥ MIN_PROVIDER_STAKE_PLG, the provider promotes to "staked".
    stake_tx_id: Optional[str] = Field(None, max_length=128)
    # Optional Cloudflare Turnstile / hCaptcha success token.
    # Combined with stake, this promotes to "verified".
    turnstile_token: Optional[str] = Field(None, max_length=2048)

    model_config = ConfigDict(extra="ignore")


class HeartbeatBody(BaseModel):
    provider_pubkey_pem: str

    model_config = ConfigDict(extra="ignore")


class DeliverBody(BaseModel):
    job_id: str
    provider_pubkey_pem: str
    result_bytes: str = Field(..., description="base64-encoded result blob")
    result_hash: str = Field(..., min_length=64, max_length=64)
    provider_sig: str = Field(..., description="ECDSA signature over result_hash")
    execution_ms: Optional[int] = Field(None, ge=0)

    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _provider_id_from_pem(pem: str) -> str:
    """Stable short id derived from the pubkey PEM. The provider's
    PEM is pasted unchanged into the auction so the receipt's
    `signature.pubkey` field already binds to identity; this id is
    purely a routing handle."""
    import hashlib
    body = pem.strip().replace("\r", "")
    return "br_" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]


def _registry(request: Request) -> HttpBrowserRegistry:
    reg = getattr(request.app.state, "browser_provider_registry", None)
    if reg is None:
        reg = HttpBrowserRegistry()
        request.app.state.browser_provider_registry = reg
    return reg


def _sybil_state(request: Request):
    """Lazy-init the per-app Sybil-guard state (rate limiter +
    fingerprint detector)."""
    from core.sybil_guard import (
        FingerprintSybilDetector,
        PerSubnetRateLimiter,
    )
    rl = getattr(request.app.state, "_sybil_ratelimiter", None)
    if rl is None:
        rl = PerSubnetRateLimiter()
        request.app.state._sybil_ratelimiter = rl
    fp = getattr(request.app.state, "_sybil_fingerprint_detector", None)
    if fp is None:
        fp = FingerprintSybilDetector()
        request.app.state._sybil_fingerprint_detector = fp
    return rl, fp


def _client_ip(request: Request) -> str:
    """Pull the real client IP. Honours Cloudflare's `CF-Connecting-IP`
    + any X-Forwarded-For chain; falls back to the socket peer."""
    cf = request.headers.get("cf-connecting-ip")
    if cf:
        return cf.strip()
    xff = request.headers.get("x-forwarded-for")
    if xff:
        # First entry in the chain is the original client.
        return xff.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return ""


def _verify_turnstile(token: str) -> bool:
    """Best-effort Cloudflare Turnstile verification. Returns True iff
    the token is present AND the verifier endpoint signs off. Absent
    or unverifiable -> False; downgrades to "staked" rather than
    "verified" via the tier resolver."""
    if not token:
        return False
    import os
    import urllib.error
    import urllib.parse
    import urllib.request
    secret = os.environ.get("TURNSTILE_SECRET_KEY", "")
    if not secret:
        # Operator hasn't configured Turnstile yet — we accept the
        # token's presence as a soft signal without crypto verifying.
        # This is the right default for closed beta; production should
        # set the secret.
        return True
    try:
        body = urllib.parse.urlencode(
            {"secret": secret, "response": token}
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://challenges.cloudflare.com/turnstile/v0/siteverify",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=2.0) as r:
            import json as _json
            return bool(_json.loads(r.read().decode("utf-8")).get("success"))
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _resolve_stake_plg(stake_tx_id: Optional[str], request: Request) -> float:
    """Map a stake-tx-id back to the staked PLG amount via the
    ComputeLedger / StakingContract attached to app.state. Absent or
    unverifiable -> 0.0 stake (the tier resolver downgrades the
    provider to 'untrusted')."""
    if not stake_tx_id:
        return 0.0
    ledger = getattr(request.app.state, "ledger", None)
    if ledger is None:
        return 0.0
    try:
        tx = getattr(ledger, "get_transaction_by_id", lambda _x: None)(stake_tx_id)
        if tx is None:
            return 0.0
        # Stake transactions are typed STAKE_DEPOSIT in the ledger; the
        # amount field holds the PLG. We don't reach into ledger
        # internals here — anything that quacks gets a float.
        amount = getattr(tx, "amount", None) or (
            tx.get("amount") if isinstance(tx, dict) else None
        )
        return float(amount) if amount is not None else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register_provider(body: RegisterBody, request: Request) -> Dict[str, Any]:
    # G5: sanctions screen at registration time. We screen the pubkey
    # AND the source IP's country (if Cloudflare or a fronting CDN
    # set CF-IPCountry / X-Country-Code). A denied registration emits
    # a compliance event and 451 ("Unavailable for Legal Reasons").
    from core.compliance import (
        emit_compliance_event,
        is_sanctioned_address,
        is_sanctioned_region,
    )
    addr_screen = is_sanctioned_address(body.provider_pubkey_pem)
    country = (
        request.headers.get("cf-ipcountry")
        or request.headers.get("x-country-code")
        or ""
    ).upper() or None
    region_screen = is_sanctioned_region(country)
    if not addr_screen.allowed or not region_screen.allowed:
        denied = addr_screen if not addr_screen.allowed else region_screen
        emit_compliance_event(
            "register-denied",
            **denied.to_event_dict(),
        )
        raise HTTPException(
            status_code=451,
            detail=denied.reason or "sanctions-block",
        )

    # G6: Sybil guardrails — per-/24 rate limit, fingerprint Sybil
    # detection, tier resolution from stake + Turnstile.
    from core.sybil_guard import fingerprint_hash, resolve_tier
    rl, fp_detector = _sybil_state(request)
    client_ip = _client_ip(request)
    if not rl.allow(client_ip):
        raise HTTPException(
            status_code=429,
            detail="per-subnet-rate-limit",
        )
    fp = fingerprint_hash(
        body.webgpu_vendor,
        body.webgpu_architecture,
        body.webgpu_device,
        body.webgpu_driver,
    )
    fp_clean = fp_detector.record_and_check(fp, client_ip)
    if not fp_clean:
        # Same GPU fingerprint registering from many subnets in a
        # short window — classic Sybil. We do not lie about it
        # (would invite probing); 429 with a structured reason.
        raise HTTPException(
            status_code=429,
            detail="fingerprint-sybil-block",
        )
    stake_plg = _resolve_stake_plg(body.stake_tx_id, request)
    turnstile_ok = _verify_turnstile(body.turnstile_token or "")
    tier = resolve_tier(stake_plg=stake_plg, turnstile_ok=turnstile_ok)

    reg = _registry(request)
    pid = _provider_id_from_pem(body.provider_pubkey_pem)
    p = reg.register(
        provider_id=pid,
        pubkey_pem=body.provider_pubkey_pem,
        hardware_class=body.hardware_class,
        base_price_per_1k_tok_usd=body.price_per_1k_tok_usd,
        base_eta_ms=body.base_eta_ms,
        base_quality=body.base_quality,
        privacy_grade=body.privacy_grade,
    )
    # Stamp tier metadata onto the provider so the auction's quality
    # scoring + leaderboard / receipts know how to weight it.
    p.tier = tier.tier
    p.stake_plg = tier.stake_plg
    p.fingerprint = fp
    # G6: hard cap on job cost — this is the primary defence against
    # malicious browser-tab providers during the first public test.
    # Untrusted tier defaults to the env-controlled MAX_USD (loose
    # for tests, $0.10 in production). bid() refuses to register a
    # bid above the cap.
    p.max_job_cost_usd = tier.max_job_cost_usd
    # Untrusted tier also gets a soft quality cap so the auction's
    # Pareto math excludes it from high-quality_floor jobs even
    # within the cost cap.
    if tier.tier == "untrusted":
        p.base_quality = min(p.base_quality, 0.55)
    auction = request.app.state.jobs.auction
    if not any(getattr(x, "provider_id", None) == pid for x in auction.providers):
        auction.register(p)
    return {
        "provider_id": pid,
        "registered_at_unix": p.last_seen_unix,
        "auction_size": len(auction.providers),
        "tier": tier.tier,
        "stake_plg": tier.stake_plg,
        "max_job_cost_usd": tier.max_job_cost_usd,
    }


# Spec alias: the browser-provider docs reference /providers/bid; we
# alias to register so a tab can re-paste its template with new pricing
# without juggling endpoints.
@router.post("/bid", status_code=status.HTTP_200_OK)
def update_bid(body: RegisterBody, request: Request) -> Dict[str, Any]:
    return register_provider(body, request)


@router.post("/heartbeat", status_code=status.HTTP_200_OK)
def heartbeat(body: HeartbeatBody, request: Request) -> Dict[str, Any]:
    reg = _registry(request)
    pid = _provider_id_from_pem(body.provider_pubkey_pem)
    p = reg.get(pid)
    if p is None:
        raise HTTPException(404, "provider_not_registered")
    p.heartbeat()
    return {"provider_id": pid, "last_seen_unix": p.last_seen_unix}


@router.get("/open_jobs")
def open_jobs(provider_pubkey: str, request: Request, limit: int = 8) -> Dict[str, Any]:
    reg = _registry(request)
    pid = _provider_id_from_pem(provider_pubkey)
    p = reg.get(pid)
    if p is None:
        # First-time tab — auto-register a minimal default profile so
        # the very first poll already lights up. Browser provider sends
        # a follow-up /register with full template right after.
        p = reg.register(
            provider_id=pid,
            pubkey_pem=provider_pubkey,
            hardware_class="browser-unknown",
            base_price_per_1k_tok_usd=0.0001,
            base_eta_ms=2000,
            base_quality=0.6,
            privacy_grade="public",
        )
        auction = request.app.state.jobs.auction
        if not any(getattr(x, "provider_id", None) == pid for x in auction.providers):
            auction.register(p)
    p.heartbeat()
    return {"provider_id": pid, "jobs": p.open_pickups(max_n=limit)}


@router.post("/deliver")
def deliver(body: DeliverBody, request: Request) -> Dict[str, Any]:
    reg = _registry(request)
    pid = _provider_id_from_pem(body.provider_pubkey_pem)
    p = reg.get(pid)
    if p is None:
        raise HTTPException(404, "provider_not_registered")
    delivered = p.deliver(body.job_id, {
        "status": "executed",
        "provider_id": pid,
        "job_id": body.job_id,
        "result_bytes": body.result_bytes,
        "result_hash": body.result_hash,
        "provider_sig": body.provider_sig,
        "provider_pubkey_pem": p.pubkey_pem,
        "execution_ms": body.execution_ms or 0,
    })
    if not delivered:
        # Common cases: job timed out, browser holding stale state, or
        # the tab raced two deliveries. The 410 tells the tab "drop it."
        raise HTTPException(410, "job_no_longer_pending")
    return {"job_id": body.job_id, "accepted": True}
