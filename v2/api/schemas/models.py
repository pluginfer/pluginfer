"""Pydantic v2 schemas for the Pluginfer REST API.

The wire contract is intentionally minimal so the same shape works for the
Python SDK, the JS SDK, and curl. Anything that callers can't express as
JSON (pubkey PEMs, signatures) ships as standard ASCII (PEM body, base64).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class VersionInfo(BaseModel):
    version: str
    git_sha: str
    api: str = "v1"


class NodeStatus(BaseModel):
    status: Literal["ok", "degraded", "starting"] = "ok"
    version: str
    git_sha: str
    chain_height: int
    peers_connected: int
    uptime_seconds: float


class WalletBalance(BaseModel):
    address: str
    balance_plg: float
    pending_plg: float = 0.0
    chain_height: int


class AuthChallenge(BaseModel):
    """Server-issued challenge that the client signs with their wallet
    private key. The wire format is a fixed-length nonce + timestamp so
    signatures are bound to a 30-second window."""
    nonce: str = Field(..., min_length=32, max_length=64)
    issued_at_unix: float
    expires_at_unix: float
    audience: str = "pluginfer-api"


class AuthVerify(BaseModel):
    nonce: str
    pubkey_pem: str
    signature_b64: str

    @field_validator("pubkey_pem")
    @classmethod
    def _looks_like_pem(cls, v: str) -> str:
        if "BEGIN PUBLIC KEY" not in v:
            raise ValueError("pubkey_pem must contain a PEM PUBLIC KEY block")
        return v


class JobCreate(BaseModel):
    """Submission body for POST /v1/jobs.

    `kind` selects which provider class can bid (e.g. 'llm.completion',
    'inference.generic'). `payload` is opaque to the API but routed to
    the chosen provider's execute() — providers know their own schema.
    """
    kind: str = Field(..., min_length=1, max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)
    cost_ceiling_usd: float = Field(0.10, ge=0.0, le=10_000.0)
    latency_ceiling_ms: int = Field(30_000, ge=10, le=10 * 60 * 1000)
    privacy_class: Literal["public", "internal", "confidential"] = "public"
    quality_floor: float = Field(0.7, ge=0.0, le=1.0)
    webhook_url: Optional[str] = None


class JobStatus(BaseModel):
    state: Literal[
        "queued", "matched", "running", "completed", "failed", "timeout",
        "cancelled",
    ]
    detail: Optional[str] = None


class JobInfo(BaseModel):
    """Returned by POST /v1/jobs and GET /v1/jobs/{id}."""
    job_id: str
    kind: str
    state: JobStatus
    submitted_at_unix: float
    matched_provider_pubkey: Optional[str] = None
    price_locked_usd: Optional[float] = None
    cost_ceiling_usd: float
    privacy_class: str


class JobResult(BaseModel):
    job_id: str
    state: JobStatus
    result_b64: Optional[str] = None
    result_hash_hex: Optional[str] = None
    provider_signature_b64: Optional[str] = None
    execution_ms: Optional[float] = None


class ProviderInfo(BaseModel):
    pubkey: str
    region: str = "unknown"
    kind: str
    quality_score: float = 0.0
    last_seen_unix: Optional[float] = None
    bid_count: int = 0
    win_count: int = 0
