"""Typed dataclasses for SDK return values.

Wrapping the API's wire JSON in dataclasses gives users IDE
autocompletion and stable attribute access independent of any future
JSON-shape changes (the SDK then becomes the compatibility layer)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JobState:
    state: str
    detail: Optional[str] = None


@dataclass
class Job:
    job_id: str
    kind: str
    state: JobState
    submitted_at_unix: float
    matched_provider_pubkey: Optional[str] = None
    price_locked_usd: Optional[float] = None
    cost_ceiling_usd: float = 0.0
    privacy_class: str = "public"

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        s = d.get("state", {})
        return cls(
            job_id=d["job_id"],
            kind=d["kind"],
            state=JobState(state=s.get("state", "unknown"), detail=s.get("detail")),
            submitted_at_unix=d.get("submitted_at_unix", 0.0),
            matched_provider_pubkey=d.get("matched_provider_pubkey"),
            price_locked_usd=d.get("price_locked_usd"),
            cost_ceiling_usd=d.get("cost_ceiling_usd", 0.0),
            privacy_class=d.get("privacy_class", "public"),
        )


@dataclass
class JobResult:
    job_id: str
    state: JobState
    result_b64: Optional[str] = None
    result_hash_hex: Optional[str] = None
    provider_signature_b64: Optional[str] = None
    execution_ms: Optional[float] = None

    @classmethod
    def from_dict(cls, d: dict) -> "JobResult":
        s = d.get("state", {})
        return cls(
            job_id=d["job_id"],
            state=JobState(state=s.get("state", "unknown"), detail=s.get("detail")),
            result_b64=d.get("result_b64"),
            result_hash_hex=d.get("result_hash_hex"),
            provider_signature_b64=d.get("provider_signature_b64"),
            execution_ms=d.get("execution_ms"),
        )


@dataclass
class WalletBalance:
    address: str
    balance_plg: float
    pending_plg: float = 0.0
    chain_height: int = 0


@dataclass
class Provider:
    pubkey: str
    kind: str
    quality_score: float = 0.0
    region: str = "unknown"


@dataclass
class Status:
    status: str
    version: str
    git_sha: str
    chain_height: int
    peers_connected: int
    uptime_seconds: float
