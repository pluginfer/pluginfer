"""Public receipts router — the marketing flywheel.

Every Pluginfer job emits a §D1 signed receipt. This router exposes the
local node's view of those receipts as a public, queryable leaderboard.

  GET /v1/receipts/leaderboard       — top providers by jobs / earnings
  GET /v1/receipts/{job_id}          — a single signed receipt blob
  GET /v1/receipts/provider/{pubkey} — every job a provider has served

Why this matters
----------------
1. **Reputation** — buyers see who actually delivers. Provider gossip
   has zero cost; signed receipts have provenance.
2. **Marketing** — providers brag about uptime + earnings. The
   leaderboard becomes a recruitment surface for the supply side.
3. **Cold-start** — the network bootstraps off receipt aggregation:
   one receipt is a transaction, ten thousand receipts is proof of a
   functioning market. Twitter screenshots write themselves.
4. **Audit** — every cell in the leaderboard is a signed claim. There
   is no equivalent on AWS / OpenAI / Anthropic.

This is a deliberately read-only router — no auth, no rate limit beyond
the global middleware. Anyone with the URL can curl it.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from ..jobs_service import JobRecord

router = APIRouter(prefix="/v1/receipts", tags=["receipts"])


def _record_to_receipt(rec: JobRecord) -> Dict[str, Any]:
    return {
        "job_id": rec.job_id,
        "kind": rec.kind,
        "state": rec.state,
        "submitted_at_unix": rec.submitted_at_unix,
        "completed_at_unix": rec.completed_at_unix,
        "execution_ms": rec.execution_ms,
        "provider_pubkey": rec.matched_provider_pubkey,
        "price_locked_usd": rec.price_locked_usd,
        "result_hash_hex": rec.result_hash_hex,
        "provider_signature_b64": rec.provider_signature_b64,
        "privacy_class": rec.privacy_class,
    }


@router.get("/leaderboard")
def leaderboard(
    request: Request,
    *,
    limit: int = Query(50, ge=1, le=500),
    since_unix: Optional[float] = Query(None, ge=0),
) -> Dict[str, Any]:
    """Top providers ranked by jobs completed, with USD earnings."""
    svc = request.app.state.jobs
    if svc is None:
        raise HTTPException(503, "jobs_service_unavailable")

    cutoff = since_unix or 0.0
    by_provider: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "provider_pubkey": "",
            "jobs_completed": 0,
            "jobs_failed": 0,
            "earnings_usd": 0.0,
            "avg_execution_ms": 0.0,
            "_total_ms": 0.0,
            "first_seen_unix": None,
            "last_seen_unix": None,
        }
    )

    for rec in svc.jobs.values():
        if rec.matched_provider_pubkey is None:
            continue
        if rec.submitted_at_unix < cutoff:
            continue
        row = by_provider[rec.matched_provider_pubkey]
        row["provider_pubkey"] = rec.matched_provider_pubkey
        row["first_seen_unix"] = (
            rec.submitted_at_unix if row["first_seen_unix"] is None
            else min(row["first_seen_unix"], rec.submitted_at_unix)
        )
        row["last_seen_unix"] = (
            rec.completed_at_unix or rec.submitted_at_unix
            if row["last_seen_unix"] is None
            else max(
                row["last_seen_unix"],
                rec.completed_at_unix or rec.submitted_at_unix,
            )
        )
        if rec.state == "completed":
            row["jobs_completed"] += 1
            row["earnings_usd"] += float(rec.price_locked_usd or 0.0)
            if rec.execution_ms is not None:
                row["_total_ms"] += rec.execution_ms
        elif rec.state in ("failed", "timeout"):
            row["jobs_failed"] += 1

    rows: List[Dict[str, Any]] = []
    for row in by_provider.values():
        completed = row["jobs_completed"]
        if completed:
            row["avg_execution_ms"] = row["_total_ms"] / completed
        row.pop("_total_ms", None)
        total = completed + row["jobs_failed"]
        row["uptime_pct"] = (
            (completed / total) * 100.0 if total > 0 else 0.0
        )
        rows.append(row)
    rows.sort(
        key=lambda r: (r["jobs_completed"], r["earnings_usd"]),
        reverse=True,
    )
    return {
        "generated_at_unix": time.time(),
        "since_unix": cutoff,
        "providers": rows[:limit],
        "total_providers": len(rows),
    }


@router.get("/provider/{provider_pubkey}")
def provider_receipts(
    provider_pubkey: str,
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
) -> Dict[str, Any]:
    svc = request.app.state.jobs
    if svc is None:
        raise HTTPException(503, "jobs_service_unavailable")
    receipts = [
        _record_to_receipt(rec)
        for rec in svc.jobs.values()
        if rec.matched_provider_pubkey == provider_pubkey
    ]
    receipts.sort(key=lambda r: r["submitted_at_unix"], reverse=True)
    completed = sum(1 for r in receipts if r["state"] == "completed")
    earnings = sum(
        r["price_locked_usd"] or 0.0
        for r in receipts if r["state"] == "completed"
    )
    return {
        "provider_pubkey": provider_pubkey,
        "summary": {
            "jobs_total": len(receipts),
            "jobs_completed": completed,
            "earnings_usd": earnings,
        },
        "receipts": receipts[:limit],
    }


@router.get("/{job_id}")
def receipt_for(job_id: str, request: Request) -> Dict[str, Any]:
    """Return the signed §A1 PNIS receipt for `job_id` when the gateway
    attested it (devserver path); otherwise the lightweight record-view
    that pre-dates W49. The signed form is the authoritative artefact
    for compliance / audit — verifiable by anyone who has the gateway
    pubkey."""
    svc = request.app.state.jobs
    if svc is None:
        raise HTTPException(503, "jobs_service_unavailable")
    signed = getattr(svc, "pnis_receipts", {}).get(job_id)
    if signed is not None:
        return signed
    rec = svc.get(job_id)
    if rec is None:
        raise HTTPException(404, "receipt_not_found")
    # Attempt a JIT attestation — covers jobs completed via the raw
    # /v1/jobs surface that didn't pass through the devserver shim.
    try:
        d = svc.attest_receipt(rec) if hasattr(svc, "attest_receipt") else None
    except Exception:
        d = None
    if d is not None:
        return d
    return _record_to_receipt(rec)
