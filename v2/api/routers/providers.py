"""Provider directory endpoints."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, Request

from ..middleware.auth import require_auth
from ..schemas import ProviderInfo

router = APIRouter(prefix="/v1/providers", tags=["providers"])


@router.get("", response_model=List[ProviderInfo])
def list_providers(request: Request, identity: str = Depends(require_auth)) -> List[ProviderInfo]:
    svc = request.app.state.jobs
    out: List[ProviderInfo] = []
    for p in svc.auction.providers:
        out.append(ProviderInfo(
            pubkey=getattr(p, "provider_id", "unknown"),
            kind=getattr(p, "kind", "compute"),
            quality_score=float(getattr(p, "quality_score", 0.0)),
            region=getattr(p, "region", "unknown"),
        ))
    return out
