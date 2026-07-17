"""G10 — first-launch profile router.

A tiny read-only endpoint the post-install HTML one-pager
(`v2/ui/first_launch/index.html`) hits to render the user's
hardware tier, expected earnings, game-detection state, wallet
status, and compliance pre-check. Deliberately auth-free — the
installer launches this in localhost mode and the page is the
*onboarding* surface, before any keys exist.

Closed to the global rate limit through the same middleware as
the rest of the API; the operator can disable the router entirely
in production gateway mode by not mounting it."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Query, Request

from core.first_launch import build_profile

router = APIRouter(prefix="/v1/first_launch", tags=["onboarding"])


@router.get("/profile")
def profile(
    request: Request,
    *,
    idle_hours_per_day: float = Query(8.0, ge=0.0, le=24.0),
    power_cost_usd_per_kwh: float = Query(0.12, ge=0.0, le=2.0),
    country_code_hint: Optional[str] = Query(None, min_length=2, max_length=4),
) -> Dict[str, Any]:
    cc = country_code_hint or (
        request.headers.get("cf-ipcountry")
        or request.headers.get("x-country-code")
        or None
    )
    prof = build_profile(
        idle_hours_per_day=idle_hours_per_day,
        power_cost_usd_per_kwh=power_cost_usd_per_kwh,
        country_code_hint=cc,
    )
    return prof.to_dict()
