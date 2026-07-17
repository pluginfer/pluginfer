"""GET /metrics - Prometheus exposition.

No auth required (Prometheus scrapers don't authenticate). If you want
to gate metrics behind a token, deploy a reverse proxy with a static
header check.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request
from starlette.responses import Response

from core.metrics import (
    REGISTRY,
    chain_height,
    peers_connected,
    uptime_seconds,
)

router = APIRouter(tags=["metrics"])
_T0 = time.monotonic()


@router.get("/metrics")
def metrics(request: Request) -> Response:
    # Refresh the gauges that mirror live state on each scrape.
    ledger = getattr(request.app.state, "ledger", None)
    if ledger is not None and hasattr(ledger, "chain"):
        chain_height.set(max(0, len(ledger.chain) - 1))
    peers_connected.set(getattr(request.app.state, "peers_connected", 0))
    uptime_seconds.set(time.monotonic() - _T0)
    return Response(REGISTRY.render(),
                    media_type="text/plain; version=0.0.4; charset=utf-8")
