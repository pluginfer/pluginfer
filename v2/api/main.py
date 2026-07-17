"""FastAPI app factory for the Pluginfer REST API.

Production entrypoint:

    from api.main import build_app
    from core.compute_ledger import ComputeLedger
    from core.tokenomics import Wallet
    from core.providers import Auction
    app = build_app(
        ledger=ComputeLedger(),
        wallet=Wallet.load_or_create(),
        auction=Auction(),
    )

Then run via uvicorn:

    uvicorn api.main:app --host 0.0.0.0 --port 8100

The factory takes its dependencies as arguments so tests can inject
in-memory chains + fake providers without monkey-patching.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .jobs_service import JobsService
from .middleware import AuthBackend, RateLimitMiddleware, RequestIDMiddleware
from .routers import (
    auth_router,
    compute_router,
    metrics_router,
    provider_jobs_router,
    providers_router,
    receipts_router,
    status_router,
    wallet_router,
)


def build_app(
    *,
    ledger=None,
    wallet=None,
    auction=None,
    auth_backend: Optional[AuthBackend] = None,
    rate_limit_capacity: float = 60.0,
    rate_limit_refill_per_sec: float = 1.0,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    if auction is None:
        from core.providers import Auction
        auction = Auction()

    app = FastAPI(
        title="Pluginfer API",
        version="1.0.0",
        description="Distributed AI compute mesh — REST API.",
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    app.state.ledger = ledger
    app.state.wallet = wallet
    app.state.peers_connected = 0
    app.state.auth_backend = auth_backend or AuthBackend()
    app.state.jobs = JobsService(auction=auction)

    # Middleware (order matters: outermost first).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        capacity=rate_limit_capacity,
        refill_per_sec=rate_limit_refill_per_sec,
    )

    # Routers.
    app.include_router(status_router.router)
    app.include_router(auth_router.router)
    app.include_router(compute_router.router)
    app.include_router(wallet_router.router)
    app.include_router(providers_router.router)
    app.include_router(provider_jobs_router.router)
    app.include_router(receipts_router.router)
    app.include_router(metrics_router.router)

    return app


# Convenience for `uvicorn api.main:app` invocation. Production callers
# should use build_app() with their own ledger / wallet / auction.
app = build_app()
