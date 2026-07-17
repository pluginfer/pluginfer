"""HTTP routers for the Pluginfer REST API."""

from . import auth as auth_router
from . import compute as compute_router
from . import metrics as metrics_router
from . import provider_jobs as provider_jobs_router
from . import providers as providers_router
from . import receipts as receipts_router
from . import status as status_router
from . import wallet as wallet_router

__all__ = [
    "auth_router",
    "compute_router",
    "metrics_router",
    "provider_jobs_router",
    "providers_router",
    "receipts_router",
    "status_router",
    "wallet_router",
]
