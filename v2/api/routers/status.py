"""Service-level status + version endpoints (no auth required)."""

from __future__ import annotations

import os
import time

from fastapi import APIRouter, Request

from ..schemas import NodeStatus, VersionInfo

router = APIRouter(prefix="/v1", tags=["status"])

_STARTED_AT = time.monotonic()


def _version() -> str:
    return os.environ.get("PLUGINFER_VERSION", "1.0.0")


def _git_sha() -> str:
    return os.environ.get("PLUGINFER_GIT_SHA", "unknown")


@router.get("/version", response_model=VersionInfo)
def get_version() -> VersionInfo:
    return VersionInfo(version=_version(), git_sha=_git_sha())


@router.get("/status", response_model=NodeStatus)
def get_status(request: Request) -> NodeStatus:
    ledger = getattr(request.app.state, "ledger", None)
    chain_height = ledger.height() if ledger is not None and hasattr(ledger, "height") else 0
    if chain_height == 0 and ledger is not None and hasattr(ledger, "blocks"):
        chain_height = max(0, len(ledger.blocks) - 1)
    peers = getattr(request.app.state, "peers_connected", 0)
    return NodeStatus(
        version=_version(),
        git_sha=_git_sha(),
        chain_height=chain_height,
        peers_connected=peers,
        uptime_seconds=time.monotonic() - _STARTED_AT,
    )
