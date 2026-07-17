"""Top-level SDK client."""

from __future__ import annotations

from typing import Optional

import httpx

from ._http import HttpSession
from .auth import AuthAPI
from .jobs import JobsAPI
from .providers import ProvidersAPI
from .types import Status
from .wallet import WalletAPI


class PluginferClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.pluginfer.network",
        timeout: float = 30.0,
        transport: Optional[httpx.BaseTransport] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._session = HttpSession(
            base_url=base_url, api_key=api_key, timeout=timeout,
            transport=transport, client=http_client,
        )
        self.jobs = JobsAPI(self._session)
        self.wallet = WalletAPI(self._session)
        self.providers = ProvidersAPI(self._session)
        self.auth = AuthAPI(self._session)

    @property
    def base_url(self) -> str:
        return self._session.base_url

    def status(self) -> Status:
        d = self._session.get("/v1/status")
        return Status(
            status=d["status"], version=d["version"], git_sha=d["git_sha"],
            chain_height=d["chain_height"], peers_connected=d["peers_connected"],
            uptime_seconds=d["uptime_seconds"],
        )

    def version(self) -> dict:
        return self._session.get("/v1/version")

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "PluginferClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# Public alias kept short for import ergonomics.
Pluginfer = PluginferClient
