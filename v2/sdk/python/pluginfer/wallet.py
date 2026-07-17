"""Wallet API surface."""

from __future__ import annotations

from ._http import HttpSession
from .types import WalletBalance


class WalletAPI:
    def __init__(self, session: HttpSession) -> None:
        self._s = session

    def balance(self) -> WalletBalance:
        d = self._s.get("/v1/wallet/balance")
        return WalletBalance(
            address=d["address"],
            balance_plg=d["balance_plg"],
            pending_plg=d.get("pending_plg", 0.0),
            chain_height=d.get("chain_height", 0),
        )
