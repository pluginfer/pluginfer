"""
Cross-Chain Interop — INTENTIONALLY NOT IMPLEMENTED
===================================================
The previous version of this module returned `True` after a 0.5s sleep
for `verify_deposit`, and returned `f"0x{int(time.time())}abcdef"` from
`release_assets`. That's a fake bridge. Connecting it to real funds
would lose them.

To ship real cross-chain interop you need:
    * web3.py           — Ethereum RPC + ABI handling
    * solders / solana  — Solana RPC + transaction signing
    * watchtowers       — independent re-org monitors
    * timelocks         — dispute windows on lock-and-mint flows
    * audited contracts — at least Trail of Bits / Halborn level

This stub raises with a clear message. The architectural plan for v3
is to NOT build a custom bridge at all; instead, settle PLG natively
on Solana as an SPL token and use the Wormhole/LayerZero bridges
that already exist and are professionally audited. That removes this
file entirely.

Effort to delete + migrate: ~3 weeks.
"""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class InteropNotImplementedError(NotImplementedError):
    """Raised by every bridge method until a real backend lands."""


class ChainBridge:
    AVAILABLE = False

    def __init__(self, chain_name: str, rpc_url: str):
        self.chain = chain_name
        self.rpc = rpc_url
        logger.warning("ChainBridge[%s] is not implemented. Use Solana SPL natively.",
                       chain_name)

    @classmethod
    def is_available(cls) -> bool:
        return cls.AVAILABLE

    def verify_deposit(self, tx_hash: str, amount: float, sender: str) -> bool:
        raise InteropNotImplementedError(
            "Real bridge not implemented. Plan: migrate PLG to Solana SPL "
            "and use Wormhole / LayerZero. See core/interop.py docstring."
        )

    def release_assets(self, recipient: str, amount: float) -> str:
        raise InteropNotImplementedError(
            "Real bridge not implemented. See core/interop.py docstring."
        )


class BridgeManager:
    AVAILABLE = False

    def __init__(self, ledger):
        self.ledger = ledger
        self.bridges: Dict[str, ChainBridge] = {}

    def bridge_in(self, chain: str, tx_hash: str, amount: float, user_address: str):
        raise InteropNotImplementedError(
            "Bridge not implemented. See core/interop.py docstring."
        )

    def bridge_out(self, chain: str, amount: float, user_address: str):
        raise InteropNotImplementedError(
            "Bridge not implemented. See core/interop.py docstring."
        )
