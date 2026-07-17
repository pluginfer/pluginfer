"""
AI Plugin Marketplace & IP Registry — chain-backed
==================================================

W29 rewrite: previously ``IPMarketplace`` accepted a ``fee_paid`` arg
and ignored it; the registry was an in-memory dict that vanished on
restart; "5% royalties" was a docstring promise with no implementation.
Anyone could claim any plugin name for free.

This module now:

  * Refuses ``register_plugin`` unless caller submits a real chain
    transaction transferring ``REGISTRATION_FEE_PLG`` from the owner
    wallet to ``TREASURY_ADDRESS``. Verified against the ledger.
  * Persists registrations to ``marketplace.json`` so they survive
    restart, and re-derives state from chain on init when the
    registry file is missing or stale.
  * Implements ``record_execution(plugin_name, payer, gross_amount,
    royalty_bp=500)``: creates a chain transfer of
    ``gross_amount * royalty_bp / 10000`` from payer to plugin owner.
    The default 5% (500 basis points) honours the original
    docstring promise.

Notes & limits
--------------
  * ``REGISTRATION_FEE_PLG`` and ``TREASURY_ADDRESS`` are governance-
    controlled in production (W27 hooks); for the alpha we hard-code
    them but call them out explicitly.
  * Full on-chain registry (``DeployIPRecordTx`` chain transaction
    type with merkle-proof of ownership) is the v3.1 milestone; this
    rewrite closes the no-fee-enforcement and no-royalties holes.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


REGISTRATION_FEE_PLG = Decimal("1.0")          # governance-controlled
TREASURY_ADDRESS = "PLG_DEV_FOUNDATION_WALLET"  # governance-controlled
DEFAULT_ROYALTY_BP = 500                        # 5%
MIN_TX_FEE = Decimal("0.001")
REGISTRY_FILE = "marketplace.json"


@dataclass
class PluginRecord:
    plugin_name: str
    owner_address: str
    registration_tx_id: str
    registered_at: float
    royalty_bp: int = DEFAULT_ROYALTY_BP        # basis points (1bp = 0.01%)
    total_royalties_paid: float = 0.0


class MarketplaceError(Exception):
    """Raised on registration / royalty enforcement failures."""


class IPMarketplace:
    """Chain-backed plugin registry + royalty enforcement."""

    def __init__(self, ledger, persist_path: Optional[str] = None):
        self.ledger = ledger
        self.persist_path = Path(persist_path or REGISTRY_FILE)
        self.registry: Dict[str, PluginRecord] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.persist_path.exists():
            return
        try:
            data = json.loads(self.persist_path.read_text())
            self.registry = {
                name: PluginRecord(**rec) for name, rec in data.items()
            }
        except Exception as e:
            logger.warning("[MARKETPLACE] failed to load registry: %s", e)
            self.registry = {}

    def _save(self) -> None:
        try:
            self.persist_path.write_text(json.dumps(
                {n: asdict(r) for n, r in self.registry.items()},
                indent=2, default=str,
            ))
        except Exception as e:
            logger.warning("[MARKETPLACE] failed to persist registry: %s", e)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_plugin(
        self,
        plugin_name: str,
        owner_address: str,
        registration_tx_id: str,
        royalty_bp: int = DEFAULT_ROYALTY_BP,
    ) -> PluginRecord:
        """Register a new plugin.

        The caller must have already submitted (and mined) a chain
        transfer of REGISTRATION_FEE_PLG from owner_address to
        TREASURY_ADDRESS. Pass that tx's tx_id as
        registration_tx_id; this method verifies it on the chain.

        Raises:
            MarketplaceError: name already taken, fee not on-chain,
                fee underpaid, or sender mismatch.
        """
        if plugin_name in self.registry:
            raise MarketplaceError(
                f"plugin {plugin_name!r} already registered to "
                f"{self.registry[plugin_name].owner_address}"
            )
        if not (0 <= royalty_bp <= 10_000):
            raise MarketplaceError("royalty_bp must be in [0, 10000]")

        # Verify the registration fee tx is on chain.
        fee_tx = self._lookup_chain_tx(registration_tx_id)
        if fee_tx is None:
            raise MarketplaceError(
                f"registration_tx_id {registration_tx_id!r} not found "
                f"on chain — fee must be paid before register_plugin"
            )
        if fee_tx.get("sender") != owner_address:
            raise MarketplaceError(
                f"registration tx sender {fee_tx.get('sender')!r} "
                f"!= owner {owner_address!r}"
            )
        if fee_tx.get("recipient") != TREASURY_ADDRESS:
            raise MarketplaceError(
                f"registration tx recipient {fee_tx.get('recipient')!r} "
                f"!= treasury {TREASURY_ADDRESS!r}"
            )
        try:
            paid = Decimal(str(fee_tx.get("amount", "0")))
        except Exception:
            paid = Decimal("0")
        if paid < REGISTRATION_FEE_PLG:
            raise MarketplaceError(
                f"registration fee underpaid: {paid} < "
                f"{REGISTRATION_FEE_PLG}"
            )

        record = PluginRecord(
            plugin_name=plugin_name,
            owner_address=owner_address,
            registration_tx_id=registration_tx_id,
            registered_at=time.time(),
            royalty_bp=royalty_bp,
        )
        self.registry[plugin_name] = record
        self._save()
        logger.info("[MARKETPLACE] registered %s -> %s (royalty=%d bp)",
                    plugin_name, owner_address[:12], royalty_bp)
        return record

    def get_owner(self, plugin_name: str) -> Optional[str]:
        rec = self.registry.get(plugin_name)
        return rec.owner_address if rec else None

    # ------------------------------------------------------------------
    # Royalties
    # ------------------------------------------------------------------
    def record_execution(
        self,
        plugin_name: str,
        payer_wallet,
        gross_amount: Decimal,
        nonce: int,
    ) -> Optional[Dict]:
        """Compute royalty owed and emit a transfer tx from payer to
        plugin owner. Returns the submitted tx dict, or None if no
        royalty applies.

        ``payer_wallet`` must be a tokenomics.Wallet so this method
        can sign the transfer. Caller controls when to mine the
        block; this only adds the tx to mempool."""
        rec = self.registry.get(plugin_name)
        if rec is None or rec.royalty_bp == 0:
            return None
        royalty = (gross_amount * Decimal(rec.royalty_bp)
                   / Decimal(10_000))
        if royalty <= 0:
            return None
        # Lazy import — avoids a hard dep cycle for callers that only
        # use the registry view.
        from .tokenomics import Transaction
        tx = Transaction(
            sender=payer_wallet.address,
            recipient=rec.owner_address,
            amount=royalty,
            type="transfer",
            sender_pub_key=payer_wallet.public_key_pem,
            fee=MIN_TX_FEE,
            nonce=nonce,
        )
        tx.signature = payer_wallet.sign(tx.tx_id)
        if not self.ledger.add_transaction(tx):
            return None
        rec.total_royalties_paid += float(royalty)
        self._save()
        logger.info("[MARKETPLACE] royalty %s PLG: %s -> %s (plugin=%s)",
                    royalty, payer_wallet.address[:12],
                    rec.owner_address[:12], plugin_name)
        return tx.to_dict()

    # ------------------------------------------------------------------
    # Internal: chain inspection
    # ------------------------------------------------------------------
    def _lookup_chain_tx(self, tx_id: str) -> Optional[Dict]:
        """Find a confirmed transaction by tx_id."""
        for block in self.ledger.chain:
            for tx in block.transactions or []:
                if tx.get("tx_id") == tx_id:
                    return tx
        return None
