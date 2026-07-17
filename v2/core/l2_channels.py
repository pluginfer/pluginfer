"""
Layer-2 State Channels — INTENTIONALLY NOT IMPLEMENTED IN v3.0-alpha
====================================================================
The previous version of this module advertised "Instant Settlement"
and "Ripple-like speed", and shipped a class shaped like a real
payment channel. None of the security properties held:

    * `instant_transfer(channel_id, sender, amount, signature)` —
      the `signature` parameter was accepted and **never verified**.
      Any caller drains any channel.
    * `open_channel` claimed to "lock funds on-chain" but only
      *checked* the balance — never actually deducted. Funds remained
      spendable elsewhere while the channel claimed to hold them.
    * `close_channel` was a no-op:
        # Note: We aren't actually writing back to ledger
      So the channel had no on-chain settlement.
    * `signatures = {}` was declared and never populated. No state
      proof was ever recorded.

A real Pluginfer-grade L2 needs:
    1. **On-chain lock TX** at open. Funds removed from L1 balance
       and credited to the channel's escrow address.
    2. **Off-chain signed state updates** — both parties sign each
       new (balance_a, balance_b, nonce) tuple. Nonce is monotonic.
    3. **Cooperative close** — both sign the final state, one
       on-chain settlement TX. Cheap.
    4. **Unilateral close + challenge window** (~24-48 h) — either
       party broadcasts the latest state they have. The other can
       counter with a higher-nonce signed state. After the window,
       balances settle.
    5. **Watchtowers** — third-party services that watch the chain
       for stale-state submissions while a participant is offline.

This is ~1500 lines of careful protocol work plus formal modeling
of the dispute path. It is NOT something to ship as a stub-with-
side-effects, because the stub silently moves imaginary balances
and gives users false confidence.

Until it lands, this module raises. Code that needs sub-second
settlement should batch through the L1 mempool with a 0-conf
mempool view (acceptable for low-trust amounts).
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Dict

logger = logging.getLogger(__name__)


class L2NotImplementedError(NotImplementedError):
    """Raised by every L2 channel method until the protocol is real."""


class PaymentChannel:
    """
    Public data class. Constructing one does not lock any funds.
    Methods that move balances are on `StateChannelManager` and they
    raise until the protocol is implemented.
    """

    def __init__(self, channel_id: str, participant_a: str,
                 participant_b: str, balance_a: Decimal,
                 balance_b: Decimal):
        self.channel_id = channel_id
        self.participants = [participant_a, participant_b]
        self.balances = {participant_a: balance_a, participant_b: balance_b}
        self.nonce = 0                # off-chain state counter
        self.is_open = True
        self.signatures: Dict[int, Dict[str, str]] = {}

    def to_dict(self) -> Dict:
        return {
            "id": self.channel_id,
            "participants": self.participants,
            "balances": {k: str(v) for k, v in self.balances.items()},
            "nonce": self.nonce,
            "open": self.is_open,
        }


class StateChannelManager:
    """
    Honest stub. Every method raises with a clear remediation message.
    The previous mock silently moved imaginary balances and gave users
    false confidence — that is worse than no L2 at all.
    """

    AVAILABLE = False

    def __init__(self, ledger):
        self.ledger = ledger
        self.channels: Dict[str, PaymentChannel] = {}
        logger.warning(
            "StateChannelManager not implemented in v3.0-alpha. "
            "Use the L1 mempool for now; L2 channels are tracked as W26."
        )

    @classmethod
    def is_available(cls) -> bool:
        return cls.AVAILABLE

    def open_channel(self, user_a_wallet, user_b_addr: str,
                     amount: Decimal) -> str:
        raise L2NotImplementedError(
            "open_channel requires an on-chain lock TX. Not implemented "
            "in v3.0-alpha. See core/l2_channels.py docstring."
        )

    def instant_transfer(self, channel_id: str, sender: str,
                         amount: Decimal, signature: str) -> bool:
        raise L2NotImplementedError(
            "instant_transfer requires verified off-chain state-update "
            "signatures and monotonic nonces. The previous mock accepted "
            "the signature parameter without verification — a fund-theft "
            "vector. Not implemented in v3.0-alpha. See "
            "core/l2_channels.py docstring."
        )

    def close_channel(self, channel_id: str) -> bool:
        raise L2NotImplementedError(
            "close_channel requires an on-chain settlement TX with the "
            "latest signed state. Not implemented in v3.0-alpha. See "
            "core/l2_channels.py docstring."
        )
