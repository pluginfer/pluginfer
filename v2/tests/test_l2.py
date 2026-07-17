"""W26 contract: L2 channels are an honest stub until real protocol ships.

Pre-W26 this file was an interactive demo of `StateChannelManager`. The
module is now an honest stub that raises `L2NotImplementedError` from
every public method (open_channel / instant_transfer / close_channel)
because the previous implementation silently accepted unsigned signatures
and never deducted balances. Until real payment-channel semantics ship
(open with on-chain lock, off-chain signed-state monotonic-nonce
updates, on-chain settlement with challenge window), the right contract
is to refuse rather than lie.

This test file pins that contract.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest  # noqa: E402

from core.compute_ledger import ComputeLedger  # noqa: E402
from core.l2_channels import (  # noqa: E402
    L2NotImplementedError,
    PaymentChannel,
    StateChannelManager,
)
from core.tokenomics import Wallet  # noqa: E402


def test_open_channel_is_honest_stub() -> None:
    mgr = StateChannelManager(ComputeLedger("test"))
    with pytest.raises(L2NotImplementedError):
        mgr.open_channel(Wallet(), "BOB_ADDR", Decimal("100.0"))


def test_instant_transfer_is_honest_stub() -> None:
    mgr = StateChannelManager(ComputeLedger("test"))
    with pytest.raises(L2NotImplementedError):
        mgr.instant_transfer("any_chan", "alice", Decimal("1.0"), "sig")


def test_close_channel_is_honest_stub() -> None:
    mgr = StateChannelManager(ComputeLedger("test"))
    with pytest.raises(L2NotImplementedError):
        mgr.close_channel("any_chan")


def test_payment_channel_dataclass_still_constructible() -> None:
    """The data class itself stays usable so future code can shape requests
    against a stable type, even while the manager refuses to act."""
    chan = PaymentChannel(
        channel_id="x",
        participant_a="alice",
        participant_b="bob",
        balance_a=Decimal("100"),
        balance_b=Decimal("0"),
    )
    assert chan.channel_id == "x"
    assert chan.balances["alice"] == Decimal("100")
    assert chan.is_open is True
