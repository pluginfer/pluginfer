"""Pluginfer revenue-share regression test (CP-1 rewrite).

Pre-W29 this file called `marketplace.register_plugin("mock_task",
"DEV_WALLET", 0.0)` directly. W29 made `register_plugin` require a
real on-chain fee transfer to TREASURY_ADDRESS. The fee enforcement
itself is covered in `tests/test_marketplace.py`; this file isolates
the 70 / 20 / 5 / 5 revenue split logic by side-loading the plugin
into the marketplace registry for the duration of the test.

Asserts:
  - `controller._pay_peer(peer_node_id, plugin_name)` emits 4 typed
    transactions: task_payment, platform_fee, burn, royalty
  - The amounts split a 100-PLG total as 70 / 20 / 5 / 5

Skips when the host can't construct a CompleteMeshController (no
torch / port-bind issues / etc.); the test isolates the math, not
the controller-startup pathway.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest  # noqa: E402

try:
    from core.complete_mesh_controller import CompleteMeshController
    from core.marketplace import PluginRecord
    from core.tokenomics import Wallet
    _HAS_CONTROLLER = True
except Exception:  # pragma: no cover - skip on env mismatch
    _HAS_CONTROLLER = False


@pytest.mark.skipif(
    not _HAS_CONTROLLER,
    reason="CompleteMeshController unavailable in this env",
)
def test_pay_peer_emits_70_20_5_5_split() -> None:
    # Use a high port to avoid colliding with anything running on the host.
    controller = CompleteMeshController(
        host="127.0.0.1", port=29999, mode="worker"
    )
    try:
        # Inject a peer with a real wallet address registered as payable.
        peer_wallet = Wallet()
        controller.nodes["peer_node_X"] = {
            "node_id": "peer_node_X",
            "wallet_address": peer_wallet.address,
            "payable": True,
        }
        # Side-load the plugin into the marketplace registry. (The W29
        # fee-enforcement path is covered separately in
        # tests/test_marketplace.py; here we isolate the split math.)
        plugin_owner_wallet = Wallet()
        controller.marketplace.registry["mock_task"] = PluginRecord(
            plugin_name="mock_task",
            owner_address=plugin_owner_wallet.address,
            registration_tx_id="test-side-loaded",
            registered_at=0.0,
        )
        # Pin the price so the math is hand-checkable
        controller.oracle.get_task_price = lambda _plugin: Decimal("100.0")
        controller.broker.estimate_safe_fee = lambda: Decimal("0.001")

        ok = controller._pay_peer("peer_node_X", "mock_task")
        assert ok is not False, "_pay_peer refused to pay"

        pending = controller.ledger.pending_transactions
        by_type = {t.type: t for t in pending}
        for required in ("task_payment", "platform_fee", "burn", "royalty"):
            assert required in by_type, (
                f"missing tx type {required!r}; got {sorted(by_type)}"
            )
        assert Decimal(str(by_type["task_payment"].amount)) == Decimal("70.00")
        assert Decimal(str(by_type["platform_fee"].amount)) == Decimal("20.00")
        assert Decimal(str(by_type["burn"].amount)) == Decimal("5.00")
        assert Decimal(str(by_type["royalty"].amount)) == Decimal("5.00")
    finally:
        # Best-effort cleanup of any background threads the controller spawns.
        for attr in ("scout",):
            obj = getattr(controller, attr, None)
            stop = getattr(obj, "stop", None) or getattr(obj, "close", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass
