"""Compliance + economics regression tests.

Pre-W30 this file asserted hardcoded "Gold Standard Listing
Requirements" certification text in the compliance report. W30
stripped those fabricated claims and replaced them with measured
facts + an `attestation_mode` field. This test file pins the post-W30
schema.

Pre-W4 / W11 this file also tested transaction fees against a stale
`tx.calculate_hash` API; the new API computes the hash via
`calculate_tx_hash` with explicit fee + nonce inputs (W11). We use
`Transaction(..., fee=...)` directly here.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest  # noqa: E402

from core.auditor import SystemAuditor  # noqa: E402
from core.compute_ledger import ComputeLedger  # noqa: E402
from core.privacy import ZKPrivacy  # noqa: E402
from core.tokenomics import TokenMinter, Transaction, Wallet  # noqa: E402


# ---------------------------------------------------------------------------
# Transaction fees still collected and paid out as fee_reward
# ---------------------------------------------------------------------------

def test_fee_is_collected_into_fee_reward_tx() -> None:
    ledger = ComputeLedger("TEST_FEE_NODE")
    minter = TokenMinter(ledger=ledger)
    alice = Wallet()
    # Mint a coinbase to alice so she has balance to spend.
    coinbase_tx = minter.mint_coinbase(alice.address, block_height=1)
    coinbase_tx.signature = ""  # system tx
    ledger.add_transaction(coinbase_tx, _internal=True)
    block = ledger.mine_block(alice.address)
    assert block is not None

    tx = Transaction(
        sender=alice.address,
        recipient="BOB",
        amount=Decimal("10.0"),
        type="transfer",
        fee=Decimal("1.5"),
        sender_pub_key=alice.public_key_pem,
    )
    tx.signature = alice.sign(tx.tx_id)
    assert ledger.add_transaction(tx)

    block = ledger.mine_block("miner_001")
    fee_txs = [t for t in block.transactions if t["type"] == "fee_reward"]
    assert fee_txs, "no fee_reward tx was emitted"
    assert Decimal(str(fee_txs[0]["amount"])) == Decimal("1.5")


# ---------------------------------------------------------------------------
# W30 compliance report schema (no fabricated certification)
# ---------------------------------------------------------------------------

def test_compliance_report_has_post_w30_schema() -> None:
    ledger = ComputeLedger("TEST_AUDIT_NODE")
    auditor = SystemAuditor(ledger, core_path="./core")
    report_str = auditor.generate_compliance_report()
    data = json.loads(report_str) if isinstance(report_str, str) else report_str

    # Top-level post-W30 fields
    assert "attestation_mode" in data, (
        "post-W30 report must declare attestation_mode so consumers know "
        "what level of integrity to assume"
    )
    assert "audit_result" in data
    assert "disclaimers" in data and isinstance(data["disclaimers"], list)
    assert "measured_metrics" in data

    # Measured-metrics block holds the actual facts
    metrics = data["measured_metrics"]
    for key in (
        "blockchain_height",
        "file_system_integrity",
        "files_monitored",
        "issues_count",
    ):
        assert key in metrics, f"missing measured_metric: {key}"

    # Forbidden pre-W30 fabricated fields
    rendered = json.dumps(data).lower()
    assert "gold standard" not in rendered
    rendered_no_commas = rendered.replace(",", "")
    assert "21000000 plg" not in rendered_no_commas, (
        "explicit max_supply claim was removed in W30 because the "
        "auditor cannot verify supply without the chain it audits"
    )


# ---------------------------------------------------------------------------
# ZK privacy round-trip (Pedersen + Schnorr; W4)
# ---------------------------------------------------------------------------

def test_zk_commitment_round_trip_and_rejection() -> None:
    zk = ZKPrivacy()
    value = 100
    # API: create_commitment returns (commitment_hex, blinding).
    commitment, blinding = zk.create_commitment(value)
    assert zk.verify_commitment(commitment, value, blinding)
    # Wrong value or wrong blinding rejected
    assert not zk.verify_commitment(commitment, 999, blinding)
    _other_c, blinding_other = zk.create_commitment(value)
    assert not zk.verify_commitment(commitment, value, blinding_other)
