"""
Marketplace fee enforcement + royalty distribution (W29 / sec6 audit)
=====================================================================

Cases:
  1. register_plugin without on-chain fee tx -> rejected.
  2. register_plugin with a real on-chain fee tx -> accepted; record
     persisted to disk and reloadable.
  3. record_execution emits a real transfer of (gross * royalty_bp/10000)
     from payer to plugin owner.
  4. Duplicate registration of the same plugin name is rejected.
  5. Fee underpaid (less than REGISTRATION_FEE_PLG) is rejected.
  6. Fee sent to wrong recipient is rejected.
  7. Royalty=0 plugins skip royalty transfers.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from decimal import Decimal
from pathlib import Path

_HERE = Path(__file__).resolve()
for parent in [_HERE.parents[1], _HERE.parents[2]]:
    if (parent / "core").is_dir():
        sys.path.insert(0, str(parent))
        break

from core.tokenomics import Wallet, Transaction, TokenMinter   # noqa: E402
from core.compute_ledger import ComputeLedger                  # noqa: E402
from core.marketplace import (                                 # noqa: E402
    IPMarketplace, MarketplaceError, REGISTRATION_FEE_PLG,
    TREASURY_ADDRESS, DEFAULT_ROYALTY_BP,
)


def _seed_balance(ledger, wallet, n_blocks=2):
    minter = TokenMinter(ledger=ledger)
    for _ in range(n_blocks):
        cb = minter.mint_coinbase(wallet.address,
                                  block_height=ledger.get_height(),
                                  difficulty_factor=1.0)
        ledger.add_transaction(cb, _internal=True)
        ledger.mine_block(wallet.address, difficulty=1)


def _pay_treasury(ledger, owner_w, amount, nonce):
    tx = Transaction(
        sender=owner_w.address, recipient=TREASURY_ADDRESS,
        amount=amount, type="transfer",
        sender_pub_key=owner_w.public_key_pem,
        fee=Decimal("0.001"), nonce=nonce,
    )
    tx.signature = owner_w.sign(tx.tx_id)
    if not ledger.add_transaction(tx):
        return None
    ledger.mine_block(owner_w.address, difficulty=1)
    return tx.tx_id


def test_register_without_fee_rejected():
    print("\n[1] REGISTER WITHOUT ON-CHAIN FEE REJECTED")
    print("-" * 60)
    ledger = ComputeLedger("m1")
    owner = Wallet()
    with tempfile.TemporaryDirectory() as td:
        mp = IPMarketplace(ledger, persist_path=str(Path(td) / "mp.json"))
        try:
            mp.register_plugin("plug-1", owner.address,
                                registration_tx_id="not-a-real-tx-id")
            assert False, "should have raised"
        except MarketplaceError as e:
            print(f"  rejected as expected: {e}")
    print("  PASS")


def test_register_with_fee_accepted_and_persisted():
    print("\n[2] REGISTER WITH ON-CHAIN FEE ACCEPTED + PERSISTED")
    print("-" * 60)
    ledger = ComputeLedger("m2")
    owner = Wallet()
    _seed_balance(ledger, owner, n_blocks=2)
    fee_tx_id = _pay_treasury(ledger, owner,
                                  REGISTRATION_FEE_PLG, nonce=0)
    assert fee_tx_id

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "mp.json"
        mp = IPMarketplace(ledger, persist_path=str(path))
        rec = mp.register_plugin("plug-2", owner.address,
                                  registration_tx_id=fee_tx_id)
        assert rec.plugin_name == "plug-2"
        assert mp.get_owner("plug-2") == owner.address
        # Persistence round-trip.
        on_disk = json.loads(path.read_text())
        assert "plug-2" in on_disk
        # Fresh marketplace loads the registry.
        mp2 = IPMarketplace(ledger, persist_path=str(path))
        assert mp2.get_owner("plug-2") == owner.address
        print(f"  registered + persisted + reloaded OK")
    print("  PASS")


def test_record_execution_pays_royalty():
    print("\n[3] EXECUTION ROYALTY EMITS REAL TRANSFER")
    print("-" * 60)
    ledger = ComputeLedger("m3")
    owner = Wallet()
    payer = Wallet()
    _seed_balance(ledger, owner, n_blocks=2)
    _seed_balance(ledger, payer, n_blocks=2)
    fee_tx_id = _pay_treasury(ledger, owner,
                                  REGISTRATION_FEE_PLG, nonce=0)
    with tempfile.TemporaryDirectory() as td:
        mp = IPMarketplace(ledger, persist_path=str(Path(td) / "mp.json"))
        mp.register_plugin("plug-3", owner.address,
                            registration_tx_id=fee_tx_id)

        gross = Decimal("100")
        pre_owner = Decimal(str(ledger.get_balance(owner.address)))
        tx_dict = mp.record_execution("plug-3", payer,
                                       gross_amount=gross, nonce=0)
        assert tx_dict
        ledger.mine_block(payer.address, difficulty=1)
        post_owner = Decimal(str(ledger.get_balance(owner.address)))
        # 5% of 100 = 5.
        assert (post_owner - pre_owner) == Decimal("5"), \
            f"royalty miscount: {post_owner - pre_owner}"
        print(f"  owner balance: {pre_owner} -> {post_owner} (+5 PLG)")
    print("  PASS")


def test_duplicate_name_rejected():
    print("\n[4] DUPLICATE PLUGIN NAME REJECTED")
    print("-" * 60)
    ledger = ComputeLedger("m4")
    owner = Wallet()
    _seed_balance(ledger, owner, n_blocks=2)
    fee1 = _pay_treasury(ledger, owner, REGISTRATION_FEE_PLG, nonce=0)
    fee2 = _pay_treasury(ledger, owner, REGISTRATION_FEE_PLG, nonce=1)

    with tempfile.TemporaryDirectory() as td:
        mp = IPMarketplace(ledger, persist_path=str(Path(td) / "mp.json"))
        mp.register_plugin("plug-4", owner.address,
                            registration_tx_id=fee1)
        try:
            mp.register_plugin("plug-4", owner.address,
                                registration_tx_id=fee2)
            assert False, "should have raised"
        except MarketplaceError as e:
            print(f"  duplicate rejected: {e}")
    print("  PASS")


def test_underpaid_fee_rejected():
    print("\n[5] UNDERPAID FEE REJECTED")
    print("-" * 60)
    ledger = ComputeLedger("m5")
    owner = Wallet()
    _seed_balance(ledger, owner, n_blocks=2)
    # Only 0.5 PLG instead of REGISTRATION_FEE_PLG (1.0).
    underpaid_id = _pay_treasury(ledger, owner,
                                     REGISTRATION_FEE_PLG / 2, nonce=0)
    with tempfile.TemporaryDirectory() as td:
        mp = IPMarketplace(ledger, persist_path=str(Path(td) / "mp.json"))
        try:
            mp.register_plugin("plug-5", owner.address,
                                registration_tx_id=underpaid_id)
            assert False, "should have raised"
        except MarketplaceError as e:
            print(f"  underpaid rejected: {e}")
    print("  PASS")


def test_fee_to_wrong_address_rejected():
    print("\n[6] FEE TO WRONG ADDRESS REJECTED")
    print("-" * 60)
    ledger = ComputeLedger("m6")
    owner = Wallet()
    decoy = Wallet()
    _seed_balance(ledger, owner, n_blocks=2)
    # Pay decoy instead of treasury.
    bad = Transaction(
        sender=owner.address, recipient=decoy.address,
        amount=REGISTRATION_FEE_PLG, type="transfer",
        sender_pub_key=owner.public_key_pem,
        fee=Decimal("0.001"), nonce=0,
    )
    bad.signature = owner.sign(bad.tx_id)
    assert ledger.add_transaction(bad)
    ledger.mine_block(owner.address, difficulty=1)

    with tempfile.TemporaryDirectory() as td:
        mp = IPMarketplace(ledger, persist_path=str(Path(td) / "mp.json"))
        try:
            mp.register_plugin("plug-6", owner.address,
                                registration_tx_id=bad.tx_id)
            assert False, "should have raised"
        except MarketplaceError as e:
            print(f"  wrong-recipient rejected: {e}")
    print("  PASS")


def test_zero_royalty_skips():
    print("\n[7] ZERO ROYALTY SKIPS TRANSFER")
    print("-" * 60)
    ledger = ComputeLedger("m7")
    owner = Wallet()
    payer = Wallet()
    _seed_balance(ledger, owner, n_blocks=2)
    _seed_balance(ledger, payer, n_blocks=2)
    fee = _pay_treasury(ledger, owner, REGISTRATION_FEE_PLG, nonce=0)
    with tempfile.TemporaryDirectory() as td:
        mp = IPMarketplace(ledger, persist_path=str(Path(td) / "mp.json"))
        mp.register_plugin("plug-7", owner.address,
                            registration_tx_id=fee, royalty_bp=0)
        result = mp.record_execution("plug-7", payer,
                                      gross_amount=Decimal("100"),
                                      nonce=0)
        assert result is None
        print("  zero-royalty plugin skipped transfer OK")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("MARKETPLACE TEST (W29)")
    print("=" * 60)
    t0 = time.time()
    test_register_without_fee_rejected()
    test_register_with_fee_accepted_and_persisted()
    test_record_execution_pays_royalty()
    test_duplicate_name_rejected()
    test_underpaid_fee_rejected()
    test_fee_to_wrong_address_rejected()
    test_zero_royalty_skips()
    print("\n" + "=" * 60)
    print(f"ALL MARKETPLACE TESTS PASSED in {time.time() - t0:.1f}s")
    print("=" * 60)
