
import sys
import os
import time
from decimal import Decimal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.complete_mesh_controller import CompleteMeshController
from core.tokenomics import BURN_ADDRESS
from core.governance import GovernanceDAO
from core.compute_ledger import ComputeLedger

print("--- Testing Tokenomics Upgrades (Burn & DAO) ---")

# Setup
# Correct Init
controller = CompleteMeshController(host="127.0.0.1", port=40000, mode="worker") 

# Mock Dependencies for Burn Test
controller.oracle.get_task_price = lambda x: Decimal("100.0")
controller.broker.estimate_safe_fee = lambda: Decimal("0.001")

print("\n[TEST 1] Deflationary Burn (5%)...")
controller._pay_peer(controller.wallet.address, "mock_task")

# Assert Burn TX
txs = controller.ledger.pending_transactions
burn_tx = next((t for t in txs if t.type == 'burn'), None)
worker_tx = next((t for t in txs if t.type == 'task_payment'), None)
treasury_tx = next((t for t in txs if t.type == 'platform_fee'), None)

if burn_tx:
    amt = burn_tx.amount
    print(f"🔥 Burned Amount: {amt} PLG")
    if amt == Decimal("5.0"):
        print("✅ Burn Rate (5%) Verified")
    else:
        print(f"❌ Burn Rate Error. Got {amt}")
else:
    print("❌ Burn Transaction Missing")

# Verify Totals
if worker_tx: print(f"Worker Got: {worker_tx.amount} (70%)")
if treasury_tx: print(f"Treasury Got: {treasury_tx.amount} (25%)")


print("\n[TEST 2] Governance DAO...")
# Mock Ledger Balance for Voting
controller.ledger.get_balance = lambda addr: Decimal("1000.0")

dao = GovernanceDAO(controller.ledger)
pid = dao.create_proposal("Admin", "increase_fees")
print(f"Prop ID: {pid}")

# Vote Yes
dao.vote(pid, "Admin", True)
result = dao.get_result(pid)
print(f"Result: {result}")

if result == "PASSED":
    print("✅ DAO Voting Logic Verified")
else:
    print("❌ DAO Failed")
