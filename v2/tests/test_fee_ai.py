
import sys
import os
from decimal import Decimal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.broker import EconomicBroker
from core.compute_ledger import ComputeLedger
from core.tokenomics import Wallet

print("--- Testing Fee AI (The Broker) ---")

ledger = ComputeLedger("TEST")
wallet = Wallet()
broker = EconomicBroker(wallet, ledger, None)

# 1. Normal Load (0 pending)
fee = broker.estimate_safe_fee()
print(f"Normal Fee: {fee}")
if fee != Decimal("0.001"):
    print("❌ ERROR: Normal fee should be 0.001")
else:
    print("✅ Normal Load Verified")

# 2. High Load (500 pending)
# Mock pending transactions
ledger.pending_transactions = [1] * 500
fee = broker.estimate_safe_fee()
print(f"High Load (500 tx) Fee: {fee}")

# Logarithmic Expectation:
# log10(410) ~ 2.6. Fee ~ 0.00126
# Should be > 0.001 but MUCH less than 0.041 (Linear)
if fee > Decimal("0.001") and fee < Decimal("0.002"):
    print("✅ Dynamic Increase Verified (Logarithmic & Cheap)")
else:
    print(f"❌ ERROR: Fee out of expected logarithmic range. Got {fee}")

# 3. Extreme Load (20,000 pending) -> Should Cap
ledger.pending_transactions = [1] * 20000
fee = broker.estimate_safe_fee()
print(f"Extreme Load Fee: {fee}")
if fee == Decimal("1.0"):
    print("✅ Safety Cap Verified")
else:
    print(f"❌ ERROR: Safety Cap Failed. Got {fee}")
