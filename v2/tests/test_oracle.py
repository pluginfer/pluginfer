
import sys
import os
from decimal import Decimal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.oracle import PricingOracle

print("--- Testing Pricing Oracle ---")

oracle = PricingOracle()

# 1. Check Base Price
# Should now default to "Compute Peg" ($0.05) since Market Mock is disabled
price = oracle.get_price()
print(f"Current PLG Price: ${price}")

if price == Decimal("0.05"):
    print("✅ Compute Peg Fallback Verified ($0.05)")
else:
    print(f"❌ Oracle Failed. Expected $0.05, Got {price}")

# 2. Check Conversion
# $1.00 should be 20.0 PLG (at $0.05)
tokens = oracle.usd_to_plg(1.00)
print(f"$1.00 = {tokens} PLG")

expected = Decimal("1.00") / price
if abs(tokens - expected) < Decimal("0.001"):
    print("✅ Conversion Logic Verified")
else:
    print(f"❌ Conversion Failed. Expected {expected}, Got {tokens}")

# 3. Check Task Pricing
# Face Swap = $0.05
# Cost in PLG = 0.5 PLG (at $0.10)
cost = oracle.get_task_price("face_swap")
print(f"Face Swap Cost: {cost} PLG")

if cost > 0:
    print("✅ Task Pricing Verified")
else:
    print("❌ Task Pricing Failed")
