
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.smart_contracts import SmartContractVM

print("--- Testing VM Stability (Infinite Loop) ---")

vm = SmartContractVM(None, storage_file="test_contracts.json")

# Malicious Code
bad_code = """
def freeze():
    while True:
        pass
"""

addr = vm.deploy_contract("HACKER", bad_code)

try:
    print("Attempting to run infinite loop (Expect Timeout in 5s)...")
    start = time.time()
    vm.execute(addr, "freeze")
    print("❌ FAILED: VM did not timeout!")
except TimeoutError:
    duration = time.time() - start
    print(f"✅ SUCCESS: VM caught the loop in {duration:.2f}s")
except Exception as e:
    print(f"FAILED with wrong error: {e}")
