import time
import sys
import os
import logging

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.security_manager import SecurityManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestIsolation")

# A dangerous function that might crash or leak memory
def dangerous_task(x, y):
    import os
    print(f"   [Isolated Process {os.getpid()}] Calculating {x} + {y}...")
    # Simulate work
    time.sleep(1)
    return x + y

def crash_task():
    print("   [Isolated Process] Simulating CRASH...")
    raise ValueError("Simulator Crash")

def test_process_isolation():
    print("\n" + "="*70)
    print("🛡️ TEST: Process Isolation (Sandboxing)")
    print("="*70)
    
    sec = SecurityManager()
    
    # 1. Normal Execution
    print("\n1️⃣  Running Safe Task in Isolation...")
    try:
        result = sec.run_isolated(dangerous_task, 10, 20)
        print(f"   ✅ Result: {result}")
        if result == 30:
            print("   ✅ SUCCESS: Task completed correctly.")
    except Exception as e:
        print(f"   ❌ FAILURE: Task failed: {e}")
        
    # 2. Crash Containment
    print("\n2️⃣  Running Crashing Task...")
    try:
        sec.run_isolated(crash_task)
        print("   ❌ FAILURE: Crash was NOT caught!")
    except Exception as e:
        print(f"   ✅ SUCCESS: Crash caught safely: {e}")
        print("   (Main process is still alive)")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    test_process_isolation()
