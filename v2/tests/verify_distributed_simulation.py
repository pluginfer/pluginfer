"""
Verify Distributed Execution Simulation

This script simulates a distributed network on localhost by starting:
1. A Coordinator Node (Port 8890)
2. A Worker Node (Port 8891)

It verifies:
- Worker registration
- Task submission to Coordinator
- Task offloading to Worker
- Result retrieval
"""
import sys
import os
import time
import threading
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.complete_mesh_controller import CompleteMeshController

# Configure logging
logging.basicConfig(level=logging.WARN, format='%(name)s - %(message)s')
logger = logging.getLogger("Simulation")
logger.setLevel(logging.INFO)

def run_simulation():
    print("="*60)
    print("DISTRIBUTED MESH SIMULATION")
    print("="*60)

    # 1. Start Coordinator
    print("\n[1] Starting Coordinator (Port 8890)...")
    coordinator = CompleteMeshController('127.0.0.1', 8890, mode='coordinator')
    coordinator.start()
    print(f"    Coordinator online: {coordinator.node_id}")

    # 2. Start Worker
    print("\n[2] Starting Worker (Port 8891)...")
    worker = CompleteMeshController('127.0.0.1', 8891, mode='worker')
    worker.start()
    print(f"    Worker online: {worker.node_id}")

    try:
        # 3. Register Worker with Coordinator
        print("\n[3] Registering Worker with Coordinator...")
        success = worker.register_with_coordinator(
            coordinator_host='127.0.0.1', 
            coordinator_port=8890,
            license_key='beta',
            hardware_info={'name': 'SimulateGPU', 'type': 'gpu'},
            perf_score=50.0
        )
        
        if success:
            print("    ✅ Registration Successful")
        else:
            print("    ❌ Registration Failed")
            return

        # Allow time for registration to propagate
        time.sleep(1)

        # 4. Submit Task to Coordinator
        print("\n[4] Submitting Task to Coordinator...")
        print("\n[4] submitting Task to Coordinator (via distribute_batch)...")
        # distribute_batch returns a list of task_ids
        task_ids = coordinator.distribute_batch(
            plugin_name='txt_upper',
            batch=[{'text': 'Smart Routing Test', 'priority': 10}],
            strategy='smart'
        )
        task_id = task_ids[0]
        print(f"    Task ID: {task_id}")

        # 5. Wait for Execution (Coordinator should send to Worker)
        print("\n[5] Waiting for distributed execution...")
        result = None
        for i in range(10):
            result = coordinator.get_result(task_id, timeout=1)
            if result:
                break
            print(f"    Waiting... ({i+1}/10)")
            time.sleep(0.5)

        # 6. Verify Results
        print("\n[6] VERIFICATION RESULTS")
        print("-" * 30)
        
        if result:
            print(f"    Status: {result.get('status')}")
            print(f"    Result: {result.get('result')}")
            
            executor_id = result.get('node_id')
            print(f"    Coordinator ID: {coordinator.node_id}")
            print(f"    Worker ID:      {worker.node_id}")
            print(f"    Executed By:    {executor_id}")
            
            if executor_id == worker.node_id:
                print("\n    ✅ SUCCESS: Task was offloaded to Worker Node!")
            elif executor_id == coordinator.node_id:
                print("\n    ⚠ WARNING: Task was executed locally (Load balancing fallback?)")
            else:
                print(f"\n    ❌ ERROR: Unknown executor {executor_id}")
                
            # ✅ VERIFY LEDGER
            print("\n[7] Verifying Compute Ledger (Blockchain)...")
            is_valid = coordinator.ledger.verify_chain()
            block_count = len(coordinator.ledger.chain)
            print(f"    Chain Valid: {is_valid}")
            print(f"    Block Count: {block_count} (Genesis + Results)")
            
            if is_valid and block_count >= 2:
                 print("    ✅ BLOCKCHAIN INTEGRITY CONFIRMED")
            else:
                 print("    ❌ BLOCKCHAIN FAILURE")

        else:
            print("\n    ❌ TIMEOUT: No result received.")

    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n[8] Cleaning up...")
        coordinator.stop()
        worker.stop()
        print("    Simulation Complete.")

if __name__ == "__main__":
    run_simulation()
