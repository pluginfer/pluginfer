"""
Verify Self-Healing (Phase 4)
Simulates a "Chaos Monkey" scenario where a worker intermittently fails.
Checks if 'orchestrate_batch' successfully retries and completes the job.
"""
import sys
import os
import time
import threading
import logging
import random

# Ensure project root is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.complete_mesh_controller import CompleteMeshController

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger('SelfHealingSim')

class ChaosWorker(CompleteMeshController):
    """A worker that effectively flips a coin to decide if it works or dies."""
    def _execute_task(self, task):
        # 50% chance of failure (simulated by doing nothing/timeout)
        if random.random() < 0.5:
            logger.warning(f"💥 CHAOS MONKEY: Dropping task {task.task_id} (Simulated Crash)")
            # Return nothing, effectively timing out the coordinator
            time.sleep(10) 
            return {'status': 'error', 'message': 'Simulated Crash'} 
        
        logger.info(f"✅ Chaos Worker: Processing {task.task_id} normally.")
        return super()._execute_task(task)

def run_simulation():
    print("\n" + "="*50)
    print("SELF-HEALING SIMULATION (Phase 4)")
    print("="*50)

    # 1. Start Coordinator
    print("\n[1] Starting Supervisor (Coordinator)...")
    coordinator = CompleteMeshController(host='127.0.0.1', port=8000, mode='coordinator')
    coordinator.start()
    
    # 2. Start Chaos Worker
    print("\n[2] Starting Chaos Worker (50% Failure Rate)...")
    worker = ChaosWorker(host='127.0.0.1', port=8001, mode='worker')
    worker.local_score = 100.0 # High score to attract tasks
    worker.start()
    
    # Register manually
    coordinator.nodes[worker.node_id] = {
        'ip': '127.0.0.1', 'port': 8001, 
        'status': 'online', 'type': 'worker',
        'perf_score': 100.0
    }

    try:
        # 3. Submit Batch via Supervisor
        batch_size = 5
        print(f"\n[3] Submitting {batch_size} tasks to Supervisor...")
        
        batch = []
        for i in range(batch_size):
            batch.append({
                'text': f"Important Task {i}", 
                'operation': 'uppercase',
                'priority': 5
            })
            
        # Call orchestrate_batch
        final_result = coordinator.orchestrate_batch(
            plugin_name='txt_upper',
            batch=batch,
            strategy='smart',
            max_retries=5 # Give it enough tries to overcome 50% failure
        )
        
        print("\n" + "="*30)
        print("FINAL REPORT")
        print("="*30)
        print(f"Status: {final_result['status']}")
        print(f"Success Count: {final_result['success_count']}/{batch_size}")
        print(f"Failed Count:  {final_result['failed_count']}")
        
        if final_result['status'] == 'success':
            print("\n✅ SELF-HEALING SUCCESS: All tasks completed despite Chaos.")
        else:
             print("\n❌ SELF-HEALING FAILED: Some tasks were lost.")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n[6] Shutdown...")
        coordinator.stop()
        worker.stop()

if __name__ == "__main__":
    run_simulation()
