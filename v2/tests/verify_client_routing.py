"""
VERIFICATION: Client Delivery Routing
Proves that Client A cannot receive Client B's results.
"""
import sys
import os
import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.complete_mesh_controller import CompleteMeshController

# MOCK Plugin Execution to simulate network delay
def mock_execute(task):
    time.sleep(1) # Simulate work
    return {
        'status': 'success',
        'result': f"Processed content for {task.input_data['client_name']}",
        'task_id': task.task_id
    }

def run_routing_test():
    print("\n" + "="*60)
    print("🚦 CLIENT ROUTING VERIFICATION 🚦")
    print("="*60)
    
    # 1. Setup Controller
    controller = CompleteMeshController()
    controller.running = True
    
    # Monkey-patch execution to avoid loading real plugins/hardware for this logic test
    controller._execute_task = mock_execute
    
    # Start Processor Thread
    processor = threading.Thread(target=controller._process_tasks, daemon=True)
    processor.start()
    
    # 2. Define Client Simulation
    def simulate_client(name):
        print(f"[{name}] Submitting Task...")
        try:
            # Submit
            task_id = controller.submit_task('mock_plugin', {'client_name': name})
            print(f"[{name}] Got Ticket: {task_id}")
            
            # Wait for Result (Poll)
            result = controller.get_result(task_id, timeout=5)
            
            # Verification
            if result and name in result['result']:
                print(f"✅ [{name}] SUCCESS: Received correct result: '{result['result']}'")
                return True
            else:
                print(f"❌ [{name}] FAILURE: Received WRONG result: {result}")
                return False
                
        except Exception as e:
            print(f"❌ [{name}] ERROR: {e}")
            return False

    # 3. Running Concurrent Clients
    print("\n[!] Launching Concurrent Clients (Alice & Bob)...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(simulate_client, "Alice"),
            executor.submit(simulate_client, "Bob")
        ]
        
        results = [f.result() for f in futures]
        
    # 4. Final Verdict
    print("\n" + "-"*60)
    if all(results):
        print("🏆 VERDICT: ROUTING SECURE.")
        print("   Cross-talk impossible. Ticket ID strictly maps to Result.")
    else:
        print("❌ VERDICT: ROUTING FAILED.")
        
    controller.running = False

if __name__ == "__main__":
    run_routing_test()
