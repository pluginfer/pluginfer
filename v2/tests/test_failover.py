import time
import sys
import os
import json
import logging
import threading

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.complete_mesh_controller import CompleteMeshController

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestFailover")

def test_failover_logic():
    print("\n" + "="*70)
    print("🛡️ TEST: Task Failover & Reassignment")
    print("="*70)
    
    # 1. Start Coordinator
    coord = CompleteMeshController('127.0.0.1', 8080, 'coordinator')
    coord.start()
    print("[+] Coordinator Started on 8080")
    
    # 2. Start Worker
    worker = CompleteMeshController('127.0.0.1', 8081, 'worker')
    worker.start()
    # Force registration (since discovery takes time)
    worker.register_with_coordinator('127.0.0.1', 8080, 'TEST_KEY', {}, 1.0)
    print("[+] Worker Started on 8081")
    
    time.sleep(1)
    
    # 3. Submit Task (Long Running)
    # We use a mocked plugin simulation
    print("[*] Submitting Task...")
    task_id = coord.submit_task("txt_wordcount", {"text": "failover testing " * 1000})
    print(f"[+] Task Submitted: {task_id}")
    
    # Wait for assignment
    time.sleep(2)
    assigned_node = coord.assignments.get(task_id)
    if not assigned_node:
        print("❌ Task was NOT assigned!")
        return
    print(f"[+] Task Assigned to: {assigned_node}")
    
    # 4. Trigger Failover (Worker Pauses)
    print("[*] Simulating Worker Pause (Gaming Mode)...")
    
    # Update worker config to point to our test coordinator for the broadcast
    # (Since we're not using default 9999)
    # Actually, _broadcast_status uses config.json or 127.0.0.1:9999 fallback. 
    # We need to manually inject the status update for this test environment 
    # OR mock the config. 
    # Let's manually trigger the coordinated failover logic to simulate the network message reception
    # because overriding the hardcoded fallback inside the class is hard without file IO.
    # 
    # BUT wait, I want to verify the WHOLE flow.
    # Let's overwrite config.json temporarily? No, risky.
    # Let's just manually send the "status_update" message from the test script to the coordinator
    # pretending to be the worker.
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('127.0.0.1', 8080))
    msg = {
        'type': 'status_update',
        'node_id': worker.node_id,
        'status': 'paused'
    }
    import socket
    sock.send(json.dumps(msg).encode('utf-8'))
    sock.close()
    
    print("[*] Status Update Sent: PAUSED")
    
    time.sleep(2)
    
    # 5. Verify Failover
    # Task should be removed from assignments
    # Task should be in task_queue (re-queued)
    
    if task_id in coord.assignments:
        print("❌ FAILURE: Task is still assigned to paused worker!")
        print(f"   Assignment: {coord.assignments[task_id]}")
    else:
        print("✅ SUCCESS: Task removed from active assignments.")
        
    # Check Queue
    # We can't easily peek Queue, but we can try to get it
    try:
        task = coord.task_queue.get_nowait()
        if task.task_id == task_id:
            print("✅ SUCCESS: Task found in retry queue!")
            print(f"   Priority: {task.priority} (Should be boosted)")
        else:
            print(f"⚠️ Warning: Found different task in queue: {task.task_id}")
    except:
        print("❌ FAILURE: Task Queue is empty! Task was lost.")

    # Cleanup
    coord.stop()
    worker.stop()

import socket # Needed for the mock send

if __name__ == "__main__":
    test_failover_logic()
