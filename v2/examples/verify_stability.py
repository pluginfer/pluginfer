"""
Stability & Failover Verification
Simulates network instability to test robustness.
"""
import sys
import os
import time
import socket
import threading

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from core.complete_mesh_controller import CompleteMeshController

def test_heartbeat_timeout():
    print("\n[TEST] Heartbeat & Failover")
    
    # 1. Start Coordinator
    coord = CompleteMeshController('127.0.0.1', 8899, 'coordinator')
    
    # Inject a "Fake" Node manually into the peers list
    fake_node_id = "worker_ghost"
    coord.peers[fake_node_id] = {
        'ip': '1.2.3.4', 
        'port': 1234, 
        'last_seen': time.time() # Alive now
    }
    
    # Assign a fake task to it
    coord.assignments['task_critical_1'] = fake_node_id
    
    print("  ✓ Coordinator started with 1 fake worker")
    
    # Start the monitoring loop (mocked start)
    monitor_thread = threading.Thread(target=coord._monitor_health)
    monitor_thread.daemon = True
    coord.running = True
    monitor_thread.start()
    
    # 2. Wait for timeout (Mocking time passage or waiting real time)
    print("  ✓ Waiting for node timeout (15s)...")
    # We will fast-forward verify by manually checking logic or waiting
    # Let's wait 16 seconds to be sure
    time.sleep(16)
    
    # 3. Verify Failover
    assert fake_node_id not in coord.peers, "Dead node should be removed"
    print("  ✓ Dead node removed from peer list")
    
    assert 'task_critical_1' not in coord.assignments, "Task assignment should be revoked"
    print("  ✓ Failed task detected and unassigned (ready for retry)")
    
    coord.stop()

if __name__ == "__main__":
    print("="*60)
    print("STABILITY VERIFICATION")
    print("="*60)
    
    try:
        test_heartbeat_timeout()
        print("\n✅ STABILITY TESTS PASSED")
    except AssertionError as e:
        print(f"\n❌ FAIL: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
