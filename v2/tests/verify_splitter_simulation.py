"""
Verify "Hive Mind" Workflow (Split-Mesh-Join)
Simulates the entire Map-Reduce lifecycle for a large video task.
"""
import sys
import os
import time
import threading
import logging

# Ensure project root is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.complete_mesh_controller import CompleteMeshController

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger('HiveSimulation')

def run_simulation():
    print("\n" + "="*50)
    print("HIVE MIND SIMULATION (Split - Distribute - Join)")
    print("="*50)

    # 1. Start Coordinator
    print("\n[1] Starting Coordinator (Home Base)...")
    coordinator = CompleteMeshController(host='127.0.0.1', port=7000, mode='coordinator')
    coordinator.start()
    time.sleep(2) # Wait for startup

    # 2. Start Worker (The Hive)
    print("\n[2] Starting Worker (The Hive Drone)...")
    worker = CompleteMeshController(host='127.0.0.1', port=7001, mode='worker')
    worker.local_score = 50.0 # Simulate Powerful Worker
    
    # Manually register for speed
    coordinator.nodes[worker.node_id] = {
        'ip': '127.0.0.1', 'port': 7001, 
        'status': 'online', 'type': 'worker',
        'perf_score': 50.0
    }
    
    worker.start()
    time.sleep(1)

    try:
        # 3. SPLIT PHASE (Map)
        print("\n[3] MAP PHASE: Splitting 'BigMovie.mp4'...")
        # In a real app, this runs locally. We mock it by calling the plugin directly or via submit
        splitter_task = coordinator.distribute_batch(
            plugin_name='video_splitter',
            batch=[{'filename': 'BigMovie.mp4', 'duration': 100, 'chunks': 5}],
            strategy='load_balanced'
        )
        # Wait for result
        split_result = coordinator.get_result(splitter_task[0], timeout=5)
        
        if not split_result or 'segments' not in split_result.get('result', {}):
             print("❌ SPLIT FAILED")
             return

        segments = split_result['result']['segments']
        print(f"    Generated {len(segments)} chunks (Expected ~50 for Privacy Shredding).")
        
        # Verify Privacy Obfuscation
        if len(segments) >= 50:
             print("    ✅ PRIVACY CHECK: High fragmentation (Shredder Active)")
        else:
             print("    ⚠️ PRIVACY WARNING: Chunks too large!")

        # 4. DISTRIBUTE PHASE (Process)
        print("\n[4] HIVE PHASE: Distributing chunks to the Mesh...")
        # Create a batch of tasks, one for each segment
        # We use 'txt_upper' as a dummy for 'video_process' since we didn't write a video_process plugin
        process_batch = []
        for seg in segments:
            process_batch.append({
                'text': f"Processing {seg['file_part']}", 
                'index': seg['index']
            })
            
        task_ids = coordinator.distribute_batch(
             plugin_name='txt_upper', # Dummy processor
             batch=process_batch,
             strategy='smart' # Use our new smart routing!
        )
        
        print(f"    Dispatched {len(task_ids)} tasks to the Hive.")
        
        # Wait for all results
        processed_segments = []
        for tid in task_ids:
            res = coordinator.get_result(tid, timeout=10)
            if res:
                print(f"    ✅ Task {tid[-4:]} completed by {res.get('node_id')[-4:]}")
                # Reconstruct the 'segment' object for the joiner
                # Map task_id back to index (since txt_upper doesn't return index)
                original_index = task_ids.index(tid)
                
                processed_segments.append({
                    'index': original_index,
                    'data': res['result'].get('text')
                })
        
        # 5. JOIN PHASE (Reduce)
        print("\n[5] REDUCE PHASE: Reassembling video...")
        join_task = coordinator.distribute_batch(
            plugin_name='video_joiner',
            batch=[{'segments': processed_segments, 'original_filename': 'BigMovie.mp4'}],
            strategy='load_balanced'
        )
        
        final_result = coordinator.get_result(join_task[0], timeout=5)
        
        if final_result and final_result.get('status') == 'success':
            print("\n" + "="*30)
            print("✅ ORCHESTRATION SUCCESS")
            print("="*30)
            print(f"Final output: {final_result['result']['final_file']}")
            
            # Verify Ledger
            print(f"Ledger Height: {len(coordinator.ledger.chain)}")
        else:
            print("\n❌ JOIN FAILED")
            print(final_result)

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
