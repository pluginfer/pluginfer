import time
import sys
import os
import json
import logging
import threading

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.complete_mesh_controller import CompleteMeshController
from core.discovery import MeshDiscovery

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestComplexJob")

def test_complex_job_execution():
    print("\n" + "="*70)
    print("🎬 TEST: Complex Job (MapReduce) - Video Processing")
    print("="*70)
    
    # 1. Start Coordinator
    coord = CompleteMeshController('127.0.0.1', 9999, 'coordinator')
    coord.start()
    print("[+] Coordinator Started")
    
    # 2. Start Workers (2 Nodes to prove distribution)
    workers = []
    for i in range(2):
        w = CompleteMeshController('127.0.0.1', 10001+i, 'worker')
        w.start()
        workers.append(w)
        print(f"[+] Worker {i} Started (ID: {w.node_id})")
        
    time.sleep(5) # Allow discovery
    
    # 3. Submit Complex Job
    print("[*] Submitting Composite Job (Video Split -> Sentiment -> Join)...")
    
    # We use 'txt_sentiment' as the processor for chunks just to prove flow, 
    # as we don't have real video files.
    # The 'video_splitter' plugin in mock mode generates dummy chunks.
    
    input_data = {
        "filename": "test_movie.mp4",
        "duration": 120, # Mock 120s video
        "process_plugin": "txt_sentiment" # Process each chunk with this
    }
    
    job_id = coord.submit_composite_job(
        split_plugin="video_splitter",
        join_plugin="video_joiner",
        input_data=input_data
    )
    
    print(f"[+] Job Submitted: {job_id}")
    
    # 4. Monitor Progress
    for _ in range(30):
        status = coord.get_job_status(job_id)
        if not status:
            print("[-] Job not found!")
            break
            
        print(f"   Status: {status['status']} | Progress: {status.get('progress',0):.1f}% | Stage: {status.get('stage','?')}")
        
        if status['status'] == 'Completed':
            print("✅ JOB COMPLETED!")
            print("Final Result:", status.get('final_result'))
            break
        
        if status['status'] == 'Failed':
            print(f"❌ JOB FAILED: {status.get('error')}")
            break
            
        time.sleep(1)
        
    # 5. Verify Token Rewards
    # Check that workers got paid
    print("\n💰 Verifying Rewards...")
    total_minted = 0
    for w in workers:
        bal = coord.ledger.get_balance(w.node_id)
        print(f"   Worker {w.node_id} Balance: {bal}")
        total_minted += bal
        
    if total_minted > 0:
        print("✅ Tokens Successfully Minted!")
    else:
        print("❌ No Tokens Minted (Check Ledger Logic)")

    # Cleanup
    coord.stop()
    for w in workers: w.stop()

if __name__ == "__main__":
    test_complex_job_execution()
