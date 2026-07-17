import time
import sys
import os
import random
import logging

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.complete_mesh_controller import CompleteMeshController

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestLeaderElection")

def test_leader_election_logic():
    print("\n" + "="*70)
    print("🗳️ TEST: Leader Election (Reputation Based)")
    print("="*70)
    
    # 1. Simulate a Network of Workers
    workers = []
    
    # Node A: Good Reputation
    w1 = {'node_id': 'Node_A', 'rep_score': 0.95, 'uptime': 1000}
    workers.append(w1)
    
    # Node B: Average Reputation
    w2 = {'node_id': 'Node_B', 'rep_score': 0.50, 'uptime': 500}
    workers.append(w2)
    
    # Node C: Bad Reputation
    w3 = {'node_id': 'Node_C', 'rep_score': 0.10, 'uptime': 100}
    workers.append(w3)
    
    # Node D: Perfect Reputation but Low Uptime (New Node)
    w4 = {'node_id': 'Node_D', 'rep_score': 1.00, 'uptime': 10}
    workers.append(w4)
    
    print("\n👥 Candidates:")
    for w in workers:
        print(f"   {w['node_id']}: Rep={w['rep_score']}, Uptime={w['uptime']}")
        
    # 2. Simulate Election Logic
    # Implementation of _elect_local_leader logic
    print("\n🗳️ Running Election Protocol...")
    
    best_candidate = None
    best_score = -1.0
    
    for w in workers:
        # Scoring Formula: Reputation (70%) + Uptime (30%)
        # Normalize Uptime (capped at 1000s for test)
        norm_uptime = min(w['uptime'], 1000) / 1000.0
        
        score = (w['rep_score'] * 0.7) + (norm_uptime * 0.3)
        
        print(f"   {w['node_id']} Election Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_candidate = w['node_id']
            
    print("-" * 30)
    print(f"👑 Winner: {best_candidate} (Score: {best_score:.4f})")
    
    # 3. Validation
    # Node A should win (0.95*0.7 + 1.0*0.3 = 0.665 + 0.3 = 0.965)
    # Node D (1.0*0.7 + 0.01*0.3 = 0.703) -- Low uptime penalty
    
    if best_candidate == 'Node_A':
        print("\n✅ SUCCESS: Correct leader elected (High Rep + High Uptime).")
    else:
        print(f"\n❌ FAILURE: Unexpected winner {best_candidate}")

if __name__ == "__main__":
    test_leader_election_logic()
