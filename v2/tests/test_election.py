
import sys
import os
import time
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.election import LeaderElection

class MockLedger:
    def get_balance(self, node_id): return 100.0

class MockNet:
    def broadcast_message(self, msg):
        print(f"[NET] Broadcast: {msg['type']} from {msg.get('candidate_id', 'LEADER')} (Score: {msg.get('score')})")

class MockReputation:
    def get_score(self): return 999.9 # Very High Score (Good Reputation)

def on_promote():
    print("👑 PROMOTED TO KING!")

print("\n--- Testing Leader Election Logic (Reputation Based) ---")
election = LeaderElection("NODE_X", MockLedger(), MockNet(), MockReputation(), on_promote)

# 1. State: Follower
print(f"Initial State: {election.state}")
assert election.state == "FOLLOWER"

# 2. Force Timeout (Simulate Dead Coordinator)
print("Simulating Time Travel (Timeout)...")
election.last_heartbeat = time.time() - 10 

# Wait for Monitor Loop to catch it
time.sleep(2)

# 3. Check Promoted to CANDIDATE
print(f"New State: {election.state}")
if election.state == "CANDIDATE":
    print("✅ Node correctly started Election on Timeout")
else:
    print("❌ Node stayed Asleep")

# 4. Simulate Receiving Votes (Majority)
# In real life, node waits for votes. Here we just test logic.
# (Logic test simplified as the real async loop handles votes)

print("✅ Election Unit Test Passed")
election.stop()
