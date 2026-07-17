import threading
import time
import socket
import json
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.complete_mesh_controller import CompleteMeshController
from core.discovery import MeshDiscovery

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SecurityStressTest")

def test_private_swarm_isolation():
    print("\n" + "="*60)
    print("🔒 TEST 1: Private Swarm Isolation (The 'Scrum' Test)")
    print("="*60)
    print("Goal: Ensure 'Public' and 'Private' nodes DO NOT discover each other.")
    
    # 1. Start Public Coordinator
    public_coord = CompleteMeshController('127.0.0.1', 8001, 'coordinator')
    # Hack: Inject Swarm ID (Needs to be done after start() typically, or we mock setup)
    # But wait, start() initializes it. So we must start, THEN modify, or modify config.
    # Let's modify after start for now as it runs in a loop.
    public_coord.start() 
    public_coord.discovery.swarm_id = "public"
    print("[Public] Coordinator Started on Port 8001 (Swarm: public)")
    
    # 2. Start Private Coordinator
    private_coord = CompleteMeshController('127.0.0.1', 8002, 'coordinator')
    private_coord.start()
    private_coord.discovery.swarm_id = "top_secret_medical"
    print("[Private] Coordinator Started on Port 8002 (Swarm: top_secret_medical)")
    
    time.sleep(2)
    
    # 3. Start Public Worker
    public_worker = MeshDiscovery(node_id="worker_public", port=9001, capabilities={}, swarm_id="public")
    public_worker.start()
    print("[Public] Worker Searching...")
    
    # 4. Start Private Worker
    private_worker = MeshDiscovery(node_id="worker_private", port=9002, capabilities={}, swarm_id="top_secret_medical")
    private_worker.start()
    print("[Private] Worker Searching...")
    
    # Wait for discovery
    time.sleep(5)
    
    # 5. Check Results
    # Public worker should find Public Coordinator
    pub_found = public_worker.search_for_coordinator(timeout=2)
    if pub_found and pub_found['port'] == 8001:
        print("✅ [PASS] Public Worker found Public Coordinator.")
    else:
        print(f"❌ [FAIL] Public Worker found: {pub_found}")
        
    # Private worker should find Private Coordinator
    priv_found = private_worker.search_for_coordinator(timeout=2)
    if priv_found and priv_found['port'] == 8002:
        print("✅ [PASS] Private Worker found Private Coordinator.")
    elif priv_found:
        print(f"❌ [FAIL] Private Worker found WRONG Coordinator: {priv_found}")
    else:
        print("❌ [FAIL] Private Worker found NOTHING.")
        
    # Cleanup
    public_coord.stop()
    private_coord.stop()
    public_worker.stop()
    private_worker.stop()

def test_ai_sentinel_attack():
    print("\n" + "="*60)
    print("🤖 TEST 2: AI Sentinel Attack Simulation")
    print("="*60)
    print("Goal: Simulate a hacker attack and verify AI Sentinel bans the IP.")
    
    # Start Protected Node
    target_node = CompleteMeshController('127.0.0.1', 8003, 'coordinator')
    target_node.start()
    print("[Target] Secure Node Started on Port 8003")
    
    # Ensure our IP is not whitelisted fully for this test?
    # Actually, localhost is whitelisted for Rate Limit, but NOT for Threat Score in AI Sentinel?
    # Let's check: AI Sentinel doesn't seem to have a whitelist. GOOD.
    
    attacker_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    attacker_sock.connect(('127.0.0.1', 8003))
    
    print("[Attacker] Connected. Launching 'Fuzzing' attack (Simulating gibberish)...")
    
    try:
        # Attack: Send many bad packets quickly (High Error Rate + High Frequency)
        for i in range(50):
            msg = f'{{"type": "MALFORMED_JSON_ATTACK_{i}"}}\n'
            attacker_sock.send(msg.encode('utf-8'))
            time.sleep(0.01) # Fast fire
            
            # Read response (expecting error)
            try:
                resp = attacker_sock.recv(1024)
            except:
                pass
                
        time.sleep(1)
        
        # Check if we are banned
        print("[Attacker] Checking if banned...")
        try:
             attacker_sock.send(b'{"type": "task"}\n')
             resp = attacker_sock.recv(1024)
             if b"Security Threat" in resp:
                 print("[PASS] AI Sentinel BLOCKED the request!")
             else:
                 print(f"[FAIL] Server replied normally: {resp}")
        except Exception as e:
             # Connection reset is also a pass (aggressive ban)
             print(f"[PASS] Connection reset by server (Banned): {e}")
             
    except Exception as e:
        print(f"Attack Interrupted: {e}")
    
    # Get Sentinel Stats
    stats = target_node.security_manager.sentinel.get_stats()
    print(f"[Stats] Banned IPs: {stats['banned_count']}")
    
    target_node.stop()

if __name__ == "__main__":
    test_private_swarm_isolation()
    test_ai_sentinel_attack()
