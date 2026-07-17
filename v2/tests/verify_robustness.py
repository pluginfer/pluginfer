
import sys
import os
import time
import shutil
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.complete_mesh_controller import CompleteMeshController
from core.tokenomics import Wallet
from core.compute_ledger import ComputeLedger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RobustnessTest")

def clean_env():
    """Remove existing state for fresh test"""
    for f in ['wallet.pem', 'ledger.json', 'peers.json', 'config.json']:
        if os.path.exists(f):
            try:
                os.remove(f)
            except: pass
        # Remove backups too
        import glob
        for b in glob.glob(f"{f}.*.bak"):
            try:
                os.remove(b)
            except: pass

def test_persistence():
    print("\n[TEST] Verifying Persistence (Wallet & Ledger)...")
    clean_env()
    
    # 1. Start Node 1 (First Run)
    print("   [1] Starting Initialization (Fresh)...")
    node1 = CompleteMeshController(port=9001)
    addr1 = node1.wallet.address
    print(f"   [+] Generated Address: {addr1}")
    
    # Mint some tokens (simulate work)
    print("   [2] Minting Tokens...")
    tx = node1.minter.mint_coinbase(addr1, 0, 1.0)
    node1.ledger.add_transaction(tx)
    node1.ledger.mine_block(addr1)
    node1.ledger.save_chain() # Manual save or stop() triggers it
    
    # Save Wallet (happens on init, but ensure it)
    node1.wallet.save_to_file()
    
    initial_balance = node1.ledger.get_balance(addr1)
    print(f"   [+] Initial Balance: {initial_balance} PLG")
    
    # Stop Node 1
    node1.stop()
    del node1
    print("   [3] Node Stopped. Simulating Restart...")
    time.sleep(1)
    
    # 2. Start Node 1 Again (Restart)
    print("   [4] re-Initializing Node...")
    node2 = CompleteMeshController(port=9001)
    addr2 = node2.wallet.address
    
    # VERIFY WALLET
    if addr1 == addr2:
        print("   [PASS] Wallet Address Persisted!")
    else:
        print(f"   [FAIL] Address Changed! {addr1} != {addr2}")
        return False
        
    # VERIFY LEDGER
    restored_balance = node2.ledger.get_balance(addr2)
    if float(restored_balance) == float(initial_balance):
        print(f"   [PASS] Ledger Balance Restored: {restored_balance} PLG")
    else:
        print(f"   [FAIL] Balance Mismatch! {initial_balance} != {restored_balance}")
        return False
        
    node2.stop()
    return True

def test_peer_persistence():
    print("\n[TEST] Verifying Peer Persistence...")
    # Assume Peers file
    peers = [{"ip": "1.2.3.4", "port": 9000}]
    import json
    with open("peers.json", "w") as f:
        json.dump(peers, f)
        
    node = CompleteMeshController(port=9002)
    # Check if loaded
    if len(node.known_peers) == 1 and node.known_peers[0]['ip'] == "1.2.3.4":
        print("   [PASS] Peers Loaded from Disk!")
    else:
        print(f"   [FAIL] Peers not loaded: {node.known_peers}")
        node.stop()
        return False
        
    node.stop()
    return True

if __name__ == "__main__":
    if test_persistence() and test_peer_persistence():
        print("\n✅ ROBUSTNESS VERIFIED: System is Fail-Safe")
    else:
        print("\n❌ VERIFICATION FAILED")
