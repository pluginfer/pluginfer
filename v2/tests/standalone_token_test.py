
import unittest
import sys
import os
import importlib.util
from decimal import Decimal

# Helper to load module from path
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load core modules directly
try:
    tokenomics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core', 'tokenomics.py'))
    ledger_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core', 'compute_ledger.py'))
    
    # Load tokenomics first
    tokenomics = load_module('tokenomics', tokenomics_path)
    
    # Context for Ledger (needs tokenomics in sys.modules or mocked)
    # compute_ledger.py does "from .tokenomics import Transaction, Wallet"
    # This relative import will fail if loaded as standalone.
    # We must patch it or load it carefully.
    # Actually, let's just make a modified copy of compute_ledger in memory? 
    # Or simpler: Just read the file content and exec it?
    pass
except Exception as e:
    print(f"Failed to load modules: {e}")

# Since relative imports inside compute_ledger.py will fail in this hacky standalone mode,
# we might need to rely on the fact that we put 'tokenomics' in sys.modules.
# But 'from .tokenomics' expects a package.

# ALTERNATIVE STRATEGY: 
# Just run unittest but rename core/__init__.py temporarily.
# That is risky.

# Let's try to mock the package structure in the test setup.
pass

class TestTokenomicsStandalone(unittest.TestCase):
    def setUp(self):
        # We need to manually initialize the classes if imports fail
        # This is getting complicated.
        pass

if __name__ == '__main__':
    # Let's try the simplest hack: 
    # 1. Read tokenomics.py
    # 2. Read compute_ledger.py
    # 3. Replace "from .tokenomics" with nothing (assuming we exec in same namespace)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core'))
    
    with open(os.path.join(base_dir, 'tokenomics.py'), 'r', encoding='utf-8') as f:
        tokenomics_code = f.read()
        
    with open(os.path.join(base_dir, 'compute_ledger.py'), 'r', encoding='utf-8') as f:
        ledger_code = f.read().replace('from .tokenomics import Transaction, Wallet', '')
        
    # Execute in a shared namespace
    global_env = globals().copy()
    
    # Run Tokenomics
    exec(tokenomics_code, global_env)
    
    # Run Ledger (now it has access to Wallet/Transaction from previous exec)
    exec(ledger_code, global_env)
    
    # Now run tests using the classes injected into global_env
    # We can define the test class inside the main block or use the one above references via global_env
    
    print("\n[INFO] Modules loaded into memory. Running tests...")
    
    Wallet = global_env['Wallet']
    Transaction = global_env['Transaction']
    TokenMinter = global_env['TokenMinter']
    ComputeLedger = global_env['ComputeLedger']
    
    # --- TEST CASES ---
    
    # 1. Wallet
    w = Wallet()
    print(f"Address: {w.address}")
    assert w.address.startswith("PLG")
    
    # 2. Minter
    minter = TokenMinter()
    assert minter.get_block_reward(0) == Decimal("50.0")
    
    # 3. Transaction & Ledger
    ledger = ComputeLedger("test_node")
    
    # Mint
    tx_mint = minter.mint_coinbase(w.address, 1)
    
    # Transfer
    tx_transfer = Transaction(
        sender=w.address,
        recipient="PLGRECIPIENT",
        amount=5,
        type='transfer',
        sender_pub_key=w.public_key_pem
    )
    tx_transfer.signature = w.sign(tx_transfer.tx_id)
    
    # Add to Ledger
    ledger.pending_transactions = [tx_mint, tx_transfer]
    block = ledger.mine_block("miner")
    
    # Verify
    assert ledger.verify_chain() == True
    print("✅ Chain Verification Passed")
    
    # Tamper
    block.transactions[1]['amount'] = "999999"
    assert ledger.verify_chain() == False
    print("✅ Tamper Detection Passed")
    
    print("\nSUCCESS: All Tokenomics Logic Verified!")
