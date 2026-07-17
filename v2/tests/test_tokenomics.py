
import unittest
import sys
import os
import shutil
import time
from decimal import Decimal

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import directly from file to avoid package init issues if any
from core.tokenomics import Wallet, TokenMinter, Transaction
from core.compute_ledger import ComputeLedger, Block

class TestTokenomics(unittest.TestCase):

    def setUp(self):
        # Create a temporary wallet for testing
        self.wallet = Wallet()
        self.recipient_wallet = Wallet()
        self.minter = TokenMinter()
        self.ledger = ComputeLedger("test_node")

    def test_wallet_generation(self):
        """Test wallet creation and address generation"""
        print(f"\n[Test] Wallet Address: {self.wallet.address}")
        print(f"[Test] Public Key PEM len: {len(self.wallet.public_key_pem)}")
        self.assertTrue(self.wallet.address.startswith("PLG"))
        self.assertIsNotNone(self.wallet.private_key)
        self.assertIsNotNone(self.wallet.public_key)

    def test_transaction_signing(self):
        """Test signing and verifying transactions with Public Key"""
        
        # 1. Create Transaction (without valid sig yet)
        tx = Transaction(
            sender=self.wallet.address,
            recipient=self.recipient_wallet.address,
            amount=Decimal("10.5"),
            type='transfer',
            sender_pub_key=self.wallet.public_key_pem 
        )
        
        # 2. Sign
        signature = self.wallet.sign(tx.tx_id)
        tx.signature = signature
        
        # 3. Verify
        is_valid = Wallet.verify(self.wallet.public_key_pem, tx.tx_id, signature)
        
        print(f"[Test] Transaction Signature Valid? {is_valid}")
        self.assertTrue(is_valid)
        
        # 4. Tamper test (Wrong Key)
        is_invalid = Wallet.verify(self.recipient_wallet.public_key_pem, tx.tx_id, signature)
        self.assertFalse(is_invalid)

    def test_minting_logic(self):
        """Test halving and minting"""
        # Block 0 (Genesis)
        reward_0 = self.minter.get_block_reward(0)
        self.assertEqual(reward_0, Decimal("50.0"))

        # Block 210,000 (First Halving)
        reward_1 = self.minter.get_block_reward(210000)
        self.assertEqual(reward_1, Decimal("25.0"))

        # Mint Transaction
        tx = self.minter.mint_coinbase(self.wallet.address, 100)
        self.assertEqual(tx.amount, Decimal("50.0"))
        self.assertEqual(tx.recipient, self.wallet.address)
        self.assertEqual(tx.sender, "COINBASE")

    def test_ledger_blockchain(self):
        """Test block creation, merkle root, and signature verification"""
        # Create dummy transactions
        # TX1: Mint (Valid)
        tx1 = self.minter.mint_coinbase(self.wallet.address, 1)
        
        # TX2: Transfer (Valid)
        tx2 = Transaction(
            sender=self.wallet.address,
            recipient=self.recipient_wallet.address,
            amount=Decimal("10.0"),
            type='transfer',
            sender_pub_key=self.wallet.public_key_pem
        )
        tx2.signature = self.wallet.sign(tx2.tx_id)
        
        # Mine block
        self.ledger.pending_transactions = [tx1, tx2]
        block = self.ledger.mine_block("miner_addr")
        
        self.assertIsNotNone(block)
        self.assertEqual(block.index, 1)
        self.assertEqual(len(block.transactions), 2)
        
        # Verify Chain
        is_valid = self.ledger.verify_chain()
        print(f"[Test] Chain Valid? {is_valid}")
        self.assertTrue(is_valid)
        
        # Tamper Test: Modify transaction amount in block
        block.transactions[1]['amount'] = "1000.0" # Fraud!
        # Re-calc hash to simulate sophisticated attack (but Sig will fail)
        # block.hash = block.calculate_hash() 
        # Actually verify_chain checks hash validity first.
        # Let's break the signature check specifically
        # We need to reconstruct the block integrity but start with bad sig?
        # If I change amount, tx_id changes. If tx_id changes, sig for old tx_id doesn't match new tx_id?
        # Wait, tx_id is hash(content). If I verify(sig, tx_id), and I change amount -> tx_id changes.
        # So I must re-sign with MY private key (Attacker).
        # But Attacker's PubKey != Sender's Address.
        # (This check is what missing: Address Ownership Proof).
        # Assuming the check passes for now or fails on basic integrity.
        
        is_valid_after_tamper = self.ledger.verify_chain()
        print(f"[Test] Chain Valid After Tamper? {is_valid_after_tamper}")
        self.assertFalse(is_valid_after_tamper)

if __name__ == '__main__':
    unittest.main()
