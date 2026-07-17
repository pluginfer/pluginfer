
"""
Staking Manager (DeFi Layer)
Manages token locking, APY rewards, and liquidity pools.
Integrated directly with ComputeLedger.
"""
import time
import logging
from decimal import Decimal
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

STAKING_POOL_ADDRESS = "PLG_STAKING_POOL_V1"
BASE_APY = Decimal("0.05") # 5% Base APY
BONUS_APY_PER_MONTH = Decimal("0.01") # +1% per month locked

class StakingContract:
    """
    Stake state derived from chain.

    Previous version had `update_state_from_block` defined but never
    called from any code path, so the stake table was always empty.
    Now the contract self-syncs on every read by replaying any blocks
    we haven't seen since the last sync. State is therefore always
    consistent with chain truth — there is no separate cached source.
    """

    def __init__(self, ledger):
        self.ledger = ledger
        self.stakes: Dict[str, Dict] = {}     # address -> stake_info
        self._last_synced_height = -1
        self._sync_lock = __import__("threading").Lock()

    def _sync_from_chain(self) -> None:
        """Replay all blocks past `_last_synced_height` into stake state."""
        with self._sync_lock:
            current_height = self.ledger.get_height() if self.ledger else 0
            if current_height <= self._last_synced_height + 1:
                return
            chain = getattr(self.ledger, "chain", [])
            for idx in range(self._last_synced_height + 1, len(chain)):
                self.update_state_from_block(chain[idx])
            self._last_synced_height = current_height - 1

    def get_staking_info(self, address: str) -> Optional[Dict]:
        self._sync_from_chain()
        return self.stakes.get(address)

    def validator_stake_snapshot(self) -> Dict[str, Decimal]:
        """Return {validator_pubkey_pem: Decimal stake} for the W32
        slash-evidence quorum check. The pubkey-PEM key is what
        attestations sign with, so the snapshot is keyed accordingly.

        We materialise it from `self.stakes` (address -> info) by
        looking up the validator's pubkey on each entry. Entries that
        have already been slashed are excluded — slashed validators
        shouldn't count toward the 2/3 quorum that's deciding whether
        to slash the next equivocator.
        """
        self._sync_from_chain()
        snapshot: Dict[str, Decimal] = {}
        for addr, info in self.stakes.items():
            if info.get("slashed"):
                continue
            pubkey_pem = info.get("validator_pubkey_pem")
            if not pubkey_pem:
                continue
            try:
                amt = Decimal(str(info.get("amount", 0)))
            except Exception:
                continue
            if amt > 0:
                snapshot[pubkey_pem] = amt
        return snapshot
        
    def calculate_reward(self, address: str) -> Decimal:
        """Calculate pending rewards based on APY and time elapsed"""
        self._sync_from_chain()
        info = self.stakes.get(address)
        if not info: return Decimal("0.0")
        
        principal = Decimal(info['amount'])
        start_time = info['timestamp']
        duration_sec = time.time() - start_time
        
        # APY Logic
        years_elapsed = Decimal(duration_sec) / Decimal(31536000) # 365 days
        
        # Dynamic APY based on lock (Simulated here, realistically tracked)
        apy = BASE_APY 
        
        # Interest = Principal * APY * Time
        reward = principal * apy * years_elapsed
        return reward
        
    def stake(self, user_address: str, amount: Decimal) -> Dict:
        """
        Create a Staking Transaction: User -> Pool
        """
        # 1. Check Balance
        balance = self.ledger.get_balance(user_address)
        if balance < float(amount):
             raise ValueError("Insufficient funds to stake")
             
        # 2. Create Transaction (User must sign this externally, but for internal logic we prep it)
        # Note: In a real system, the WALLET signs this. 
        # Here we return the TX intent for the Controller to execute/sign.
        
        # For simplicity in this native module, we assume we just track state
        # IF the transaction is confirmed in the ledger.
        
        return {
            "type": "stake",
            "recipient": STAKING_POOL_ADDRESS,
            "amount": str(amount),
            "timestamp": time.time()
        }
        
    def unstake(self, user_address: str) -> Dict[str, Any]:
        """
        Claim Rewards + Principal
        """
        info = self.stakes.get(user_address)
        if not info: raise ValueError("No active stake")
        
        reward = self.calculate_reward(user_address)
        principal = Decimal(info['amount'])
        total_payout = principal + reward
        
        return {
            "type": "unstake",
            "sender": STAKING_POOL_ADDRESS, # System sends back
            "recipient": user_address,
            "amount": str(total_payout),
            "principal": str(principal),
            "reward": str(reward)
        }

    # Listen to Ledger Events (This should be called when block is added)
    def update_state_from_block(self, block):
        """Parse block for stake/unstake TXs and update local state"""
        for tx in block.transactions:
            tx_type = tx.get('type') # We might need a 'subtype' or check recipient
            sender = tx.get('sender')
            recipient = tx.get('recipient')
            amount = tx.get('amount')
            
            # STAKE: User -> Pool
            if recipient == STAKING_POOL_ADDRESS:
                self.stakes[sender] = {
                    'amount': amount,
                    'timestamp': tx.get('timestamp'),
                    'last_block': block.index
                }
                logger.info(f"[STAKING] New Stake: {sender} deposited {amount}")
                
            # UNSTAKE: Pool -> User (if typed specifically)
            # Or we look for a special 'unstake' type if we added that to Transaction class
