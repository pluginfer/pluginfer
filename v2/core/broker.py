
"""
The Broker (Economic Agent)
Autonomously manages wallet assets to maximize yield.
- Auto-Stakes idle tokens when APY is attractive.
- Keeps a 'Liquid Reserve' for operational costs (fees).
"""
import time
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)

class EconomicBroker:
    def __init__(self, wallet, ledger, staking_contract):
        self.wallet = wallet
        self.ledger = ledger
        self.staking = staking_contract
        self.min_liquid_reserve = Decimal("10.0") # Keep 10 PLG liquid
        self.target_apy_threshold = Decimal("0.04") # Stake if APY > 4%
        
    def estimate_safe_fee(self) -> Decimal:
        """
        [PHASE 5] Dynamic Fee AI
        Calculates the optimal transaction fee based on network congestion.
        Ensures fees are never too high (Robustness).
        """
        try:
            pending_count = len(self.ledger.pending_transactions)
            
            # Base Fee: Very cheap (0.001 PLG)
            base_fee = Decimal("0.001")
            
            # Congestion Multiplier (Logarithmic Scaling for Cheap Fees at Scale)
            if pending_count > 100:
                import math
                # Use Log10 so fees grow very slowly.
                # e.g., 10,000 pending -> log10 is 4. Fee adds 0.0004. Cheap!
                # This solves the "Ethereum High Gas" problem.
                congestion_factor = Decimal(str(math.log10(pending_count - 90)))
                congestion_cost = congestion_factor * Decimal("0.0001")
                optimal_fee = base_fee + congestion_cost
            else:
                optimal_fee = base_fee
                
            # SAFETY CAP: Never exceed 1.0 PLG (User Protection)
            if optimal_fee > Decimal("1.0"):
                logger.warning(f"[BROKER] Fee Suppression Active! Measured {optimal_fee}, Capping at 1.0")
                return Decimal("1.0")
                
            return optimal_fee
            
        except Exception:
            return Decimal("0.001") # Fail-safe default

    def evaluate_strategy(self):
        """
        [GOLD STANDARD] Autonomous Decision Making
        Called periodically to optimize portfolio.
        """
        try:
            # 1. Check Balance
            balance = Decimal(str(self.ledger.get_balance(self.wallet.address)))
            
            # 2. Check Market Conditions — pull live APY from the staking
            # contract (which derives from chain state). Fallback to the
            # contract's BASE_APY constant if the staking module is
            # unavailable (e.g. during isolated unit tests).
            try:
                from .staking import BASE_APY
                current_apy = BASE_APY
                # If the contract exposes a dynamic APY hook later, prefer it.
                dynamic_apy = getattr(self.staking, "current_apy", None)
                if callable(dynamic_apy):
                    current_apy = Decimal(str(dynamic_apy()))
            except Exception:
                current_apy = Decimal("0.05")
            
            if current_apy >= self.target_apy_threshold:
                # 3. Calculate Investable Amount
                investable = balance - self.min_liquid_reserve
                
                if investable > Decimal("5.0"): # Minimum stake size
                    logger.info(f"[BROKER] Opportunity Detected! APY {current_apy:.1%} > {self.target_apy_threshold:.1%}")
                    
                    # 4. Execute Strategy (Auto-Stake)
                    self._execute_stake(investable, current_apy=current_apy)

        except Exception as e:
            logger.error(f"[BROKER] Strategy evaluation failed: {e}")

    def _execute_stake(self, amount: Decimal,
                       current_apy: Decimal = Decimal("0.05")):
        """Execute the staking transaction"""
        try:
            # We need to access the controller's stake method or manually build TX.
            # Ideally the broker constructs the TX and asks the Wallet to sign.
            
            from .tokenomics import Transaction

            # Pay the dynamic fee or the new MIN_TX_FEE will reject this
            # at add_transaction time. estimate_safe_fee floors at 0.001
            # so this is always sufficient.
            tx = Transaction(
                sender=self.wallet.address,
                recipient="PLG_STAKING_POOL_V1",
                amount=amount,
                type='transfer',
                sender_pub_key=self.wallet.public_key_pem,
                fee=self.estimate_safe_fee(),
            )
            # Sign over the tx_id (which now incorporates the fee).
            tx.signature = self.wallet.sign(tx.tx_id)
            
            # Add to Ledger
            if self.ledger.add_transaction(tx):
                # Immediate mine (for demo)
                new_block = self.ledger.mine_block(self.wallet.address)
                if new_block:
                    self.ledger.save_chain()
                    # We should broadcast, but we don't have direct access to gossip here easily
                    # unless we pass it in. For now, we rely on the controller or assumes ledger handles it (it doesn't).
                    # Improvement: Broker should return an Action Plan to the Controller, 
                    # rather than executing side-effects deep in logic.
                    logger.info(
                        "[BROKER] AUTO-STAKED %s PLG (APY %.1f%%)",
                        amount, float(current_apy) * 100.0,
                    )
                    return True
                    
        except Exception as e:
            logger.error(f"[BROKER] Failed to execute stake: {e}")
            return False
