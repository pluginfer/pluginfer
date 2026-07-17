
"""
Pricing Oracle
Decouples Task Cost from Token Volatility.
- Fetches PLG/USD Price (Mocked for now)
- Converts Task Value (USD) -> Token Amount (PLG)
"""
import time
import logging
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)

class PricingOracle:
    """
    Hybrid pricing oracle:
        1. Market price (CEX/DEX feed, when ledger is live).
        2. Compute peg (always available; derived from network state).

    The compute peg is the *floor*: PLG cannot sustainably trade below
    the cost of producing it, because workers can mine instead of buying.
    Buyers above the peg = real demand.

    Compute peg formula:
        peg_usd = (avg_kwh_per_block * spot_kwh_usd) / block_reward_plg

    All inputs are observable on-chain or from local sensors. No magic
    constants. Set `external_market_fn` to a callable that returns a
    market price; oracle prefers it when available.
    """

    DEFAULT_KWH_PER_BLOCK = Decimal("0.001")          # ≈ 1Wh of work for a 4-zero PoW
    DEFAULT_KWH_USD = Decimal("0.12")                  # global average residential rate
    DEFAULT_BLOCK_REWARD = Decimal("50.0")             # PLG (matches TokenMinter genesis)

    def __init__(self,
                 kwh_per_block: Optional[Decimal] = None,
                 spot_kwh_usd: Optional[Decimal] = None,
                 block_reward_plg: Optional[Decimal] = None,
                 external_market_fn=None):
        self.kwh_per_block = kwh_per_block or self.DEFAULT_KWH_PER_BLOCK
        self.spot_kwh_usd = spot_kwh_usd or self.DEFAULT_KWH_USD
        self.block_reward_plg = block_reward_plg or self.DEFAULT_BLOCK_REWARD
        self.external_market_fn = external_market_fn
        self.last_update = time.time()

    def get_price(self) -> Decimal:
        # 1. external market feed if configured
        if self.external_market_fn is not None:
            try:
                p = self.external_market_fn()
                if p and Decimal(p) > 0:
                    return Decimal(p)
            except Exception as e:
                logger.debug("market feed failed: %s", e)
        # 2. compute peg (always available)
        return self._get_compute_peg_price()

    def _get_compute_peg_price(self) -> Decimal:
        if self.block_reward_plg <= 0:
            return Decimal("0")
        return (self.kwh_per_block * self.spot_kwh_usd) / self.block_reward_plg

    def update_compute_peg_inputs(self, *,
                                  kwh_per_block: Optional[Decimal] = None,
                                  spot_kwh_usd: Optional[Decimal] = None,
                                  block_reward_plg: Optional[Decimal] = None) -> None:
        """Called by the validator set when difficulty / halving / energy mix changes."""
        if kwh_per_block is not None:
            self.kwh_per_block = Decimal(str(kwh_per_block))
        if spot_kwh_usd is not None:
            self.spot_kwh_usd = Decimal(str(spot_kwh_usd))
        if block_reward_plg is not None:
            self.block_reward_plg = Decimal(str(block_reward_plg))
        self.last_update = time.time()
        
    def usd_to_plg(self, usd_amount: float) -> Decimal:
        """Convert USD Value to PLG Tokens"""
        price = self.get_price()
        if price == 0: return Decimal("0")
        
        token_amount = Decimal(str(usd_amount)) / price
        return token_amount.quantize(Decimal("0.000001"))

    def get_task_price(self, task_type: str) -> Decimal:
        """Get price of a task in PLG"""
        # Base Prices in USD
        usd_prices = {
            "img_resize": 0.01,
            "img_grayscale": 0.01,
            "face_swap": 0.05,
            "txt_sentiment": 0.005,
            "pdf_compress": 0.02,
            "dynamic_executor": 0.10 # Expensive
        }
        
        cost_usd = usd_prices.get(task_type, 0.01)
        cost_plg = self.usd_to_plg(cost_usd)

        # `self.current_price_usd` did not exist on the class — that was an
        # AttributeError on every call. Use the live oracle price instead.
        rate = self.get_price()
        logger.info(
            "[ORACLE] Task %s ($%s) = %s PLG (Rate: $%.4f)",
            task_type, cost_usd, cost_plg, float(rate),
        )
        return cost_plg
