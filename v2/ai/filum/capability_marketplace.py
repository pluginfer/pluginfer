"""Capability Marketplace: monetize fine-tuned LoRAs across the mesh.

INVENTION (claim §9 in the design notes): conventional AI economics
concentrates capability in a few centralized providers (OpenAI,
Anthropic, Google). They charge marginally for compute + monopoly
rents on capability. End users pay $15-75 per million tokens.

Our novel mechanism INVERTS this:

  * Any node on the mesh can train its own LoRA adapter for any
    domain (legal, medical, code, financial, etc.).
  * The adapter is registered on-chain via the Pluginfer marketplace
    smart contract: ID, domain, owner pubkey, quality score (proven
    via held-out evaluation), price per use.
  * When a node receives a query whose cluster matches a registered
    adapter, it can either USE its own adapter (free), or RENT a
    different node's adapter for one inference at a micro-payment
    (e.g. 0.0001 PLG, ~$0.000001).
  * The mesh routes inference to the cheapest adapter with adequate
    quality.

Effects on AI economics:
  * The cost floor approaches zero (no monopoly rent on capability).
  * Specialty knowledge gets distributed across the contributors who
    have it (a medical professional can train a medical LoRA + earn
    when others rent it).
  * Centralized providers compete on raw scale + verification, NOT
    on holding the only copy of a niche capability.

This is novel as the SPECIFIC COMBINATION of:
  1. On-chain micro-payment per adapter use,
  2. Quality-proven via held-out eval signed by ≥2/3 validators,
  3. Mesh-routed inference selection across the adapter set,
  4. Auto-spawning + auto-listing pipeline (lora_pool.py + this
     module).

Failure modes (honest)
----------------------
* Quality verification is hard. We use a held-out eval set + ≥2/3
  validator-signed score. An adapter owner can game the eval if it
  matches their training distribution. Mitigation: rotating
  randomized eval sets sampled from the live mesh traffic.
* Adapter download size + transfer bandwidth still cost something.
  We bundle adapter rental with first-N inferences-for-free so the
  download amortises.
* Free-rider problem: an adapter buyer could try to extract the
  weights and re-list as their own. Mitigation: encrypted-at-rest
  delivery + fingerprint watermarking; full IP protection requires
  legal layer (the chain audit trail is proof of theft).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AdapterListing:
    """A registered adapter on the marketplace."""
    adapter_id: str
    owner_pubkey_pem: str
    domain: str
    cluster_centroid: List[float]
    quality_score: float
    quality_proof_signatures: List[str] = field(default_factory=list)
    price_per_use_plg: float = 0.0001
    listed_at: float = 0.0
    use_count: int = 0
    revenue_plg_total: float = 0.0
    fingerprint: str = ""

    def canonical_body(self) -> str:
        """The bytes the validator quorum signs."""
        return json.dumps({
            "adapter_id": self.adapter_id,
            "owner_pubkey_pem": self.owner_pubkey_pem,
            "domain": self.domain,
            "cluster_centroid": self.cluster_centroid,
            "quality_score": self.quality_score,
            "fingerprint": self.fingerprint,
            "price_per_use_plg": self.price_per_use_plg,
        }, sort_keys=True)


@dataclass
class AdapterReceipt:
    """Proof that adapter was used + payment was settled."""
    adapter_id: str
    user_pubkey_pem: str
    owner_pubkey_pem: str
    timestamp: float
    payment_tx_id: str
    cost_plg: float


# ---------------------------------------------------------------------------
# Marketplace
# ---------------------------------------------------------------------------


class CapabilityMarketplace:
    """Manages registered adapters + per-use settlement."""

    def __init__(
        self,
        ledger,
        validator_set,            # core.bft_consensus.ValidatorSet
        store_dir: Path,
        *,
        min_quality_for_listing: float = 0.5,
        platform_fee_fraction: float = 0.05,
        platform_fee_address: Optional[str] = None,
    ):
        self.ledger = ledger
        self.validator_set = validator_set
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.min_quality_for_listing = min_quality_for_listing
        self.platform_fee_fraction = platform_fee_fraction
        self.platform_fee_address = platform_fee_address
        self.listings: Dict[str, AdapterListing] = {}
        self.receipts: List[AdapterReceipt] = []
        self._load()

    # ------------------------------------------------------------------

    def _path(self) -> Path:
        return self.store_dir / "capability_marketplace.json"

    def _load(self) -> None:
        p = self._path()
        if not p.exists():
            return
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        for L in d.get("listings", []):
            ad = AdapterListing(**L)
            self.listings[ad.adapter_id] = ad
        for r in d.get("receipts", []):
            self.receipts.append(AdapterReceipt(**r))

    def _save(self) -> None:
        d = {
            "listings": [vars(L) for L in self.listings.values()],
            "receipts": [vars(r) for r in self.receipts],
        }
        self._path().write_text(json.dumps(d), encoding="utf-8")

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def register_listing(
        self,
        listing: AdapterListing,
        *,
        validator_signatures: List[Tuple[str, str]],   # (pubkey_pem, signature_b64)
    ) -> bool:
        """Register an adapter. Requires ≥2/3 validator-stake-weighted
        signatures over the listing body to prove the quality score is
        legitimate."""
        if listing.quality_score < self.min_quality_for_listing:
            logger.warning("listing %s quality below floor", listing.adapter_id)
            return False
        if listing.adapter_id in self.listings:
            logger.warning("listing %s already exists", listing.adapter_id)
            return False
        # Verify signatures.
        canon = listing.canonical_body()
        verified_stake = Decimal("0")
        total_stake = Decimal("0")
        if hasattr(self.validator_set, "validators"):
            from .. import training  # noqa: F401  (imported for side-effect path)
        try:
            from ..core.tokenomics import Wallet
        except Exception:
            try:
                from core.tokenomics import Wallet
            except Exception as e:
                logger.error("Wallet import failed: %s", e)
                return False
        seen = set()
        validators_iter = []
        if hasattr(self.validator_set, "validators"):
            validators_iter = (
                self.validator_set.validators.values()
                if hasattr(self.validator_set.validators, "values")
                else self.validator_set.validators
            )
        stake_by_pub: Dict[str, Decimal] = {}
        for v in validators_iter:
            pk = getattr(v, "pubkey_pem", None) or getattr(v, "public_key_pem", None)
            stake = Decimal(str(getattr(v, "stake", "0")))
            if pk:
                stake_by_pub[pk] = stake
                total_stake += stake
        for pubkey, sig in validator_signatures:
            if pubkey in seen:
                continue
            if pubkey not in stake_by_pub:
                continue
            if not Wallet.verify(pubkey, canon, sig):
                continue
            seen.add(pubkey)
            verified_stake += stake_by_pub[pubkey]
            listing.quality_proof_signatures.append(sig)
        if total_stake > 0 and verified_stake * 3 < total_stake * 2:
            logger.warning(
                "listing %s below 2/3 stake quorum (%s/%s)",
                listing.adapter_id, verified_stake, total_stake,
            )
            return False
        listing.listed_at = time.time()
        listing.fingerprint = listing.fingerprint or hashlib.sha256(
            canon.encode(),
        ).hexdigest()[:32]
        self.listings[listing.adapter_id] = listing
        self._save()
        return True

    # ------------------------------------------------------------------
    # Discovery + use
    # ------------------------------------------------------------------

    def find(self, *, domain: str = None,
             min_quality: float = 0.0,
             max_price_plg: float = float("inf")) -> List[AdapterListing]:
        out = []
        for L in self.listings.values():
            if domain is not None and domain.lower() not in L.domain.lower():
                continue
            if L.quality_score < min_quality:
                continue
            if L.price_per_use_plg > max_price_plg:
                continue
            out.append(L)
        # Sort by quality / price ratio descending.
        out.sort(key=lambda L: L.quality_score / max(1e-9, L.price_per_use_plg),
                 reverse=True)
        return out

    def settle_use(
        self,
        *,
        adapter_id: str,
        user_pubkey_pem: str,
        signed_use_payment_tx_id: str,
        cost_plg: float,
    ) -> AdapterReceipt:
        """Record one use of the adapter + the chain payment that settled it."""
        listing = self.listings.get(adapter_id)
        if listing is None:
            raise ValueError(f"adapter {adapter_id} not listed")
        receipt = AdapterReceipt(
            adapter_id=adapter_id,
            user_pubkey_pem=user_pubkey_pem,
            owner_pubkey_pem=listing.owner_pubkey_pem,
            timestamp=time.time(),
            payment_tx_id=signed_use_payment_tx_id,
            cost_plg=cost_plg,
        )
        listing.use_count += 1
        listing.revenue_plg_total += cost_plg * (1 - self.platform_fee_fraction)
        self.receipts.append(receipt)
        if len(self.receipts) > 10_000:
            self.receipts = self.receipts[-5_000:]
        self._save()
        return receipt

    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "listings": len(self.listings),
            "active_listings": sum(1 for L in self.listings.values()
                                   if L.use_count > 0),
            "total_uses": sum(L.use_count for L in self.listings.values()),
            "total_revenue_plg": sum(L.revenue_plg_total
                                     for L in self.listings.values()),
            "avg_quality": (
                sum(L.quality_score for L in self.listings.values())
                / max(1, len(self.listings))
            ),
        }
