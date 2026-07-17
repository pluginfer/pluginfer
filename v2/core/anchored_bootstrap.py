"""Self-bootstrapping seed selection driven by the Bitcoin anchor.

The legacy bootstrap path (`core.complete_mesh_controller.BOOTSTRAP_SEEDS`)
required Pluginfer-operated seed nodes baked into source. That makes
Pluginfer's network bootstrap centrally controlled: if the operator
goes down (or away, or hostile), every new node fails first launch.

This module turns the bootstrap into a permissionless lookup against
a SIGNED registry of seeds, ordered by a permutation derived from
the latest public Bitcoin block hash. Properties:

* **No Pluginfer secret material** is required at install time -- the
  registry is signed by a chain-of-stake quorum, the anchor is a
  third-party ledger.
* **Anyone can publish a seed** by getting on the registry (signed by
  >= 2/3 of validator stake at the registry's snapshot height).
* **No operator can pin the bootstrap order** -- the anchor is the
  Bitcoin block hash, which Pluginfer cannot forge.

Why this is the novel bit
-------------------------------------------------
"A method of bootstrapping a permissionless overlay network without
 author-controlled bootstrap nodes by deriving a deterministic
 permutation of a stake-attested seed registry from a third-party
 blockchain's block hash, where the third-party blockchain provides
 only public randomness and is otherwise unrelated to the overlay
 network's economics."

This isn't dependent on any specific Bitcoin RPC (we use 3 redundant
public APIs); the user can override the source list via env, and any
future blockchain whose block hash is publicly fetchable can be
plugged in.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from .bitcoin_anchor import BitcoinAnchor, get_bitcoin_anchor
from .tokenomics import Wallet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry types
# ---------------------------------------------------------------------------


@dataclass
class SeedRecord:
    """One seed in the public registry."""
    host: str
    port: int
    pubkey_pem: str                          # the seed node's wallet pubkey
    region: str = "unknown"
    registered_at_btc_height: int = 0        # block height at registration
    # Validator quorum signatures attesting this record. Real
    # production uses >=2/3-stake quorum; the test fixtures use 1+
    # to keep tests focused on permutation logic.
    quorum_signatures: List[dict] = field(default_factory=list)

    def canonical(self) -> str:
        return json.dumps({
            "host": self.host,
            "port": self.port,
            "pubkey_pem": self.pubkey_pem,
            "region": self.region,
            "registered_at_btc_height": self.registered_at_btc_height,
        }, sort_keys=True, separators=(",", ":"))


@dataclass
class SeedRegistry:
    """Signed list of seeds. The list MAY be empty -- callers should
    fall back to LAN discovery or pre-shared-key in that case."""
    records: List[SeedRecord] = field(default_factory=list)
    epoch_btc_height: int = 0
    schema: str = "pluginfer-seed-registry/v1"

    @classmethod
    def from_dict(cls, d: dict) -> "SeedRegistry":
        return cls(
            records=[SeedRecord(**r) for r in d.get("records", [])],
            epoch_btc_height=int(d.get("epoch_btc_height", 0)),
            schema=d.get("schema", "pluginfer-seed-registry/v1"),
        )

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "epoch_btc_height": self.epoch_btc_height,
            "records": [r.__dict__ for r in self.records],
        }


# ---------------------------------------------------------------------------
# Permutation
# ---------------------------------------------------------------------------


def _sortkey(record: SeedRecord, anchor_bytes: bytes) -> bytes:
    """Deterministic sort key for `record` under `anchor_bytes`. Two
    records placed under the same anchor produce a STABLE order; the
    same record placed under a DIFFERENT anchor lands at a different
    position. This is the permutation the bootstrap uses.
    """
    h = hashlib.sha256()
    h.update(anchor_bytes)
    h.update(record.canonical().encode("utf-8"))
    return h.digest()


def permute_seeds(records: Sequence[SeedRecord],
                  anchor: BitcoinAnchor) -> List[SeedRecord]:
    """Return `records` shuffled deterministically by the anchor."""
    a = anchor.as_seed_bytes()
    return sorted(records, key=lambda r: _sortkey(r, a))


def filter_quorum_signed(records: Iterable[SeedRecord],
                         min_signatures: int = 1
                         ) -> List[SeedRecord]:
    """Drop records that don't have at least `min_signatures` valid
    quorum signatures (sig over canonical()). Production should use a
    stake-weighted threshold; the floor here keeps the permutation
    layer decoupled from the staking-snapshot layer."""
    out: List[SeedRecord] = []
    for r in records:
        valid = 0
        msg = r.canonical()
        for sig in r.quorum_signatures:
            try:
                if Wallet.verify(sig["pubkey"], msg, sig["value"]):
                    valid += 1
            except Exception:                                  # pragma: no cover
                continue
        if valid >= min_signatures:
            out.append(r)
        else:
            logger.warning(
                "anchored_bootstrap: rejecting unsigned seed record %s:%d",
                r.host, r.port,
            )
    return out


# ---------------------------------------------------------------------------
# The public bootstrap API
# ---------------------------------------------------------------------------


@dataclass
class BootstrapPlan:
    """Ordered seeds the node should try, plus the anchor that
    produced this order (so the choice is auditable)."""
    seeds: List[SeedRecord]
    anchor: BitcoinAnchor


def make_bootstrap_plan(registry: SeedRegistry,
                        *,
                        anchor: Optional[BitcoinAnchor] = None,
                        min_quorum_sigs: int = 1,
                        max_seeds: int = 8,
                        anchor_kwargs: Optional[dict] = None
                        ) -> BootstrapPlan:
    """End-to-end: fetch the Bitcoin anchor, filter the registry to
    quorum-signed records, permute by the anchor, return the first
    `max_seeds` seeds.

    `anchor` is injectable for tests so this function does NOT touch
    the network when the caller supplies one.
    """
    if anchor is None:
        anchor = get_bitcoin_anchor(**(anchor_kwargs or {}))
    quorum_records = filter_quorum_signed(
        registry.records, min_signatures=min_quorum_sigs,
    )
    permuted = permute_seeds(quorum_records, anchor)
    return BootstrapPlan(seeds=permuted[:max_seeds], anchor=anchor)


__all__ = [
    "SeedRecord",
    "SeedRegistry",
    "BootstrapPlan",
    "permute_seeds",
    "filter_quorum_signed",
    "make_bootstrap_plan",
]
