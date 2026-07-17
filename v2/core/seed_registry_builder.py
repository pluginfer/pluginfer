"""G1 — self-signing seed registry bootstrap.

A brand-new Pluginfer deployment can't sign its first seed record with
a 2/3-validator quorum (there are no validators yet). Without a real
registry, every fresh node fails on first launch — the chicken-and-egg
deathblow for decentralised networks.

This module ships the **TOFU (trust-on-first-use)** path:

  1. The first node generates its own wallet, picks a host + port, and
     constructs a `SeedRecord` self-signed by that wallet — the
     registry has *one* signature, tagged
     `bootstrap_mode="single-validator-tofu"`.

  2. As more nodes join, each spins up its own seed record and joins
     the registry. When ≥3 distinct wallets have signed any given
     record, the registry is **promoted** to a normal 2/3-quorum
     registry and the TOFU flag clears.

The TOFU flag is auditable + on-chain (broadcast as part of every
mesh-handshake) so clients pessimistically prefer multi-signer
records over single-signer ones. This means an attacker who spins
up a TOFU seed only successfully serves bootstrap traffic until the
first quorum-signed alternative arrives — after which they're never
selected again.

The 2/3-quorum semantics, the Bitcoin-anchored permutation, and the
client-side bootstrap path all live in `core/anchored_bootstrap.py`;
this module is the *creation* + *promotion* side.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence

from .anchored_bootstrap import SeedRecord, SeedRegistry
from .tokenomics import Wallet

logger = logging.getLogger(__name__)


TOFU_BOOTSTRAP_FLAG = "single-validator-tofu"
QUORUM_MIN_DISTINCT_SIGNERS = 3      # promote after this many distinct signers
DEFAULT_BTC_HEIGHT = 0               # epoch zero when no anchor available


def build_tofu_seed_record(
    *,
    host: str,
    port: int,
    wallet: Wallet,
    region: str = "tofu-bootstrap",
    btc_height: int = DEFAULT_BTC_HEIGHT,
) -> SeedRecord:
    """Construct a SeedRecord signed exclusively by `wallet`. The
    registry treats this as `single-validator-tofu` until at least
    `QUORUM_MIN_DISTINCT_SIGNERS` wallets co-sign."""
    pubkey_pem = wallet.export_keys()["public"]
    rec = SeedRecord(
        host=host,
        port=port,
        pubkey_pem=pubkey_pem,
        region=region,
        registered_at_btc_height=btc_height,
        quorum_signatures=[],
    )
    sig = wallet.sign(rec.canonical())
    rec.quorum_signatures.append({
        "pubkey": pubkey_pem,
        "value": sig,
        "label": TOFU_BOOTSTRAP_FLAG,
    })
    return rec


def co_sign_seed_record(
    *,
    record: SeedRecord,
    wallet: Wallet,
) -> SeedRecord:
    """Add another wallet's signature to an existing record. Idempotent
    — a wallet that has already signed gets refused."""
    pubkey_pem = wallet.export_keys()["public"]
    for sig in record.quorum_signatures:
        if sig.get("pubkey") == pubkey_pem:
            return record
    sig_value = wallet.sign(record.canonical())
    record.quorum_signatures.append({
        "pubkey": pubkey_pem,
        "value": sig_value,
        "label": "quorum-cosign",
    })
    return record


def is_tofu_only(record: SeedRecord) -> bool:
    """Return True iff the record has fewer than the quorum threshold
    of distinct signers."""
    return len(distinct_signers(record)) < QUORUM_MIN_DISTINCT_SIGNERS


def distinct_signers(record: SeedRecord) -> List[str]:
    """Pubkeys, deduplicated, only those whose signature actually
    verifies under the record's canonical body."""
    out: List[str] = []
    seen: set = set()
    msg = record.canonical()
    for sig in record.quorum_signatures:
        pem = sig.get("pubkey")
        if not pem or pem in seen:
            continue
        try:
            if Wallet.verify(pem, msg, sig.get("value", "")):
                out.append(pem)
                seen.add(pem)
        except Exception:
            continue
    return out


def build_initial_registry(
    *,
    host: str,
    port: int,
    wallet: Wallet,
    region: str = "tofu-bootstrap",
    btc_height: int = DEFAULT_BTC_HEIGHT,
) -> SeedRegistry:
    """First-run convenience: a single-record TOFU registry. The
    epoch_btc_height is recorded so the registry's lineage is
    auditable when (eventually) re-signed by a real quorum."""
    rec = build_tofu_seed_record(
        host=host, port=port, wallet=wallet,
        region=region, btc_height=btc_height,
    )
    return SeedRegistry(records=[rec], epoch_btc_height=btc_height)


def promote_to_quorum(
    registry: SeedRegistry,
    additional_wallets: Iterable[Wallet],
) -> SeedRegistry:
    """Co-sign every TOFU record in the registry with each provided
    wallet. Returns the same registry mutated in place + returned for
    fluent style. After promotion, `is_tofu_only` for each record may
    transition to False (depending on how many additional wallets
    are supplied)."""
    wallets = list(additional_wallets)
    for rec in registry.records:
        for w in wallets:
            co_sign_seed_record(record=rec, wallet=w)
    return registry


__all__ = [
    "QUORUM_MIN_DISTINCT_SIGNERS",
    "TOFU_BOOTSTRAP_FLAG",
    "build_initial_registry",
    "build_tofu_seed_record",
    "co_sign_seed_record",
    "distinct_signers",
    "is_tofu_only",
    "promote_to_quorum",
]
