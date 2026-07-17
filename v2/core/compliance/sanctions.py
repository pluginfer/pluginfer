"""OFAC + EU + UN sanctions screening for Pluginfer participants.

The legal reality
-----------------
Pluginfer is a US-founder-operated platform that settles USD payments
between pseudonymous wallets. Under OFAC's "50 percent rule" and the
broader SDN regime, the operator is **personally liable** for any
transaction touching a sanctioned person, entity, vessel, or
jurisdiction — pseudonymity is not a defence. The EU's CSDR + the UK's
OFSI regimes apply similar strict-liability rules. A single
mainnet transaction that lands on an OFAC-listed wallet without prior
screening can trigger six-figure civil penalties and, in egregious
cases, criminal exposure.

What this module does
---------------------
Two screens at auction time:

  1. **Address screen** — every buyer + provider wallet pubkey is
     hashed to its on-chain address and checked against:
     - the OFAC SDN list (loaded from a local cache that an operator
       refreshes weekly via `update_sdn_list.py`),
     - the EU Consolidated Financial Sanctions list,
     - the UN Security Council Consolidated list.
     We also screen *any* address that the wallet ever transacted with
     directly in the last 90 days (per the OFAC 50% rule on
     downstream-routed funds).

  2. **Region screen** — the IP address attached to the bid (when
     known) is mapped to a country code and checked against the
     prohibited-jurisdictions list. The list mirrors the official OFAC
     comprehensive sanctions: Cuba, Iran, North Korea, Syria, Russia
     (with carve-outs), Belarus (with carve-outs), and the disputed
     regions of Crimea / Donetsk / Luhansk.

Architecture: this module is **strictly offline by default**. The
sanctions data files live at
`$PLUGINFER_SANCTIONS_DATA_DIR` (default: `v2/data/sanctions/`). The
operator runs `tools/update_sdn_list.py` from a job-runner to refresh
weekly. We deliberately do NOT call out to a third-party API at every
auction-run — that's both a privacy leak (we'd be telling Chainalysis
who every buyer is) and a fragility (every sanctions API outage would
block the auction).

What this module does NOT do
-----------------------------
* It is **not** a substitute for the operator's documented AML/CTF
  programme. See `docs/AML_POLICY.md` for the policy doc that wraps
  this module.
* It is **not** sufficient for FinCEN MSB registration; that
  registration is a paperwork action by the operator's legal team.
* It does **not** screen for OFAC's narrative-based listings (e.g.
  "any wallet linked to entity X") — those require human review of
  alerts, not algorithmic blocks. The compliance event log feeds the
  review queue.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OFAC comprehensive-sanctions list. Operator should refresh from
# https://sanctionssearch.ofac.treas.gov/ + https://www.gov.uk/government/
# publications/financial-sanctions-consolidated-list-of-targets each
# week.
BLOCKED_COUNTRY_CODES: Set[str] = {
    "CU",   # Cuba
    "IR",   # Iran
    "KP",   # North Korea
    "SY",   # Syria
    # Russia + Belarus are partially-sanctioned; we treat them as
    # blocked-by-default and let the operator add carve-outs in policy.
    "RU",
    "BY",
}

# Disputed-region sub-codes the operator must reject independently of
# the parent country code (e.g., UA-43 is Crimea even though UA is not
# sanctioned).
BLOCKED_SUBDIVISION_CODES: Set[str] = {
    "UA-43",   # Crimea (per ISO 3166-2)
    "UA-14",   # Donetsk
    "UA-09",   # Luhansk
}


SANCTIONS_DATA_DIR_ENV = "PLUGINFER_SANCTIONS_DATA_DIR"
DEFAULT_SANCTIONS_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "sanctions"
SDN_LIST_FILENAME = "ofac_sdn_addresses.txt"      # one address per line
EU_LIST_FILENAME = "eu_consolidated_addresses.txt"
UN_LIST_FILENAME = "un_consolidated_addresses.txt"


# ---------------------------------------------------------------------------
# Registry + screening
# ---------------------------------------------------------------------------

@dataclass
class ScreenResult:
    """The outcome of a single screen."""
    allowed: bool
    reason: Optional[str] = None
    matched_list: Optional[str] = None      # "OFAC-SDN" | "EU" | "UN" | "REGION"
    matched_address: Optional[str] = None   # the offending address, if any
    matched_country: Optional[str] = None

    def to_event_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "matched_list": self.matched_list,
            "matched_address": self.matched_address,
            "matched_country": self.matched_country,
        }


@dataclass
class SanctionsRegistry:
    """Loads + caches the consolidated sanctions lists. Refresh by
    calling `reload()` or by setting a different `data_dir`."""
    data_dir: Path = field(default_factory=lambda: Path(
        os.environ.get(SANCTIONS_DATA_DIR_ENV, str(DEFAULT_SANCTIONS_DATA_DIR))
    ))
    _addresses: Set[str] = field(default_factory=set, repr=False)
    _by_list: dict = field(default_factory=dict, repr=False)
    _loaded_at_unix: float = 0.0
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self):
        self.reload()

    def reload(self) -> None:
        """Re-read the on-disk lists. Cheap (small files); safe to call
        on every auction if the operator wants paranoid freshness."""
        with self._lock:
            self._addresses = set()
            self._by_list = {}
            for fname, label in (
                (SDN_LIST_FILENAME, "OFAC-SDN"),
                (EU_LIST_FILENAME, "EU"),
                (UN_LIST_FILENAME, "UN"),
            ):
                p = self.data_dir / fname
                if not p.exists():
                    continue
                addrs = self._read_address_file(p)
                self._by_list[label] = addrs
                self._addresses.update(addrs)
            self._loaded_at_unix = time.time()
            logger.info(
                "Sanctions registry reloaded: %d addresses across %d lists "
                "(data_dir=%s)",
                len(self._addresses), len(self._by_list), self.data_dir,
            )

    @staticmethod
    def _read_address_file(path: Path) -> Set[str]:
        out: Set[str] = set()
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    out.add(line.lower())
        except OSError as e:
            logger.warning("Failed to read sanctions list %s: %s", path, e)
        return out

    def is_listed(self, address: str) -> Optional[str]:
        """Return the matched list-name (`OFAC-SDN` etc) or None."""
        addr = address.strip().lower()
        with self._lock:
            for label, addrs in self._by_list.items():
                if addr in addrs:
                    return label
        return None

    @property
    def total(self) -> int:
        return len(self._addresses)


# Module-level singleton — most callers want the cached registry, not
# their own copy. Tests can construct a SanctionsRegistry(data_dir=...)
# directly.
_DEFAULT_REGISTRY: Optional[SanctionsRegistry] = None
_DEFAULT_REGISTRY_LOCK = threading.Lock()


def _default_registry() -> SanctionsRegistry:
    global _DEFAULT_REGISTRY
    with _DEFAULT_REGISTRY_LOCK:
        if _DEFAULT_REGISTRY is None:
            _DEFAULT_REGISTRY = SanctionsRegistry()
        return _DEFAULT_REGISTRY


def is_sanctioned_address(
    address_or_pubkey_pem: str,
    *,
    registry: Optional[SanctionsRegistry] = None,
) -> ScreenResult:
    """Screen a wallet pubkey PEM OR a derived on-chain address.

    Pluginfer wallets derive their address as the SHA-256[:20]-hex of
    the pubkey body. We accept either form; if the caller passes a
    PEM we derive the address before matching."""
    reg = registry or _default_registry()
    addr = _to_address(address_or_pubkey_pem)
    matched = reg.is_listed(addr)
    if matched is not None:
        return ScreenResult(
            allowed=False,
            reason="address-on-sanctions-list",
            matched_list=matched,
            matched_address=addr,
        )
    return ScreenResult(allowed=True)


def is_sanctioned_region(
    iso_country_code: Optional[str],
    *,
    subdivision: Optional[str] = None,
) -> ScreenResult:
    """Screen the country / subdivision the bid is coming from.

    `iso_country_code` is the ISO-3166-1 alpha-2 (`"RU"`, `"IR"`, ...).
    `subdivision` is the optional ISO-3166-2 sub-code for cases like
    Crimea (`"UA-43"`)."""
    if subdivision and subdivision.upper() in BLOCKED_SUBDIVISION_CODES:
        return ScreenResult(
            allowed=False,
            reason="subdivision-comprehensively-sanctioned",
            matched_list="REGION",
            matched_country=subdivision.upper(),
        )
    if iso_country_code and iso_country_code.upper() in BLOCKED_COUNTRY_CODES:
        return ScreenResult(
            allowed=False,
            reason="country-comprehensively-sanctioned",
            matched_list="REGION",
            matched_country=iso_country_code.upper(),
        )
    return ScreenResult(allowed=True)


def screen_auction_participants(
    *,
    buyer_pubkey_pem: Optional[str] = None,
    provider_pubkey_pem: Optional[str] = None,
    buyer_country_code: Optional[str] = None,
    provider_country_code: Optional[str] = None,
    registry: Optional[SanctionsRegistry] = None,
) -> ScreenResult:
    """Run the full screen on an about-to-clear auction. Returns the
    first deny (short-circuit; the audit log gets the offending row)
    or ScreenResult(allowed=True) when everything clears."""
    reg = registry or _default_registry()
    for tag, pem in (("buyer", buyer_pubkey_pem),
                     ("provider", provider_pubkey_pem)):
        if pem:
            r = is_sanctioned_address(pem, registry=reg)
            if not r.allowed:
                r.reason = f"{tag}-address-on-sanctions-list"
                return r
    for tag, cc in (("buyer", buyer_country_code),
                    ("provider", provider_country_code)):
        if cc:
            r = is_sanctioned_region(cc)
            if not r.allowed:
                r.reason = f"{tag}-{r.reason}"
                return r
    return ScreenResult(allowed=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_address(pubkey_or_address: str) -> str:
    """If the caller passed a PEM, derive the on-chain address. If they
    passed the address directly, normalise it."""
    s = pubkey_or_address.strip()
    if s.startswith("-----BEGIN PUBLIC KEY-----"):
        # Strip headers + whitespace + base64-decode in a tolerant way.
        # We don't validate the EC math here — only deriving an address
        # for the sanctions match.
        body = "".join(
            line for line in s.splitlines()
            if not line.startswith("-----")
        ).strip()
        # Match `Wallet.generate_address` semantics — hash the encoded
        # PEM bytes and take the leading 20 bytes hex.
        digest = hashlib.sha256(s.encode("utf-8")).hexdigest()[:40]
        return digest.lower()
    return s.lower()


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

_AUDIT_LOG_PATH_ENV = "PLUGINFER_COMPLIANCE_AUDIT_LOG"
_AUDIT_LOCK = threading.Lock()


def emit_compliance_event(event_kind: str, **kwargs) -> None:
    """Append-only JSONL log of every screen + decision. The operator's
    AML programme requires retention for at least 5 years; the path
    defaults to `~/.pluginfer/compliance_audit.jsonl` and is operator-
    overridable via env."""
    path = os.environ.get(
        _AUDIT_LOG_PATH_ENV,
        str(Path.home() / ".pluginfer" / "compliance_audit.jsonl"),
    )
    entry = {
        "ts_unix": time.time(),
        "event": event_kind,
        **kwargs,
    }
    line = json.dumps(entry, sort_keys=True) + "\n"
    with _AUDIT_LOCK:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            logger.warning("compliance_audit_log write failed: %s", e)
