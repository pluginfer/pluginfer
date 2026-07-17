"""Pluginfer compliance plumbing — OFAC sanctions, IP geo screening,
AML controls. All of the below is the regulatory hardening required
before Pluginfer (the operator) settles any USD between pseudonymous
wallets across borders.

Public API:
    sanctions.is_sanctioned_address(pubkey_pem | wallet_address)
    sanctions.is_sanctioned_region(iso_country_code)
    sanctions.screen_auction_participants(...) -> ScreenResult
    sanctions.emit_compliance_event(...)        # appends to audit log
"""

from .sanctions import (
    BLOCKED_COUNTRY_CODES,
    ScreenResult,
    SanctionsRegistry,
    emit_compliance_event,
    is_sanctioned_address,
    is_sanctioned_region,
    screen_auction_participants,
)

__all__ = [
    "BLOCKED_COUNTRY_CODES",
    "ScreenResult",
    "SanctionsRegistry",
    "emit_compliance_event",
    "is_sanctioned_address",
    "is_sanctioned_region",
    "screen_auction_participants",
]
