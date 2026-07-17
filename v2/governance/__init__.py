"""Pluginfer Governance — standalone AI-spend control + token thrift.

A DELIBERATELY SELF-CONTAINED package: it imports nothing from
``core`` or ``api`` (those drag in the mesh + torch, ~1650 modules /
~3 s). This package is pure stdlib + FastAPI, so an organization can
deploy it ON THEIR PREMISES with none of the mesh — budget envelopes,
governed streaming, exact + semantic cache, prompt compression, signed
hash-chained receipts, per-key attribution, and the dashboard, fronting
their own OpenAI/Anthropic-compatible endpoint.

The mesh path (JobsService) also imports ``BudgetLedger`` from here, so
there is one implementation, not two — the mesh depends on governance,
never the reverse. That one-way arrow is what keeps this package
installable alone.

Import is lazy so pulling in ``BudgetLedger`` alone doesn't import
FastAPI:

    from governance import BudgetLedger          # stdlib only
    from governance import build_governance_gateway  # + FastAPI
"""

__all__ = [
    "BudgetLedger", "Envelope",
    "ResponseCache", "SemanticCache", "PromptCompressor",
    "GovernanceGateway", "build_governance_gateway",
]


def __getattr__(name):
    if name in ("BudgetLedger", "Envelope"):
        from . import budget_ledger
        return getattr(budget_ledger, name)
    if name in ("SemanticCache", "PromptCompressor"):
        from . import token_thrift
        return getattr(token_thrift, name)
    if name in ("GovernanceGateway", "build_governance_gateway",
                "ResponseCache"):
        from . import gateway
        return getattr(gateway, name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")


# Printed at service startup (CLI boot output only — never spammed
# into per-file headers; the NOTICE file carries the binding
# attribution under Apache-2.0 4(d)).
BANNER = r"""
  ____  _  ____ _   _ _____ _____
 / ___|| |/ ___| \ | | ____|_   _|
 \___ \| | |  _|  \| |  _|   | |
  ___) | | |_| | |\  | |___  | |
 |____/|_|\____|_| \_|_____| |_|
 Pluginfer SIGNET - enforced | signed | provable
"""
