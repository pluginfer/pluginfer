"""Pluginfer tax reporting plumbing — 1099-NEC (US), GST (India),
VAT (EU). Aggregates per-provider annual earnings from the PNIS
receipt log, attaches a W-9 / VAT-ID / GSTIN per provider, emits the
filing-ready CSV drafts.

The compliance officer reviews + files; this module is the data
plumbing, not the policy."""

from .reporting import (
    PayoutRecord,
    ProviderTaxProfile,
    TaxReporter,
    aggregate_annual_earnings,
    emit_1099_nec_csv,
    emit_eu_vat_csv,
    emit_in_gst_csv,
)

__all__ = [
    "PayoutRecord",
    "ProviderTaxProfile",
    "TaxReporter",
    "aggregate_annual_earnings",
    "emit_1099_nec_csv",
    "emit_eu_vat_csv",
    "emit_in_gst_csv",
]
