"""G11 — tax reporting (1099-NEC + EU VAT + India GST).

Deterministic tests: synthesise PNIS receipts, run the aggregator,
emit CSV drafts, verify the contents.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from core.tax import (  # noqa: E402
    ProviderTaxProfile,
    TaxReporter,
    aggregate_annual_earnings,
    emit_1099_nec_csv,
    emit_eu_vat_csv,
    emit_in_gst_csv,
)
from core.tax.reporting import (  # noqa: E402
    EU_VAT_RATES_BY_COUNTRY,
    IN_GST_RATE,
)


_RECEIPT_SEQ = [0]


def _make_receipt(*, provider_id: str, year: int, cost_usd: str) -> dict:
    _RECEIPT_SEQ[0] += 1
    ts_ns = int(
        datetime(year, 6, 15, tzinfo=timezone.utc).timestamp() * 1e9
    ) + _RECEIPT_SEQ[0]
    return {
        "schema": "pnis-receipt/v1",
        "job_id": f"job-{provider_id}-{ts_ns}",
        "provider": {"id": provider_id},
        "provider_attestation": {"provider_id": provider_id},
        "timestamp_ns": ts_ns,
        "cost": {"plg": cost_usd, "usd_estimate": cost_usd},
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_aggregate_sums_per_provider_for_the_year():
    receipts = [
        _make_receipt(provider_id="alice", year=2026, cost_usd="100.00"),
        _make_receipt(provider_id="alice", year=2026, cost_usd="50.00"),
        _make_receipt(provider_id="bob",   year=2026, cost_usd="700.00"),
        _make_receipt(provider_id="alice", year=2025, cost_usd="999.00"),  # not in 2026
    ]
    out = aggregate_annual_earnings(receipts, year=2026,
                                    commission_rate=Decimal("0.05"))
    # alice in 2026: (100 + 50) × 0.95 = 142.50
    assert out["alice"] == Decimal("142.50")
    # bob: 700 × 0.95 = 665.00
    assert out["bob"] == Decimal("665.00")
    assert "carol" not in out


def test_aggregate_skips_receipts_without_provider_id():
    receipts = [
        {"timestamp_ns": int(
            datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1e9),
         "cost": {"usd_estimate": "100"}},
    ]
    assert aggregate_annual_earnings(receipts, year=2026) == {}


# ---------------------------------------------------------------------------
# 1099-NEC
# ---------------------------------------------------------------------------

def _us_profile(pubkey: str) -> ProviderTaxProfile:
    return ProviderTaxProfile(
        provider_pubkey=pubkey, legal_name="Alice US Provider",
        country_code="US", us_tax_id_present=True,
    )


def test_1099_csv_includes_only_us_providers_over_600usd(tmp_path):
    earnings = {
        "alice": Decimal("700.00"),     # over -> included
        "bob":   Decimal("500.00"),     # under -> excluded
    }
    out = tmp_path / "1099.csv"
    n = emit_1099_nec_csv(
        out, earnings, year=2026,
        profile_loader=lambda pk: _us_profile(pk),
    )
    assert n == 1
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    assert rows[0]["recipient_pubkey"] == "alice"
    assert rows[0]["box1_nonemployee_compensation_usd"] == "700.00"
    assert rows[0]["tax_year"] == "2026"


def test_1099_csv_excludes_non_us_providers(tmp_path):
    earnings = {"alice": Decimal("700.00")}
    out = tmp_path / "1099.csv"
    n = emit_1099_nec_csv(
        out, earnings, year=2026,
        profile_loader=lambda pk: ProviderTaxProfile(
            provider_pubkey=pk, country_code="IN",
        ),
    )
    assert n == 0
    # Empty file still carries the header (auditor-friendly).
    text = out.read_text()
    assert text.startswith("tax_year,payer_name")


# ---------------------------------------------------------------------------
# EU VAT
# ---------------------------------------------------------------------------

def test_eu_vat_csv_applies_per_country_rate(tmp_path):
    out = tmp_path / "vat.csv"
    n = emit_eu_vat_csv(
        out,
        earnings_by_provider={},   # not used by emitter directly
        buyer_countries_per_provider={
            "alice": {"DE": Decimal("100.00"), "FR": Decimal("50.00")},
            "bob":   {"NL": Decimal("200.00")},
        },
        year=2026, quarter=4,
        profile_loader=lambda pk: ProviderTaxProfile(provider_pubkey=pk),
    )
    assert n == 3   # alice/DE, alice/FR, bob/NL
    rows = list(csv.DictReader(out.open()))
    de = next(r for r in rows if r["buyer_country"] == "DE")
    assert Decimal(de["vat_rate"]) == EU_VAT_RATES_BY_COUNTRY["DE"]
    # VAT due = 100 × 0.19 = 19.00
    assert de["vat_due_usd"] == "19.00"


def test_eu_vat_csv_unknown_country_is_skipped(tmp_path):
    out = tmp_path / "vat.csv"
    n = emit_eu_vat_csv(
        out, {}, {"alice": {"ZW": Decimal("100.00")}},
        year=2026, quarter=4,
        profile_loader=lambda pk: ProviderTaxProfile(provider_pubkey=pk),
    )
    # ZW is not in the EU rate table.
    assert n == 0


# ---------------------------------------------------------------------------
# India GST
# ---------------------------------------------------------------------------

def test_in_gst_csv_only_for_in_providers(tmp_path):
    out = tmp_path / "gst.csv"
    n = emit_in_gst_csv(
        out,
        earnings_by_provider={
            "alice": Decimal("700.00"),
            "bob":   Decimal("700.00"),
        },
        year=2026,
        profile_loader=lambda pk: ProviderTaxProfile(
            provider_pubkey=pk,
            country_code="IN" if pk == "alice" else "US",
            in_gstin="29AAACC1206D1Z2" if pk == "alice" else None,
        ),
    )
    assert n == 1
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["provider_pubkey"] == "alice"
    assert rows[0]["provider_gstin"] == "29AAACC1206D1Z2"
    assert Decimal(rows[0]["gst_rate"]) == IN_GST_RATE


# ---------------------------------------------------------------------------
# End-to-end via TaxReporter
# ---------------------------------------------------------------------------

class _FakeJobsService:
    def __init__(self, receipts):
        self.pnis_receipts = {r["job_id"]: r for r in receipts}


def test_tax_reporter_writes_1099_csv_end_to_end(tmp_path):
    svc = _FakeJobsService([
        _make_receipt(provider_id="us-alice", year=2026, cost_usd="800.00"),
        _make_receipt(provider_id="us-alice", year=2026, cost_usd="800.00"),
    ])
    tr = TaxReporter(jobs_service=svc)
    out = tmp_path / "2026_1099.csv"
    # Patch the profile loader to make alice US so the row lands.
    import core.tax.reporting as rep
    original = rep._load_profile
    rep._load_profile = lambda pk: ProviderTaxProfile(
        provider_pubkey=pk, country_code="US", us_tax_id_present=True,
    )
    try:
        n = tr.write_1099_nec_csv(year=2026, out_path=out)
    finally:
        rep._load_profile = original
    assert n == 1
    rows = list(csv.DictReader(out.open()))
    # 800 + 800 = 1600 gross; net = 1600 × 0.95 = 1520
    assert rows[0]["box1_nonemployee_compensation_usd"] == "1520.00"
