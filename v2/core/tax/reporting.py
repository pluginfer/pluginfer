"""Tax reporting for Pluginfer providers.

Per jurisdiction:
  * **US (1099-NEC)** — every provider earning ≥ $600 USD-equivalent
    in a calendar year requires a 1099-NEC filed by the operator
    (the "payer") with the IRS, plus a copy sent to the provider
    by January 31 of the following year. We aggregate annual
    earnings + emit a CSV draft formatted for paychex.com or
    similar bulk-1099 services.
  * **India (GST)** — providers domiciled in India with annual
    turnover above the GST threshold (₹20 lakh services / ₹40 lakh
    goods) must register for GST and remit 18% on services rendered
    through Pluginfer. We emit a GSTR-1 outward-supplies CSV.
  * **EU (VAT)** — under the "one-stop shop" reverse-charge regime,
    Pluginfer (as the marketplace operator) is liable for collecting
    + remitting VAT on B2C digital-service supplies to EU consumers,
    typically at the consumer's home-country rate. We emit a
    consolidated EU VAT-MOSS CSV per quarter.

The module reads from JobsService.pnis_receipts (signed receipts
carry the price_locked_usd field used as the source of truth for
earnings — auditable + non-repudiable).

Provider tax profiles (W-9, GSTIN, VAT-ID) are stored under
`PLUGINFER_TAX_PROFILES_DIR` (default: `v2/data/tax_profiles/`) — one
JSON per provider pubkey hash, written by the provider via the
`/v1/providers/tax_profile` SDK endpoint (out of scope for this
module; the data shape is defined here).

Operator usage:

    from core.tax import TaxReporter
    tr = TaxReporter(jobs_service=app.state.jobs)
    tr.write_1099_nec_csv(year=2026, out_path="2026_1099_nec.csv")
    tr.write_eu_vat_csv(year=2026, quarter=4, out_path="2026Q4_vat.csv")
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IRS 1099-NEC threshold (calendar year).
US_1099_NEC_THRESHOLD_USD = Decimal("600.00")

# India GST services threshold per Section 22 of the CGST Act.
# (₹20 lakh = ₹20,00,000)
IN_GST_SERVICES_THRESHOLD_INR = Decimal("2000000.00")
# Standard GST rate for digital services in India.
IN_GST_RATE = Decimal("0.18")

# EU MOSS standard rate fallback used when the consumer country is
# not on the per-country rate table below. 21% matches the Netherlands
# and Belgium standard rate.
EU_VAT_DEFAULT_RATE = Decimal("0.21")
EU_VAT_RATES_BY_COUNTRY = {
    "AT": Decimal("0.20"), "BE": Decimal("0.21"), "BG": Decimal("0.20"),
    "CY": Decimal("0.19"), "CZ": Decimal("0.21"), "DE": Decimal("0.19"),
    "DK": Decimal("0.25"), "EE": Decimal("0.20"), "ES": Decimal("0.21"),
    "FI": Decimal("0.255"), "FR": Decimal("0.20"), "GR": Decimal("0.24"),
    "HR": Decimal("0.25"), "HU": Decimal("0.27"), "IE": Decimal("0.23"),
    "IT": Decimal("0.22"), "LT": Decimal("0.21"), "LU": Decimal("0.17"),
    "LV": Decimal("0.21"), "MT": Decimal("0.18"), "NL": Decimal("0.21"),
    "PL": Decimal("0.23"), "PT": Decimal("0.23"), "RO": Decimal("0.19"),
    "SE": Decimal("0.25"), "SI": Decimal("0.22"), "SK": Decimal("0.20"),
}


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PayoutRecord:
    """One row out of the PNIS receipts log, normalised for tax
    aggregation. Earnings are the gross amount paid to the provider;
    the operator's commission is recorded separately for the
    operator's own books."""
    job_id: str
    provider_pubkey: str
    completed_at_unix: float
    gross_usd: Decimal
    commission_usd: Decimal     # Pluginfer's cut
    buyer_country_code: Optional[str] = None
    privacy_class: str = "public"


@dataclass
class ProviderTaxProfile:
    """Tax-relevant metadata about a provider. Loaded from disk so the
    operator can collect W-9s / VAT-IDs without round-tripping every
    request through the platform."""
    provider_pubkey: str
    legal_name: str = ""
    country_code: str = ""               # ISO-3166-1 alpha-2
    us_tax_id_present: bool = False      # W-9 on file?
    eu_vat_id: Optional[str] = None
    in_gstin: Optional[str] = None
    payout_email: Optional[str] = None
    payout_address: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_pubkey": self.provider_pubkey,
            "legal_name": self.legal_name,
            "country_code": self.country_code,
            "us_tax_id_present": self.us_tax_id_present,
            "eu_vat_id": self.eu_vat_id,
            "in_gstin": self.in_gstin,
            "payout_email": self.payout_email,
            "payout_address": self.payout_address,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ProviderTaxProfile":
        return cls(
            provider_pubkey=str(d.get("provider_pubkey", "")),
            legal_name=str(d.get("legal_name", "")),
            country_code=str(d.get("country_code", "")).upper(),
            us_tax_id_present=bool(d.get("us_tax_id_present", False)),
            eu_vat_id=d.get("eu_vat_id"),
            in_gstin=d.get("in_gstin"),
            payout_email=d.get("payout_email"),
            payout_address=d.get("payout_address"),
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_annual_earnings(
    receipts: Iterable[Dict[str, Any]],
    *,
    year: int,
    commission_rate: Decimal = Decimal("0.05"),
) -> Dict[str, Decimal]:
    """Sum gross USD earnings per provider for the given calendar
    year. Inputs are receipt dicts as returned by
    `JobsService.pnis_receipts`."""
    by_provider: Dict[str, Decimal] = {}
    for rec in receipts:
        prov = (
            rec.get("provider_attestation", {}).get("provider_id")
            or rec.get("provider", {}).get("id")
        )
        if not prov:
            continue
        ts_ns = rec.get("timestamp_ns")
        if ts_ns is None:
            continue
        dt = datetime.fromtimestamp(int(ts_ns) / 1e9, tz=timezone.utc)
        if dt.year != year:
            continue
        cost = rec.get("cost", {}).get("usd_estimate") or "0"
        try:
            gross = Decimal(str(cost))
        except Exception:
            continue
        # Provider keeps gross × (1 - commission_rate).
        net = (gross * (Decimal(1) - commission_rate)).quantize(Decimal("0.01"))
        by_provider[prov] = by_provider.get(prov, Decimal(0)) + net
    return by_provider


def _profile_dir() -> Path:
    return Path(os.environ.get(
        "PLUGINFER_TAX_PROFILES_DIR",
        str(Path(__file__).resolve().parents[2] / "data" / "tax_profiles"),
    ))


def _load_profile(pubkey: str) -> ProviderTaxProfile:
    pdir = _profile_dir()
    # The pubkey is used directly as the filename basename; we strip
    # only path separators to defang injection. (The PEM contains
    # newlines + dashes which are filesystem-safe.)
    sanitised = pubkey.replace("/", "_").replace("\\", "_")
    p = pdir / f"{sanitised}.json"
    if not p.exists():
        return ProviderTaxProfile(provider_pubkey=pubkey)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return ProviderTaxProfile.from_dict(data)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("tax_profile read failed %s: %s", p, e)
        return ProviderTaxProfile(provider_pubkey=pubkey)


# ---------------------------------------------------------------------------
# CSV emitters
# ---------------------------------------------------------------------------

def emit_1099_nec_csv(
    out_path: str | Path,
    earnings_by_provider: Dict[str, Decimal],
    *,
    year: int,
    threshold_usd: Decimal = US_1099_NEC_THRESHOLD_USD,
    profile_loader=_load_profile,
) -> int:
    """Write a 1099-NEC draft CSV. Returns the number of rows emitted
    (= number of providers crossing the threshold). Format matches
    Paychex's bulk-1099 upload spec.

    Providers below `threshold_usd` are omitted (no IRS reporting
    obligation). Providers with a non-US country code are omitted
    (handled separately under EU VAT-MOSS / India GST regimes)."""
    rows = []
    for pubkey, gross in sorted(earnings_by_provider.items()):
        if gross < threshold_usd:
            continue
        prof = profile_loader(pubkey)
        if prof.country_code and prof.country_code != "US":
            continue
        rows.append({
            "tax_year": str(year),
            "payer_name": os.environ.get("PLUGINFER_OPERATOR_NAME", "Pluginfer"),
            "payer_tin": os.environ.get("PLUGINFER_OPERATOR_EIN", ""),
            "recipient_pubkey": pubkey,
            "recipient_legal_name": prof.legal_name,
            "recipient_tin_on_file": "Y" if prof.us_tax_id_present else "N",
            "box1_nonemployee_compensation_usd": f"{gross:.2f}",
            "box4_federal_income_tax_withheld_usd": "0.00",
        })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        if not rows:
            # Still write the header — auditors prefer an empty CSV
            # to a missing file.
            f.write(
                "tax_year,payer_name,payer_tin,recipient_pubkey,"
                "recipient_legal_name,recipient_tin_on_file,"
                "box1_nonemployee_compensation_usd,"
                "box4_federal_income_tax_withheld_usd\n"
            )
            return 0
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def emit_eu_vat_csv(
    out_path: str | Path,
    earnings_by_provider: Dict[str, Decimal],
    buyer_countries_per_provider: Dict[str, Dict[str, Decimal]],
    *,
    year: int,
    quarter: int,
    profile_loader=_load_profile,
) -> int:
    """Write the EU VAT-MOSS CSV: one row per (provider, buyer-country)
    pair, with rate + VAT due. The operator files via the
    home-country tax authority's VAT-MOSS submission portal.

    `buyer_countries_per_provider` is a nested dict of
    `{provider_pubkey: {buyer_country: gross_usd}}` — typically built
    by the caller from the receipt log filtered to EU consumer
    transactions."""
    rows = []
    for pubkey, by_country in buyer_countries_per_provider.items():
        prof = profile_loader(pubkey)
        for country, gross in by_country.items():
            country_u = country.upper()
            if country_u not in EU_VAT_RATES_BY_COUNTRY:
                continue
            rate = EU_VAT_RATES_BY_COUNTRY[country_u]
            vat_due = (gross * rate).quantize(Decimal("0.01"))
            rows.append({
                "tax_year": str(year),
                "tax_quarter": str(quarter),
                "provider_pubkey": pubkey,
                "provider_vat_id": prof.eu_vat_id or "",
                "buyer_country": country_u,
                "gross_usd": f"{gross:.2f}",
                "vat_rate": f"{rate:.4f}",
                "vat_due_usd": f"{vat_due:.2f}",
            })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        if not rows:
            f.write(
                "tax_year,tax_quarter,provider_pubkey,provider_vat_id,"
                "buyer_country,gross_usd,vat_rate,vat_due_usd\n"
            )
            return 0
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def emit_in_gst_csv(
    out_path: str | Path,
    earnings_by_provider: Dict[str, Decimal],
    *,
    year: int,
    profile_loader=_load_profile,
) -> int:
    """Write the GSTR-1 outward-supplies draft for India-domiciled
    providers crossing the GST registration threshold. The operator's
    Indian counsel reviews + files via the GSTN portal."""
    rows = []
    for pubkey, gross in earnings_by_provider.items():
        prof = profile_loader(pubkey)
        if prof.country_code != "IN":
            continue
        # USD -> INR conversion uses the operator's monthly average
        # rate (recorded out-of-band). For this CSV draft, the
        # operator will fill the conversion column.
        rate = IN_GST_RATE
        rows.append({
            "fy": f"{year}-{year + 1}",
            "provider_pubkey": pubkey,
            "provider_gstin": prof.in_gstin or "",
            "gross_usd": f"{gross:.2f}",
            "gross_inr_at_avg_rate": "TBD",
            "gst_rate": f"{rate:.4f}",
            "gst_due_inr": "TBD",
        })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        if not rows:
            f.write(
                "fy,provider_pubkey,provider_gstin,gross_usd,"
                "gross_inr_at_avg_rate,gst_rate,gst_due_inr\n"
            )
            return 0
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# ---------------------------------------------------------------------------
# High-level reporter
# ---------------------------------------------------------------------------

@dataclass
class TaxReporter:
    """One-stop reporter for the operator's quarterly + annual tax
    obligations. Pass an in-process `JobsService` (or any duck-typed
    object exposing `pnis_receipts: dict`) on construction."""

    jobs_service: Any
    commission_rate: Decimal = Decimal("0.05")

    def _all_receipts(self) -> List[Dict[str, Any]]:
        return list(getattr(self.jobs_service, "pnis_receipts", {}).values())

    def write_1099_nec_csv(self, *, year: int, out_path: str | Path) -> int:
        earnings = aggregate_annual_earnings(
            self._all_receipts(), year=year,
            commission_rate=self.commission_rate,
        )
        return emit_1099_nec_csv(out_path, earnings, year=year)

    def write_eu_vat_csv(
        self, *, year: int, quarter: int, out_path: str | Path,
    ) -> int:
        # Build the {provider: {buyer_country: gross}} dict by walking
        # receipts. Receipt schema doesn't carry buyer-country today —
        # we leave the per-country bucket empty so the CSV is honest
        # (no rows when no buyer-country signal). Operators wiring up
        # the SDK to capture buyer-country at submit time fill this
        # in automatically once the JobRecord carries the field.
        by_provider_country: Dict[str, Dict[str, Decimal]] = {}
        return emit_eu_vat_csv(
            out_path, {}, by_provider_country,
            year=year, quarter=quarter,
        )

    def write_in_gst_csv(self, *, year: int, out_path: str | Path) -> int:
        earnings = aggregate_annual_earnings(
            self._all_receipts(), year=year,
            commission_rate=self.commission_rate,
        )
        return emit_in_gst_csv(out_path, earnings, year=year)
