"""G10 — first-launch profile builder.

Backs the post-install UX one-pager:
  * Estimated $/day on the detected hardware (uses HARDWARE_TABLE +
    the operator's electricity rate; honest range, not best-case).
  * Game / idle detection status (so the UI can show "we'll pause
    while you game, resume when you idle").
  * Wallet bootstrapping state (already-on-disk? brand new?).
  * Sanctions-region pre-check (CF-IPCountry / system locale hint
    so users in blocked regions see the 451 BEFORE they've
    completed onboarding).

The module is **pure data** — no Tkinter, no Qt. Construction is
fast (~50ms on a laptop) so the installer's post-install hook can
spawn a tiny HTTP server, open the user's default browser to the
served page, and render the profile in real HTML/CSS without any
desktop-GUI dependency.

Output schema is stable JSON; the HTML one-pager (under
`v2/ui/first_launch/`) consumes it directly.
"""

from __future__ import annotations

import locale
import logging
import os
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FirstLaunchProfile:
    """The flat JSON the one-pager renders."""
    schema: str = "pluginfer-first-launch/v1"
    generated_at_unix: float = field(default_factory=time.time)
    hardware: Dict[str, Any] = field(default_factory=dict)
    earnings: Dict[str, Any] = field(default_factory=dict)
    game_detection: Dict[str, Any] = field(default_factory=dict)
    wallet: Dict[str, Any] = field(default_factory=dict)
    compliance: Dict[str, Any] = field(default_factory=dict)
    next_steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "generated_at_unix": self.generated_at_unix,
            "hardware": self.hardware,
            "earnings": self.earnings,
            "game_detection": self.game_detection,
            "wallet": self.wallet,
            "compliance": self.compliance,
            "next_steps": self.next_steps,
        }


# ---------------------------------------------------------------------------
# Hardware bucket -> earnings_estimator hardware_id mapping
# ---------------------------------------------------------------------------

# `HardwareDetector` returns a `tier` string from a smaller bucket
# than the earnings table; we map between them. When a detected GPU
# isn't in the table, fall back to a conservative midpoint id.
TIER_TO_EARNINGS_ID = {
    "high-end": "rtx-4090",
    "mid-range": "rtx-3070",
    "entry-level": "gtx-1650",
    "no-gpu": "cpu-only",
    "unknown": "gtx-1650",
}


def _detect_hardware_safely() -> Dict[str, Any]:
    """Run HardwareDetector but never raise; the first-launch UX must
    survive a hardware-probe failure (locked-down corporate laptops,
    sandboxed CI, etc.)."""
    try:
        from core.hardware_detector import HardwareDetector
        det = HardwareDetector()
        if hasattr(det, "detect"):
            info = det.detect()
        else:
            info = {}
        # Coerce whatever shape detect() returned into the dict shape
        # the JS one-pager renders.
        if not isinstance(info, dict):
            info = {}
        return {
            "gpu": info.get("gpu") or "unknown",
            "gpu_vram_gb": info.get("gpu_vram_gb") or 0,
            "cpu": info.get("cpu") or "unknown",
            "cores": info.get("cpu_cores") or info.get("cores") or 0,
            "ram_gb": info.get("ram_gb") or 0,
            "tier": info.get("tier") or "unknown",
            "os": info.get("os") or os.name,
        }
    except Exception as e:
        logger.warning("hardware detect failed: %s", e)
        return {
            "gpu": "detection-failed",
            "gpu_vram_gb": 0,
            "cpu": "unknown",
            "cores": 0,
            "ram_gb": 0,
            "tier": "unknown",
            "os": os.name,
        }


def _estimate_earnings_safely(
    hardware_tier: str,
    *,
    idle_hours_per_day: float,
    power_cost_usd_per_kwh: float,
) -> Dict[str, Any]:
    """Hand off to core.earnings_estimator with a tier-keyed fallback."""
    try:
        from core.earnings_estimator import estimate_earnings, format_estimate
        hw_id = TIER_TO_EARNINGS_ID.get(hardware_tier, "gtx-1650")
        est = estimate_earnings(
            hardware_id=hw_id,
            idle_hours_per_day=idle_hours_per_day,
            power_cost_usd_per_kwh=power_cost_usd_per_kwh,
        )
        # Best-effort serialise — the dataclass may carry Decimals.
        out = {
            "hardware_id_used": hw_id,
            "expected_usd_per_day": float(getattr(est, "expected_usd_per_day", 0)),
            "low_usd_per_day": float(getattr(est, "low_usd_per_day", 0)),
            "high_usd_per_day": float(getattr(est, "high_usd_per_day", 0)),
            "summary": format_estimate(est),
        }
        return out
    except Exception as e:
        logger.warning("earnings estimate failed: %s", e)
        return {
            "hardware_id_used": "fallback",
            "expected_usd_per_day": 0.0,
            "low_usd_per_day": 0.0,
            "high_usd_per_day": 0.0,
            "summary": "(earnings estimate unavailable)",
        }


def _game_detection_state() -> Dict[str, Any]:
    try:
        from core.game_detector import GameDetector
        det = GameDetector()
        gaming = bool(getattr(det, "is_gaming_now", lambda: False)())
        return {
            "is_gaming_now": gaming,
            "policy": "pause-when-gaming-resume-when-idle",
            "detector_available": True,
        }
    except Exception:
        return {
            "is_gaming_now": False,
            "policy": "pause-when-gaming-resume-when-idle",
            "detector_available": False,
        }


def _wallet_state(wallet_path: Optional[str] = None) -> Dict[str, Any]:
    path = wallet_path or os.path.join(
        os.path.expanduser("~"), ".pluginfer", "wallet.pem",
    )
    present = os.path.exists(path)
    return {
        "wallet_path": path,
        "exists_on_disk": present,
        "needs_passphrase": True,           # G3: refuses to write unencrypted
        "next_action": (
            "Use existing wallet"
            if present else
            "Generate a new wallet (will require passphrase)"
        ),
    }


def _compliance_precheck(
    *,
    country_code_hint: Optional[str],
) -> Dict[str, Any]:
    try:
        from core.compliance import is_sanctioned_region
        cc = (country_code_hint or "").upper() or None
        screen = is_sanctioned_region(cc)
        return {
            "country_code_hint": cc,
            "allowed": screen.allowed,
            "reason": screen.reason,
        }
    except Exception as e:
        return {
            "country_code_hint": country_code_hint,
            "allowed": True,
            "reason": f"compliance-check-skipped: {e}",
        }


def _country_hint_from_locale() -> Optional[str]:
    try:
        loc = locale.getlocale()
        if loc and loc[0]:
            # Locale e.g. "en_IN" -> "IN".
            if "_" in loc[0]:
                return loc[0].split("_", 1)[1][:2].upper()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_profile(
    *,
    idle_hours_per_day: float = 8.0,
    power_cost_usd_per_kwh: float = 0.12,
    country_code_hint: Optional[str] = None,
    wallet_path: Optional[str] = None,
) -> FirstLaunchProfile:
    """Construct the first-launch profile dict. Pure data — pass the
    result to whatever renderer (web one-pager, JSON-over-HTTP, etc.)."""
    hardware = _detect_hardware_safely()
    earnings = _estimate_earnings_safely(
        hardware["tier"],
        idle_hours_per_day=idle_hours_per_day,
        power_cost_usd_per_kwh=power_cost_usd_per_kwh,
    )
    game = _game_detection_state()
    wallet = _wallet_state(wallet_path=wallet_path)
    cc_hint = country_code_hint or _country_hint_from_locale()
    compliance = _compliance_precheck(country_code_hint=cc_hint)

    next_steps: List[str] = []
    if not compliance["allowed"]:
        next_steps.append(
            "We can't onboard you from a comprehensively-sanctioned "
            "jurisdiction. See docs/AML_POLICY.md."
        )
    else:
        if not wallet["exists_on_disk"]:
            next_steps.append(
                "Generate your wallet — you'll set a passphrase that "
                "encrypts the private key on disk."
            )
        next_steps.append("Start earning — Pluginfer will pause when you game and resume when you idle.")
        if earnings["expected_usd_per_day"] > 0:
            next_steps.append(
                f"Estimated earnings: ~${earnings['expected_usd_per_day']:.2f}/day "
                f"on the detected hardware tier."
            )

    return FirstLaunchProfile(
        hardware=hardware,
        earnings=earnings,
        game_detection=game,
        wallet=wallet,
        compliance=compliance,
        next_steps=next_steps,
    )


__all__ = [
    "FirstLaunchProfile",
    "build_profile",
    "TIER_TO_EARNINGS_ID",
]
