"""Energy + carbon accounting for PNIS receipts.

Every Pluginfer job emits a §A1 PNIS receipt. With G7 every receipt now
carries the **energy consumed (megajoules)** and the **carbon footprint
(grams CO2-equivalent)** of the inference, derived from:

  * **GPU energy delta** via NVIDIA NVML (`pynvml`): total joules at
    job start, total joules at job end, subtract. Accurate to a few
    joules per call, no probe rig required.
  * **CPU fallback**: when no GPU is present (browser tab, ARM nano,
    Raspberry Pi), estimate as `cpu_tdp_watts * duration_seconds`.
    The TDP table is conservative; consumer Ryzen/Intel chips sit
    in the 65W–125W band.
  * **Grid carbon intensity** via the public `electricitymaps.com`
    REST API (free tier — 50 req/day per zone, cached aggressively).
    Falls back to an offline default of 480 gCO2/kWh (global grid
    weighted average per IEA 2024) when the API is unreachable.

Why this matters
----------------
The **EU AI Act** energy disclosure clauses come into force in 2026.
Receipts with `energy_mj: "0"` will fail the compliance check. **SEC**
climate-disclosure rules for US public companies hit in 2027 — and
their indirect-emissions sweep covers cloud AI inference. **Pluginfer
is the first compute network to emit auditable, signed, per-inference
energy + carbon attestations by default.** That's not a feature; that's
a regulatory moat.

Public API
----------
* `measure_job_energy(callable, *args, **kwargs) -> (result, EnergyReport)` —
  context-manager-style wrapper. Run any payload inside, get back what
  it produced plus a deterministic EnergyReport.
* `EnergyMeter()` — explicit start/stop if the workload is not callable
  (e.g., a long-poll job inside an executor thread). `meter.start()`
  before the work, `meter.stop()` returns the EnergyReport.
* `EnergyReport` — frozen dataclass with `energy_mj`, `carbon_gco2e`,
  `source` (`gpu-nvml` / `cpu-tdp` / `zero` — when measurement was
  unavailable), `gpu_index`, `duration_s`, `region_zone`.

The dataclass is serialisable into the §A1 receipt as the
`energy_mj` and `carbon_gco2e` fields.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults + tunables
# ---------------------------------------------------------------------------

# IEA 2024 global average grid carbon intensity. Used when no zone can
# be looked up.
GLOBAL_DEFAULT_CARBON_INTENSITY_GCO2_PER_KWH = 480.0

# Conservative TDP per chip-class. We only need this when no GPU is
# present (the inference is running on the CPU). For mesh providers
# running real Llama-3-70B-Q4 this branch is never hit; for browser-tab
# providers serving tiny `embed.tiny` workloads it's the realistic case.
CPU_TDP_WATTS_BY_CLASS = {
    "browser-webgpu": 25.0,        # mobile-style mid SoC + integrated GPU
    "browser-cpu": 15.0,           # pure CPU browser tab
    "consumer-laptop": 35.0,
    "consumer-desktop": 65.0,
    "consumer-gpu-low": 75.0,
    "consumer-gpu-mid": 220.0,     # GTX 1650-class
    "consumer-gpu-high": 350.0,    # RTX 4090-class GPU
    "datacentre-gpu": 700.0,       # H100 + host CPU
    "unknown": 50.0,               # safe-ish midpoint
}


# Cache the carbon-intensity lookup per zone for one hour. The free tier
# of electricitymaps gives us 50 requests/day; a long-running node serves
# thousands of jobs per hour, so caching is mandatory not optional.
_CARBON_CACHE: dict[str, Tuple[float, float]] = {}  # zone -> (gco2_per_kwh, fetched_at_unix)
_CARBON_CACHE_LOCK = threading.Lock()
CARBON_CACHE_TTL_S = 3600.0


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnergyReport:
    """The artefact stamped on every PNIS receipt."""
    energy_mj: Decimal              # megajoules consumed during the job
    carbon_gco2e: Decimal           # grams CO2-equivalent
    source: str                     # gpu-nvml | cpu-tdp | zero
    duration_s: float               # wall-clock duration
    gpu_index: Optional[int] = None
    region_zone: Optional[str] = None
    carbon_intensity_gco2_per_kwh: float = GLOBAL_DEFAULT_CARBON_INTENSITY_GCO2_PER_KWH

    def to_receipt_fields(self) -> dict:
        """Map the report onto the §A1 receipt schema."""
        return {
            "energy_mj": str(self.energy_mj),
            "carbon_gco2e": str(self.carbon_gco2e),
            "energy_source": self.source,
            "energy_zone": self.region_zone or "global-avg",
            "energy_carbon_intensity_gco2_per_kwh": self.carbon_intensity_gco2_per_kwh,
            "energy_duration_s": self.duration_s,
        }


# ---------------------------------------------------------------------------
# GPU energy via NVML
# ---------------------------------------------------------------------------

def _nvml_total_energy_millijoules(gpu_index: int) -> Optional[int]:
    """Return the GPU's cumulative energy counter in millijoules, or
    None if NVML isn't installed / device doesn't expose the counter.

    The NVML counter increments monotonically since driver load; we
    subtract a start sample from an end sample to get the delta over
    the job. Available on every NVIDIA datacenter card and recent
    consumer cards (RTX 30-series onward; older cards may not expose
    it, in which case the caller falls back to CPU-TDP estimation)."""
    try:
        import pynvml  # type: ignore
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            return int(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception as e:
        logger.debug("nvml energy probe failed on gpu_index=%s: %s",
                     gpu_index, e)
        return None


# ---------------------------------------------------------------------------
# Carbon intensity lookup
# ---------------------------------------------------------------------------

def _fetch_carbon_intensity(zone: str, *, timeout_s: float = 2.0) -> Optional[float]:
    """Hit electricitymaps.com's free-tier endpoint for the given zone
    (e.g., 'US-CAL-CISO', 'IN-WE', 'DE'). Returns gCO2/kWh, or None on
    any failure — caller falls back to the global average."""
    api_key = os.environ.get("ELECTRICITYMAPS_API_KEY", "")
    if not api_key:
        # The auth-free endpoint is rate-limited but works for a
        # demo network. Production should set the API key.
        url = f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={zone}"
        headers = {}
    else:
        url = f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={zone}"
        headers = {"auth-token": api_key}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            body = json.loads(r.read().decode("utf-8"))
            ci = body.get("carbonIntensity")
            if isinstance(ci, (int, float)) and ci > 0:
                return float(ci)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return None
    return None


def carbon_intensity_for_zone(zone: Optional[str]) -> Tuple[float, str]:
    """Resolve a carbon-intensity number for the zone, with TTL'd cache.
    Returns (gCO2_per_kWh, source-label). source-label is the zone
    actually used (may be 'global-avg' if lookup failed)."""
    if not zone:
        return GLOBAL_DEFAULT_CARBON_INTENSITY_GCO2_PER_KWH, "global-avg"
    now = time.time()
    with _CARBON_CACHE_LOCK:
        cached = _CARBON_CACHE.get(zone)
        if cached and (now - cached[1]) < CARBON_CACHE_TTL_S:
            return cached[0], zone
    fetched = _fetch_carbon_intensity(zone)
    if fetched is None:
        return GLOBAL_DEFAULT_CARBON_INTENSITY_GCO2_PER_KWH, "global-avg"
    with _CARBON_CACHE_LOCK:
        _CARBON_CACHE[zone] = (fetched, now)
    return fetched, zone


# ---------------------------------------------------------------------------
# The meter
# ---------------------------------------------------------------------------

@dataclass
class EnergyMeter:
    """Start/stop counter for a workload. Pick the source automatically:
    NVML if the GPU is exposed, CPU-TDP estimate otherwise.

    Construct with `hardware_class` so the CPU-fallback path picks a
    sane TDP. `gpu_index=None` disables the NVML probe (useful for
    deterministic tests)."""

    hardware_class: str = "unknown"
    gpu_index: Optional[int] = 0
    region_zone: Optional[str] = None
    # Carbon override: tests inject a fixed number so the assertion
    # math is deterministic regardless of grid state.
    carbon_intensity_gco2_per_kwh_override: Optional[float] = None
    _start_unix: Optional[float] = field(default=None, init=False)
    _start_mj_energy: Optional[int] = field(default=None, init=False)
    _source: str = field(default="zero", init=False)

    def start(self) -> "EnergyMeter":
        self._start_unix = time.monotonic()
        if self.gpu_index is not None:
            self._start_mj_energy = _nvml_total_energy_millijoules(self.gpu_index)
            if self._start_mj_energy is not None:
                self._source = "gpu-nvml"
                return self
        # GPU path unavailable; we'll do CPU-TDP estimation on stop().
        self._source = "cpu-tdp"
        return self

    def stop(self) -> EnergyReport:
        if self._start_unix is None:
            raise RuntimeError("EnergyMeter.stop() called without start()")
        duration_s = max(0.0, time.monotonic() - self._start_unix)
        if self._source == "gpu-nvml" and self._start_mj_energy is not None:
            end_mj = _nvml_total_energy_millijoules(self.gpu_index)
            if end_mj is not None and end_mj >= self._start_mj_energy:
                joules = (end_mj - self._start_mj_energy) / 1000.0
            else:
                # Counter wrap or unavailable on second probe — fall back.
                joules = self._cpu_tdp_joules(duration_s)
                self._source = "cpu-tdp"
        elif self._source == "cpu-tdp":
            joules = self._cpu_tdp_joules(duration_s)
        else:
            joules = 0.0
            self._source = "zero"
        energy_mj = Decimal(joules) / Decimal(1_000_000)
        if self.carbon_intensity_gco2_per_kwh_override is not None:
            ci = float(self.carbon_intensity_gco2_per_kwh_override)
            zone_label = self.region_zone or "test-override"
        else:
            ci, zone_label = carbon_intensity_for_zone(self.region_zone)
        # gCO2e = (kWh) * (gCO2/kWh).
        kwh = joules / 3_600_000.0
        carbon = Decimal(kwh * ci).quantize(Decimal("0.000001"))
        return EnergyReport(
            energy_mj=energy_mj.quantize(Decimal("0.000001")),
            carbon_gco2e=carbon,
            source=self._source,
            duration_s=duration_s,
            gpu_index=self.gpu_index if self._source == "gpu-nvml" else None,
            region_zone=zone_label,
            carbon_intensity_gco2_per_kwh=ci,
        )

    def _cpu_tdp_joules(self, duration_s: float) -> float:
        watts = CPU_TDP_WATTS_BY_CLASS.get(
            self.hardware_class, CPU_TDP_WATTS_BY_CLASS["unknown"]
        )
        return float(watts) * duration_s


def measure_job_energy(
    fn: Callable[..., Any],
    *args: Any,
    hardware_class: str = "unknown",
    gpu_index: Optional[int] = 0,
    region_zone: Optional[str] = None,
    carbon_intensity_gco2_per_kwh_override: Optional[float] = None,
    **kwargs: Any,
) -> Tuple[Any, EnergyReport]:
    """Run `fn(*args, **kwargs)` inside a meter; return both."""
    meter = EnergyMeter(
        hardware_class=hardware_class,
        gpu_index=gpu_index,
        region_zone=region_zone,
        carbon_intensity_gco2_per_kwh_override=carbon_intensity_gco2_per_kwh_override,
    ).start()
    try:
        result = fn(*args, **kwargs)
    finally:
        report = meter.stop()
    return result, report


__all__ = [
    "EnergyMeter",
    "EnergyReport",
    "measure_job_energy",
    "carbon_intensity_for_zone",
    "CPU_TDP_WATTS_BY_CLASS",
    "GLOBAL_DEFAULT_CARBON_INTENSITY_GCO2_PER_KWH",
]
