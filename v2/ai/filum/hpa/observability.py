"""Prometheus-compatible metrics exposition.

Emits the standard Prometheus text-exposition format for scraping
by a Prometheus server, Grafana Agent, or any compatible collector.
We don't depend on the Prometheus client library — the format is a
6-line spec and we own ~20 metrics, so a 100-line generator is
cheaper than a 50 KiB dependency.

Metrics families exposed:

* ``pluginfer_hpa_pressure``                 (gauge, labels=metric)
* ``pluginfer_hpa_yields_total``             (counter)
* ``pluginfer_hpa_oom_recoveries_total``     (counter)
* ``pluginfer_nbgga_grains_received_total``  (counter)
* ``pluginfer_nbgga_grains_applied_total``   (counter)
* ``pluginfer_nbgga_grains_rejected_total``  (counter)
* ``pluginfer_nbgga_versions_emitted_total`` (counter)
* ``pluginfer_transport_packets_sent_total``     (counter)
* ``pluginfer_transport_packets_received_total`` (counter)
* ``pluginfer_transport_grains_assembled_total`` (counter)
* ``pluginfer_transport_grains_duplicate_total`` (counter)
* ``pluginfer_transport_retries_sent_total``     (counter)
* ``pluginfer_gossip_grains_forwarded_total``    (counter)
* ``pluginfer_gossip_peers``                 (gauge, labels=state)
* ``pluginfer_safety_decisions_total``       (counter, labels=decision)
* ``pluginfer_market_volume_tflop_hr``       (gauge)
* ``pluginfer_market_clearing_price``        (gauge)

Plus dynamic metrics from any callable registered via
``MetricsRegistry.register(name, fn)``.

Public API::

    registry = MetricsRegistry()
    registry.bind_pressure_sampler(sampler)
    registry.bind_nbgga(nbgga)
    registry.bind_transport(tx)
    registry.bind_gossip(gossip)
    registry.bind_safety_gate(gate)
    text = registry.render()    # send to /metrics handler
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional


# ---------- generic samples + render ----------------------------------------

def _fmt(name: str, value: Any, labels: Optional[dict] = None) -> str:
    """One-line Prometheus exposition. Caller emits HELP/TYPE separately."""
    if labels:
        kv = ",".join(f'{k}="{_esc(v)}"' for k, v in labels.items())
        return f"{name}{{{kv}}} {value}"
    return f"{name} {value}"


def _esc(v: Any) -> str:
    s = str(v)
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# ---------- registry --------------------------------------------------------

class MetricsRegistry:
    def __init__(self):
        self._sampler = None
        self._nbgga = None
        self._tx = None
        self._gossip = None
        self._safety = None
        self._market = None
        self._extras: list[Callable[[], str]] = []
        self._lock = threading.Lock()

    # binders — none required; render only emits what's bound.

    def bind_pressure_sampler(self, sampler) -> None:
        with self._lock:
            self._sampler = sampler

    def bind_nbgga(self, nbgga) -> None:
        with self._lock:
            self._nbgga = nbgga

    def bind_transport(self, tx) -> None:
        with self._lock:
            self._tx = tx

    def bind_gossip(self, gossip) -> None:
        with self._lock:
            self._gossip = gossip

    def bind_safety_gate(self, gate) -> None:
        with self._lock:
            self._safety = gate

    def bind_market_report(self, report_or_fn) -> None:
        """Either a static EpochClearReport or a callable returning one."""
        with self._lock:
            self._market = report_or_fn

    def register(self, fn: Callable[[], str]) -> None:
        """Register a custom metrics-block emitter. Free-form."""
        with self._lock:
            self._extras.append(fn)

    # ---- render --------------------------------------------------------

    def render(self) -> str:
        lines: list[str] = []
        lines.append("# Pluginfer HPA-LRD metrics")
        lines.append(f"# generated_ts {time.time()}")

        if self._sampler is not None:
            lines.extend(self._render_pressure())
        if self._nbgga is not None:
            lines.extend(self._render_nbgga())
        if self._tx is not None:
            lines.extend(self._render_transport())
        if self._gossip is not None:
            lines.extend(self._render_gossip())
        if self._safety is not None:
            lines.extend(self._render_safety())
        if self._market is not None:
            lines.extend(self._render_market())
        for fn in list(self._extras):
            try:
                lines.append(fn())
            except Exception:
                continue
        return "\n".join(lines) + "\n"

    # ---- per-binding renderers -----------------------------------------

    def _render_pressure(self) -> list[str]:
        s = self._sampler.last()
        out = [
            "# HELP pluginfer_hpa_pressure Hardware pressure scalar 0..1",
            "# TYPE pluginfer_hpa_pressure gauge",
        ]
        for k in ("vram_used_frac", "gpu_util_frac", "gpu_temp_frac",
                   "ram_used_frac", "cpu_used_frac"):
            v = getattr(s, k, -1.0)
            out.append(_fmt("pluginfer_hpa_pressure", v if v >= 0 else 0.0,
                             {"metric": k}))
        return out

    def _render_nbgga(self) -> list[str]:
        st = self._nbgga.stats
        out = [
            "# HELP pluginfer_nbgga_grains_received_total Grains received from peers + local",
            "# TYPE pluginfer_nbgga_grains_received_total counter",
            _fmt("pluginfer_nbgga_grains_received_total", st.grains_received),
            "# HELP pluginfer_nbgga_grains_applied_total Grains successfully merged",
            "# TYPE pluginfer_nbgga_grains_applied_total counter",
            _fmt("pluginfer_nbgga_grains_applied_total", st.grains_applied),
            "# HELP pluginfer_nbgga_grains_rejected_total Grains rejected at signature/shape",
            "# TYPE pluginfer_nbgga_grains_rejected_total counter",
            _fmt("pluginfer_nbgga_grains_rejected_total", st.grains_rejected),
            "# HELP pluginfer_nbgga_grains_evicted_total Grains evicted as too stale",
            "# TYPE pluginfer_nbgga_grains_evicted_total counter",
            _fmt("pluginfer_nbgga_grains_evicted_total", st.grains_evicted),
            "# HELP pluginfer_nbgga_versions_emitted_total New shard versions",
            "# TYPE pluginfer_nbgga_versions_emitted_total counter",
            _fmt("pluginfer_nbgga_versions_emitted_total", st.versions_emitted),
        ]
        return out

    def _render_transport(self) -> list[str]:
        st = self._tx.stats
        out = [
            "# HELP pluginfer_transport_packets_sent_total UDP datagrams sent",
            "# TYPE pluginfer_transport_packets_sent_total counter",
            _fmt("pluginfer_transport_packets_sent_total", st.packets_sent),
            "# HELP pluginfer_transport_packets_received_total UDP datagrams received",
            "# TYPE pluginfer_transport_packets_received_total counter",
            _fmt("pluginfer_transport_packets_received_total", st.packets_received),
            "# HELP pluginfer_transport_grains_assembled_total Grains fully reassembled",
            "# TYPE pluginfer_transport_grains_assembled_total counter",
            _fmt("pluginfer_transport_grains_assembled_total", st.grains_assembled),
            "# HELP pluginfer_transport_grains_duplicate_total Grains dropped by dedup ring",
            "# TYPE pluginfer_transport_grains_duplicate_total counter",
            _fmt("pluginfer_transport_grains_duplicate_total", st.grains_duplicate),
            "# HELP pluginfer_transport_retries_sent_total Retransmissions",
            "# TYPE pluginfer_transport_retries_sent_total counter",
            _fmt("pluginfer_transport_retries_sent_total", st.retries_sent),
        ]
        return out

    def _render_gossip(self) -> list[str]:
        st = self._gossip.stats
        peers = self._gossip.all_peers()
        by_state: dict[str, int] = {}
        for p in peers:
            by_state[p.state] = by_state.get(p.state, 0) + 1
        out = [
            "# HELP pluginfer_gossip_grains_forwarded_total Grains forwarded to peers",
            "# TYPE pluginfer_gossip_grains_forwarded_total counter",
            _fmt("pluginfer_gossip_grains_forwarded_total", st.grains_forwarded),
            "# HELP pluginfer_gossip_pings_sent_total Failure-detection pings",
            "# TYPE pluginfer_gossip_pings_sent_total counter",
            _fmt("pluginfer_gossip_pings_sent_total", st.pings_sent),
            "# HELP pluginfer_gossip_peers Membership table count by state",
            "# TYPE pluginfer_gossip_peers gauge",
        ]
        for state in ("alive", "suspect", "dead"):
            out.append(_fmt("pluginfer_gossip_peers",
                             by_state.get(state, 0), {"state": state}))
        return out

    def _render_safety(self) -> list[str]:
        st = self._safety.stats()
        out = [
            "# HELP pluginfer_safety_decisions_total Safety-gate decisions",
            "# TYPE pluginfer_safety_decisions_total counter",
        ]
        for k in ("allowed", "denied", "rate_limited", "quarantined"):
            out.append(_fmt("pluginfer_safety_decisions_total",
                             st.get(k, 0), {"decision": k}))
        return out

    def _render_market(self) -> list[str]:
        rep = self._market() if callable(self._market) else self._market
        if rep is None:
            return []
        out = [
            "# HELP pluginfer_market_volume_tflop_hr Total matched volume",
            "# TYPE pluginfer_market_volume_tflop_hr gauge",
            _fmt("pluginfer_market_volume_tflop_hr",
                  getattr(rep, "total_volume_tflop_hr", 0.0)),
            "# HELP pluginfer_market_clearing_price Average clearing price",
            "# TYPE pluginfer_market_clearing_price gauge",
            _fmt("pluginfer_market_clearing_price",
                  getattr(rep, "avg_clearing_price", 0.0)),
            "# HELP pluginfer_market_multiplier Time-of-use multiplier in effect",
            "# TYPE pluginfer_market_multiplier gauge",
            _fmt("pluginfer_market_multiplier",
                  getattr(rep, "multiplier_used", 1.0)),
        ]
        return out
