"""Prometheus-text metrics for Pluginfer.

Hand-rolled exposition (no `prometheus-client` dep) so the seed-node
container stays small and the surface remains a 50-line file -- not a
dependency tree. Every counter / gauge / histogram serializes to the
official Prometheus text-format spec at:
    https://prometheus.io/docs/instrumenting/exposition_formats/

Usage from the API layer:

    from core.metrics import REGISTRY, jobs_total, job_duration_seconds
    jobs_total.inc(labels={"status": "completed"})
    with job_duration_seconds.time():
        ...

    # In a router:
    return Response(REGISTRY.render(), media_type="text/plain; version=0.0.4")
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def _esc(s: str) -> str:
    """Escape label values per Prometheus text-format spec."""
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels_to_str(labels: Optional[Dict[str, str]]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_esc(str(v))}"' for k, v in sorted(labels.items()))
    return "{" + inner + "}"


@dataclass
class _MetricBase:
    name: str
    help: str
    metric_type: str = "gauge"
    _values: Dict[Tuple[Tuple[str, str], ...], float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _key(self, labels: Optional[Dict[str, str]]) -> Tuple[Tuple[str, str], ...]:
        return tuple(sorted((labels or {}).items()))

    def render(self) -> str:
        out = [f"# HELP {self.name} {self.help}",
               f"# TYPE {self.name} {self.metric_type}"]
        with self._lock:
            for k, v in sorted(self._values.items()):
                ldict = dict(k)
                out.append(f"{self.name}{_labels_to_str(ldict)} {v}")
        return "\n".join(out)


class Counter(_MetricBase):
    """Monotonically-increasing counter."""

    def __init__(self, name: str, help: str) -> None:
        super().__init__(name=name, help=help, metric_type="counter")

    def inc(self, n: float = 1.0, *, labels: Optional[Dict[str, str]] = None) -> None:
        if n < 0:
            raise ValueError("Counter.inc must be non-negative")
        k = self._key(labels)
        with self._lock:
            self._values[k] = self._values.get(k, 0.0) + n


class Gauge(_MetricBase):
    """Set-able gauge."""

    def __init__(self, name: str, help: str) -> None:
        super().__init__(name=name, help=help, metric_type="gauge")

    def set(self, v: float, *, labels: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            self._values[self._key(labels)] = float(v)

    def inc(self, n: float = 1.0, *, labels: Optional[Dict[str, str]] = None) -> None:
        k = self._key(labels)
        with self._lock:
            self._values[k] = self._values.get(k, 0.0) + n

    def dec(self, n: float = 1.0, *, labels: Optional[Dict[str, str]] = None) -> None:
        self.inc(-n, labels=labels)


@dataclass
class Histogram:
    """Cumulative-bucket histogram. `buckets` is a sorted, ascending list
    of bucket upper bounds (inclusive); `+Inf` is added implicitly."""

    name: str
    help: str
    buckets: List[float]
    metric_type: str = "histogram"
    _bucket_counts: Dict[Tuple[Tuple[str, str], ...], List[int]] = field(default_factory=lambda: defaultdict(list))
    _sums: Dict[Tuple[Tuple[str, str], ...], float] = field(default_factory=lambda: defaultdict(float))
    _counts: Dict[Tuple[Tuple[str, str], ...], int] = field(default_factory=lambda: defaultdict(int))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if not all(self.buckets[i] <= self.buckets[i + 1] for i in range(len(self.buckets) - 1)):
            raise ValueError("Histogram buckets must be non-decreasing")

    def _key(self, labels: Optional[Dict[str, str]]) -> Tuple[Tuple[str, str], ...]:
        return tuple(sorted((labels or {}).items()))

    def observe(self, v: float, *, labels: Optional[Dict[str, str]] = None) -> None:
        k = self._key(labels)
        with self._lock:
            counts = self._bucket_counts.get(k)
            if counts is None:
                counts = [0] * len(self.buckets)
                self._bucket_counts[k] = counts
            for i, bound in enumerate(self.buckets):
                if v <= bound:
                    counts[i] += 1
            self._sums[k] += v
            self._counts[k] += 1

    def time(self, *, labels: Optional[Dict[str, str]] = None):
        return _Timer(self, labels)

    def render(self) -> str:
        out = [f"# HELP {self.name} {self.help}",
               f"# TYPE {self.name} {self.metric_type}"]
        with self._lock:
            for k in sorted(self._bucket_counts):
                ldict = dict(k)
                # _bucket_counts already stores cumulative counts:
                # `observe()` bumps every bucket where v <= bound.
                # Just emit them in order.
                for i, bound in enumerate(self.buckets):
                    bls = dict(ldict, le=str(bound))
                    out.append(f"{self.name}_bucket{_labels_to_str(bls)} {self._bucket_counts[k][i]}")
                out.append(f"{self.name}_bucket{_labels_to_str(dict(ldict, le='+Inf'))} {self._counts[k]}")
                out.append(f"{self.name}_sum{_labels_to_str(ldict)} {self._sums[k]}")
                out.append(f"{self.name}_count{_labels_to_str(ldict)} {self._counts[k]}")
        return "\n".join(out)


class _Timer:
    def __init__(self, hist: Histogram, labels: Optional[Dict[str, str]]) -> None:
        self._h = hist
        self._labels = labels
        self._t0 = 0.0

    def __enter__(self) -> "_Timer":
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *exc) -> None:
        self._h.observe(time.monotonic() - self._t0, labels=self._labels)


@dataclass
class Registry:
    metrics: List[object] = field(default_factory=list)

    def register(self, m) -> None:
        self.metrics.append(m)

    def render(self) -> str:
        return "\n".join(m.render() for m in self.metrics) + "\n"


# ---------------------------------------------------------------------------
# Default registry + standard Pluginfer metrics
# ---------------------------------------------------------------------------

REGISTRY = Registry()

jobs_total = Counter(
    "pluginfer_jobs_total",
    "Total Pluginfer jobs partitioned by terminal status.",
)
job_duration_seconds = Histogram(
    "pluginfer_job_duration_seconds",
    "Time from job submit to terminal state.",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 300],
)
auction_duration_seconds = Histogram(
    "pluginfer_auction_duration_seconds",
    "Time from auction start to winner selection.",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)
peers_connected = Gauge(
    "pluginfer_peers_connected",
    "Number of peers in the active routing table.",
)
chain_height = Gauge(
    "pluginfer_chain_height",
    "Height of the local blockchain (max block index seen).",
)
balance_plg = Gauge(
    "pluginfer_balance_plg",
    "Current PLG balance of the operator wallet.",
)
uptime_seconds = Gauge(
    "pluginfer_uptime_seconds",
    "Seconds since this node process started.",
)

for _m in (jobs_total, job_duration_seconds, auction_duration_seconds,
           peers_connected, chain_height, balance_plg, uptime_seconds):
    REGISTRY.register(_m)
