"""Sun-Planet Pressure-Weighted Consensus — §C2.

The mesh has no master. It has *Suns* — locally-elected, low-pressure
nodes that aggregate gradients from a regional ring of high-pressure
*Planets*. Suns then gossip aggregated state to other Suns globally,
forming a small (~100-node) Sun-of-Suns ring on which classical BFT
is feasible and cheap.

Rules:

* Every node maintains a smoothed *stability score* S = EMA(1 - P)
  over a sliding window. Lowest pressure → highest S → most stable.
* Within an ε-neighbourhood (geographic / latency cluster of
  ~K_local nodes, default 100), the K_sun nodes (default 3) with
  highest S are elected as Suns. K=3 lets the ring tolerate one Sun
  failure with 2-of-3 majority on attestations.
* Election runs continuously: every M_election seconds (default 60)
  AND on any current Sun's S dropping below cut threshold S_cut.
* Sun-of-Suns ring: Suns gossip every M_global seconds (default
  300). The Sun-of-Suns is itself elected from the Suns by stability.
* Planets stream gradients only to their nearest Sun by latency.
* If a Planet's Sun goes down, the Planet retries with the
  next-nearest Sun in its membership view.

This module is *protocol-pure* — it has no opinion on the transport.
It exposes ``elect_local_suns(members)`` which a TaskRouter or any
other transport can call to make routing decisions. Production
deployment composes this with `core.task_router.TaskRouter`.

The §C2 invention is that elections are driven by *measured hardware
pressure*, not by stake or randomness. Stability is a property the
hardware *demonstrates* — it cannot be bought, only earned by
running uninterrupted.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass
class NodeMembership:
    """A view of a node from another node's membership table.

    ``stability_score`` is the smoothed (1 - P) value computed locally
    by the node and gossiped to peers.
    """
    node_id: str
    public_key: bytes = b""
    stability_score: float = 0.0
    last_seen_ts: float = 0.0
    advertised_capacity_tflops: float = 0.0
    region_hint: str = ""           # e.g. "iad", "ams", "sin" — for nearest-Sun
    latency_ms_to_self: float = 1e9


@dataclass
class StabilityEMA:
    """Per-self pressure stability tracker. ``alpha`` controls smoothing."""
    alpha: float = 0.05
    value: float = 0.0
    window_size: int = 1
    started: float = field(default_factory=time.time)

    def update(self, pressure: float) -> float:
        """Feed a new pressure sample; returns the new stability score."""
        s = max(0.0, min(1.0, 1.0 - float(pressure)))
        if self.window_size == 1:
            self.value = s
        else:
            self.value = self.alpha * s + (1.0 - self.alpha) * self.value
        self.window_size += 1
        return self.value


@dataclass
class SunElectionPolicy:
    k_sun: int = 3                  # # of Suns elected per region
    k_local: int = 100              # # of nodes per ε-neighbourhood
    m_election_s: float = 60.0      # election cadence (seconds)
    m_global_s: float = 300.0       # Sun-of-Suns gossip cadence
    s_cut: float = 0.3              # below this, a Sun is forcibly demoted
    s_promote: float = 0.7          # above this, a Planet is promotable
    membership_ttl_s: float = 120.0 # drop members not seen this long


@dataclass
class ElectionResult:
    """Snapshot of an election. Suns is sorted by stability_score desc."""
    suns: list[NodeMembership] = field(default_factory=list)
    planets: list[NodeMembership] = field(default_factory=list)
    quorum_score: float = 0.0       # mean stability of elected suns
    elected_at_ts: float = 0.0


class SunElection:
    """Stateless election function over a membership view.

    Held as a class only so policies travel cleanly with the function.
    """

    def __init__(self, policy: SunElectionPolicy = SunElectionPolicy()):
        self.policy = policy

    def elect_local_suns(
        self,
        self_view: NodeMembership,
        peers: Iterable[NodeMembership],
        now_ts: Optional[float] = None,
    ) -> ElectionResult:
        """Run an election over self + peers. Returns suns + planets."""
        if now_ts is None:
            now_ts = time.time()
        live: list[NodeMembership] = [self_view]
        for p in peers:
            if now_ts - p.last_seen_ts > self.policy.membership_ttl_s:
                continue
            live.append(p)

        # Region partition: bucket by region_hint, then within each region
        # pick the K_local closest by latency.
        region_buckets: dict[str, list[NodeMembership]] = {}
        for n in live:
            region_buckets.setdefault(n.region_hint or "_", []).append(n)

        # For the simplest baseline we treat the whole live set as one
        # region; production will refine by region_buckets.
        cluster = sorted(live, key=lambda n: n.latency_ms_to_self)[
            : self.policy.k_local
        ]

        # Elect: top-K_sun by stability, with tie-break by capacity.
        ranked = sorted(
            cluster,
            key=lambda n: (-n.stability_score, -n.advertised_capacity_tflops),
        )
        suns = ranked[: self.policy.k_sun]
        planets = ranked[self.policy.k_sun:]

        # Demote any Sun whose stability is below s_cut.
        suns = [s for s in suns if s.stability_score >= self.policy.s_cut]
        if not suns:
            # Fall back: highest-S node even if below s_cut, to ensure
            # liveness. Caller can re-trigger an election.
            suns = ranked[:1] if ranked else []

        quorum = (
            sum(s.stability_score for s in suns) / max(1, len(suns))
            if suns else 0.0
        )

        return ElectionResult(
            suns=suns,
            planets=planets,
            quorum_score=quorum,
            elected_at_ts=now_ts,
        )

    def role_for_self(
        self, result: ElectionResult, self_id: str,
    ) -> str:
        """Return ``"sun"`` or ``"planet"`` based on the election result."""
        if any(s.node_id == self_id for s in result.suns):
            return "sun"
        return "planet"


class PlanetLink:
    """A Planet's stable wiring to its nearest Sun.

    Resolves which Sun to stream to; falls back to the next-nearest
    Sun in the membership view if the primary fails.
    """

    def __init__(self, election: SunElection, self_id: str):
        self._election = election
        self._self_id = self_id
        self._cached_result: Optional[ElectionResult] = None
        self._failed_suns: set[str] = set()

    def update_election(self, result: ElectionResult) -> None:
        self._cached_result = result
        # Reset the failed-sun set on a new election; trust the new view.
        self._failed_suns.clear()

    def primary_sun(self) -> Optional[NodeMembership]:
        if self._cached_result is None:
            return None
        for s in self._cached_result.suns:
            if s.node_id in self._failed_suns:
                continue
            if s.node_id == self._self_id:
                continue   # we're a Sun ourselves; this link is unused
            return s
        return None

    def report_sun_failure(self, sun_id: str) -> None:
        """Caller marks a Sun as currently unreachable; we route around."""
        self._failed_suns.add(sun_id)


class SunOfSunsRing:
    """The global ring formed by all elected Suns.

    Sun-of-Suns operates on an order-of-magnitude smaller graph
    (~K_total / K_local nodes). Classical BFT (Tendermint) is cheap
    here. For this module we expose the membership view; the actual
    BFT step is delegated to ``core.bft_consensus.BFTConsensus``.
    """

    def __init__(self):
        self._members: dict[str, NodeMembership] = {}

    def update(self, sun: NodeMembership) -> None:
        self._members[sun.node_id] = sun

    def members(self) -> list[NodeMembership]:
        return list(self._members.values())

    def quorum_size(self) -> int:
        """Tendermint-style 2/3 quorum."""
        n = len(self._members)
        return (2 * n) // 3 + 1

    def stability_weighted_quorum_size(self) -> float:
        """Sum of stability scores >= 2/3 of total."""
        if not self._members:
            return 0.0
        total = sum(m.stability_score for m in self._members.values())
        return (2.0 * total) / 3.0
