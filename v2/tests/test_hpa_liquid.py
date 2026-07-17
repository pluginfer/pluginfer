"""§C Liquid Intelligence Layer — CPU smoke tests.

Covers:
* Grain sign / verify / serialize / staleness decay
* NBGGA feed / snapshot / cursor persistence
* NBGGA CRDT property: same final state regardless of merge order
* SunElection.elect_local_suns + role_for_self
* PlanetLink fail-over to next-nearest Sun
* clear_epoch reverse-auction match (provider-optimal Gale-Shapley)
* TimeOfUseCurve surge / discount monotonicity
* ProviderEarnings bonded curve (cold-start half-rate)

All tests are CPU-only and avoid torch / GPU. The §C bundle is
designed to be CPU-inspectable end to end so the protocol can be
audited without specialised hardware.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest


# ---------- §C1 + §C4: Grain ------------------------------------------------

def test_grain_sign_and_verify_roundtrip():
    from ai.filum.hpa.grain import Grain, GrainMeta, fresh_keypair

    seed, pub = fresh_keypair()
    g = Grain(meta=GrainMeta(
        model_shard_id="layer.0.attn.q",
        version_v=10,
        contributor_id="alice",
        optimizer_seed=42,
        pressure_at_birth=0.4,
        shape_m=64, shape_n=32, shape_r=8,
    ))
    g.grad_bytes = np.zeros((64, 8), dtype="float32").tobytes()
    g.sign(seed)

    assert g.signature, "signature must be non-empty"
    assert g.verify(pub), "signature must verify"
    # Tampering breaks verification.
    g.grad_bytes = np.ones((64, 8), dtype="float32").tobytes()
    assert not g.verify(pub), "tampered grain must NOT verify"


def test_grain_serialise_roundtrip():
    from ai.filum.hpa.grain import Grain, GrainMeta, fresh_keypair

    seed, pub = fresh_keypair()
    g = Grain(meta=GrainMeta(
        model_shard_id="x", version_v=3,
        contributor_id="bob", optimizer_seed=1,
        pressure_at_birth=0.7,
        shape_m=8, shape_n=8, shape_r=2,
    ))
    g.grad_bytes = np.arange(16, dtype="float32").tobytes()
    g.sign(seed)
    blob = g.to_bytes()
    g2 = Grain.from_bytes(blob)
    assert g2.meta.model_shard_id == "x"
    assert g2.meta.version_v == 3
    assert g2.grad_bytes == g.grad_bytes
    assert g2.verify(pub)


def test_grain_staleness_and_decay():
    from ai.filum.hpa.grain import Grain, GrainMeta

    g = Grain(meta=GrainMeta(version_v=5))
    assert g.staleness(current_v=10) == 5
    w_close = g.decay_weight(current_v=5, tau=200.0)
    w_far   = g.decay_weight(current_v=400, tau=200.0)
    assert w_close == 1.0
    assert 0.0 < w_far < 0.2


def test_make_grain_from_numpy():
    from ai.filum.hpa.grain import make_grain
    grad = np.random.randn(64, 8).astype("float32")
    g = make_grain(
        model_shard_id="W1", version_v=0, contributor_id="alice",
        optimizer_seed=0, pressure=0.2, grad_low_rank=grad,
        full_shape=(64, 32),
    )
    assert g.meta.shape_m == 64
    assert g.meta.shape_n == 32
    assert g.meta.shape_r == 8
    assert g.grad_bytes == grad.tobytes()


# ---------- §C5: Non-Blocking Global Gradient Aggregator --------------------

def _make_grain_for(shard, v, p, arr):
    from ai.filum.hpa.grain import make_grain, fresh_keypair
    seed, pub = fresh_keypair()
    g = make_grain(
        model_shard_id=shard, version_v=v, contributor_id="x",
        optimizer_seed=0, pressure=p, grad_low_rank=arr,
        full_shape=arr.shape,
    )
    g.sign(seed)
    return g


def test_nbgga_feeds_and_accumulates(tmp_path: Path):
    from ai.filum.hpa.global_aggregator import (
        NonBlockingGlobalAggregator, AggregatorPolicy,
    )

    nbgga = NonBlockingGlobalAggregator(
        tmp_path,
        policy=AggregatorPolicy(
            tau=200.0, version_bump_norm=1e9,  # disable version bumps for this test
        ),
    )
    arr = np.ones((4, 4), dtype="float32")
    g = _make_grain_for("L1", 0, 0.2, arr)
    assert nbgga.feed(g)
    assert nbgga.stats.grains_applied == 1

    snap = nbgga.snapshot("L1")
    assert snap is not None
    # Pending should be ~ (1 - 0.2) * 1.0 * arr = 0.8 * ones
    # but snapshot returns running, which is still zero (no version bump).
    assert np.allclose(snap, np.zeros((4, 4)))


def test_nbgga_crdt_property_order_invariant(tmp_path: Path):
    """Same set of grains in different orders -> same final state."""
    from ai.filum.hpa.global_aggregator import (
        NonBlockingGlobalAggregator, AggregatorPolicy,
    )

    rng = np.random.default_rng(7)
    grains = [
        _make_grain_for("L1", v=v, p=0.3, arr=rng.standard_normal((4, 4)).astype("float32"))
        for v in range(8)
    ]

    pol = AggregatorPolicy(tau=200.0, version_bump_norm=1e9)
    nbgga_a = NonBlockingGlobalAggregator(tmp_path / "a", policy=pol)
    nbgga_b = NonBlockingGlobalAggregator(tmp_path / "b", policy=pol)

    # Order A: forward
    for g in grains:
        nbgga_a.feed(g)
    # Order B: reverse
    for g in reversed(grains):
        nbgga_b.feed(g)

    sa = nbgga_a.snapshot("L1")
    sb = nbgga_b.snapshot("L1")
    # Accumulator only commits to running on version bump; for a test
    # of the CRDT property we look at the *pending* via internal state.
    # Both must have the same in-flight delta after applying the same
    # grain set.
    pa = nbgga_a._state["L1"].pending
    pb = nbgga_b._state["L1"].pending
    # Decay weights depend only on (version_v, current shard version_v,
    # pressure) so order shouldn't matter as long as version_v bump
    # didn't fire.
    assert np.allclose(pa, pb, atol=1e-5)


def test_nbgga_drops_overstale_grain(tmp_path: Path):
    from ai.filum.hpa.global_aggregator import (
        NonBlockingGlobalAggregator, AggregatorPolicy,
    )

    pol = AggregatorPolicy(
        tau=10.0, eviction_horizon_tau=2.0, version_bump_norm=0.001,
    )
    nbgga = NonBlockingGlobalAggregator(tmp_path, policy=pol)

    # Bump shard version a few times via fresh grains.
    for v in range(0, 50):
        g = _make_grain_for("L1", v=v, p=0.0,
                            arr=np.full((4, 4), 0.5, dtype="float32"))
        nbgga.feed(g)
    cur = nbgga.current_version("L1")
    assert cur > 0

    # Now feed a very-stale grain. Should be evicted.
    very_stale = _make_grain_for("L1", v=0, p=0.0,
                                 arr=np.ones((4, 4), dtype="float32"))
    res = nbgga.feed(very_stale)
    assert res is False
    assert nbgga.stats.grains_evicted >= 1


def test_nbgga_cursor_survives_restart(tmp_path: Path):
    from ai.filum.hpa.global_aggregator import (
        NonBlockingGlobalAggregator, AggregatorPolicy,
    )

    pol = AggregatorPolicy(version_bump_norm=0.0001)
    nbgga = NonBlockingGlobalAggregator(tmp_path, policy=pol)
    g = _make_grain_for("L1", 0, 0.2, np.ones((4, 4), dtype="float32"))
    nbgga.feed(g)
    nbgga.tick()

    # Reopen at the same path.
    nbgga2 = NonBlockingGlobalAggregator(tmp_path, policy=pol)
    assert "L1" in nbgga2.shard_ids()
    # Version should have advanced because version_bump_norm is tiny.
    assert nbgga2.current_version("L1") >= 1


# ---------- §C2: Sun-Planet election ---------------------------------------

def test_sun_election_picks_top_stability():
    from ai.filum.hpa.sun_election import (
        SunElection, NodeMembership, SunElectionPolicy,
    )

    self_view = NodeMembership(
        node_id="me", stability_score=0.9, last_seen_ts=time.time(),
    )
    peers = [
        NodeMembership(
            node_id=f"p{i}", stability_score=s,
            last_seen_ts=time.time(),
            latency_ms_to_self=10 * i,
        ) for i, s in enumerate([0.95, 0.5, 0.85, 0.2, 0.75])
    ]
    elect = SunElection(SunElectionPolicy(k_sun=3))
    res = elect.elect_local_suns(self_view, peers)

    assert len(res.suns) == 3
    sids = [s.node_id for s in res.suns]
    # Top three by stability among {me=0.9, p0=0.95, p1=0.5, p2=0.85, p3=0.2, p4=0.75}
    assert "p0" in sids
    assert "me" in sids
    assert "p2" in sids

    role = elect.role_for_self(res, "me")
    assert role == "sun"


def test_sun_election_demotes_unstable():
    from ai.filum.hpa.sun_election import (
        SunElection, NodeMembership, SunElectionPolicy,
    )

    self_view = NodeMembership(node_id="me", stability_score=0.1,
                               last_seen_ts=time.time())
    peers = [
        NodeMembership(node_id="p1", stability_score=0.05,
                       last_seen_ts=time.time()),
    ]
    elect = SunElection(SunElectionPolicy(k_sun=3, s_cut=0.3))
    res = elect.elect_local_suns(self_view, peers)
    # All below s_cut; fallback to top-1.
    assert len(res.suns) == 1


def test_planet_link_falls_back():
    from ai.filum.hpa.sun_election import (
        SunElection, NodeMembership, PlanetLink, ElectionResult,
    )

    suns = [
        NodeMembership(node_id="s1", stability_score=0.9),
        NodeMembership(node_id="s2", stability_score=0.85),
    ]
    res = ElectionResult(suns=suns)
    link = PlanetLink(SunElection(), self_id="planet")
    link.update_election(res)
    assert link.primary_sun().node_id == "s1"
    link.report_sun_failure("s1")
    assert link.primary_sun().node_id == "s2"


# ---------- §C7 + §C8: reverse auction --------------------------------------

def test_clear_epoch_matches_supply_and_demand():
    from ai.filum.hpa.reverse_auction import (
        ProviderBid, BuyerAsk, clear_epoch, TimeOfUseCurve,
    )

    bids = [
        ProviderBid(provider_id="p1", price_per_tflop_hr=0.10,
                    capacity_tflop_hr=10, available_until_ts=1e12,
                    stability_score=0.9, energy_source="green"),
        ProviderBid(provider_id="p2", price_per_tflop_hr=0.20,
                    capacity_tflop_hr=10, available_until_ts=1e12,
                    stability_score=0.8, energy_source="grid"),
    ]
    asks = [
        BuyerAsk(buyer_id="b1", needed_tflop_hr=5,
                 max_price_per_tflop_hr=0.30, deadline_ts=1e12,
                 min_reliability=0.5),
    ]
    tou = TimeOfUseCurve()
    rep = clear_epoch(bids, asks, tou=tou)

    assert len(rep.matches) >= 1
    # Should prefer p1 (cheaper).
    assert rep.matches[0].bid.provider_id == "p1"
    assert rep.matches[0].matched_tflop_hr == 5
    assert rep.total_volume_tflop_hr == 5


def test_clear_epoch_respects_min_reliability():
    from ai.filum.hpa.reverse_auction import (
        ProviderBid, BuyerAsk, clear_epoch,
    )

    bids = [
        ProviderBid(provider_id="cheap_flaky", price_per_tflop_hr=0.05,
                    capacity_tflop_hr=10, available_until_ts=1e12,
                    stability_score=0.2),
        ProviderBid(provider_id="expensive_stable", price_per_tflop_hr=0.50,
                    capacity_tflop_hr=10, available_until_ts=1e12,
                    stability_score=0.95),
    ]
    asks = [
        BuyerAsk(buyer_id="b1", needed_tflop_hr=5,
                 max_price_per_tflop_hr=1.00, deadline_ts=1e12,
                 min_reliability=0.8),
    ]
    rep = clear_epoch(bids, asks)
    # cheap_flaky must be filtered out.
    assert all(m.bid.provider_id == "expensive_stable" for m in rep.matches)


def test_clear_epoch_green_routing():
    from ai.filum.hpa.reverse_auction import (
        ProviderBid, BuyerAsk, clear_epoch,
    )

    bids = [
        ProviderBid(provider_id="grid_cheap", price_per_tflop_hr=0.05,
                    capacity_tflop_hr=10, available_until_ts=1e12,
                    energy_source="grid"),
        ProviderBid(provider_id="green_premium", price_per_tflop_hr=0.20,
                    capacity_tflop_hr=10, available_until_ts=1e12,
                    energy_source="green"),
    ]
    asks = [
        BuyerAsk(buyer_id="green_buyer", needed_tflop_hr=5,
                 max_price_per_tflop_hr=1.0, deadline_ts=1e12,
                 energy_preference="green"),
    ]
    rep = clear_epoch(bids, asks)
    assert all(m.bid.energy_source == "green" for m in rep.matches)


def test_time_of_use_curve_monotonic():
    from ai.filum.hpa.reverse_auction import TimeOfUseCurve

    tou = TimeOfUseCurve()
    # More asks than bids -> surge.
    surge = tou.multiplier(asks_count=100, bids_count=10)
    # More bids than asks -> discount.
    discount = tou.multiplier(asks_count=10, bids_count=100)
    # Balanced -> neutral.
    neutral = tou.multiplier(asks_count=10, bids_count=10)

    assert surge > 1.0
    assert discount < 1.0
    assert abs(neutral - 1.0) < 0.01


def test_provider_earnings_cold_start_half_rate():
    from ai.filum.hpa.reverse_auction import ProviderEarnings

    new_provider = ProviderEarnings(
        base_compute_credit=100.0,
        stability_score=1.0,
        attestation_score=0.0,    # cold start
    )
    veteran = ProviderEarnings(
        base_compute_credit=100.0,
        stability_score=1.0,
        attestation_score=1.0,    # fully attested
    )
    # Cold start earns half (0.5 + 0.5*0 = 0.5); veteran earns full.
    assert new_provider.total() == pytest.approx(50.0)
    assert veteran.total() == pytest.approx(100.0)
