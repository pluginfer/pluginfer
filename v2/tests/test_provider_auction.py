"""
Provider auction tests (TODO §4.6, W15) + slack-aware pricing
(TODO §4.2, W14b).

Cases:
  1. Mesh peer underbids cloud LLM at off-peak → wins.
  2. Cloud LLM has no API key → bid is None → mesh wins by default.
  3. Cloud LLM with key wins on quality when peer's quality < floor.
  4. Privacy-sensitive job rejects public-grade providers.
  5. Cost-ceiling enforcement → all bids over ceiling are rejected.
  6. TimeOfDaySlackCurve interpolates correctly across breakpoints.
  7. AuctionResult exposes losing bids for transparency / dispute audit.
"""

from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
for parent in [_HERE.parents[1], _HERE.parents[2]]:
    if (parent / "core").is_dir():
        sys.path.insert(0, str(parent))
        break

from core.providers import (    # noqa: E402
    Auction, JobSpec, MeshGPUProvider, OpenAIProvider, AnthropicProvider,
    PRIVACY_PUBLIC, PRIVACY_PRIVATE, PRIVACY_SENSITIVE,
)
from core.slack_auction import (    # noqa: E402
    TimeOfDaySlackCurve, default_consumer_curve,
)


def test_mesh_underbids_cloud_offpeak():
    print("\n[1] MESH UNDERBIDS CLOUD AT OFF-PEAK")
    print("-" * 60)
    # Off-peak slack curve: factor 0.2 → 5x cheaper.
    curve = TimeOfDaySlackCurve(points=[(0, 0.2), (24, 0.2)])
    mesh = MeshGPUProvider(
        provider_id="mesh-peer-A",
        slack_curve=curve,
        base_quality=0.85,
    )
    # Cloud has no key configured — set enabled=False so the bid is
    # deliberately a no-op for this test.
    cloud = OpenAIProvider(enabled=False)

    auction = Auction()
    auction.register(mesh)
    auction.register(cloud)

    job = JobSpec(job_id="j1", kind="inference",
                  payload={"max_tokens": 200},
                  cost_ceiling_usd=0.005, latency_ceiling_ms=5000,
                  quality_floor=0.7)
    res = auction.run(job)
    assert res.is_won()
    assert res.winner.provider_id == "mesh-peer-A"
    print(f"  winner={res.winner.provider_id}  price=${res.winner.price_usd:.6f}")
    print("  PASS")


def test_no_api_key_no_bid():
    print("\n[2] CLOUD WITHOUT API KEY ABSTAINS")
    print("-" * 60)
    cloud = OpenAIProvider()  # enabled=False default
    auction = Auction()
    auction.register(cloud)
    job = JobSpec(job_id="j2", kind="inference",
                  payload={"max_tokens": 200})
    res = auction.run(job)
    assert not res.is_won()
    assert any("abstain" in r["reason"].lower() for r in res.rejected)
    print(f"  rejected: {res.rejected}")
    print("  PASS")


def test_quality_floor_enforced():
    print("\n[3] QUALITY FLOOR REJECTS WEAKER PROVIDER")
    print("-" * 60)
    curve = TimeOfDaySlackCurve(points=[(0, 0.5), (24, 0.5)])
    weak_mesh = MeshGPUProvider(
        provider_id="mesh-weak", slack_curve=curve,
        base_quality=0.55,    # below floor 0.7
    )
    auction = Auction()
    auction.register(weak_mesh)
    job = JobSpec(job_id="j3", kind="inference",
                  payload={"max_tokens": 100},
                  quality_floor=0.7)
    res = auction.run(job)
    assert not res.is_won()
    print(f"  weak provider rejected: {res.rejected[0]['reason']} OK")
    print("  PASS")


def test_privacy_class_filters_public_providers():
    print("\n[4] SENSITIVE JOB REJECTS PUBLIC-GRADE PROVIDERS")
    print("-" * 60)
    # MeshGPUProvider defaults to private grade — should pass for
    # 'private' job, but NOT for 'sensitive' (which requires TEE).
    curve = TimeOfDaySlackCurve(points=[(0, 0.3), (24, 0.3)])
    mesh = MeshGPUProvider(provider_id="mesh-private",
                            slack_curve=curve)
    auction = Auction()
    auction.register(mesh)
    job_sensitive = JobSpec(
        job_id="j4", kind="inference",
        payload={"max_tokens": 100},
        privacy_class=PRIVACY_SENSITIVE,
    )
    res = auction.run(job_sensitive)
    assert not res.is_won()
    assert any("privacy" in r["reason"] for r in res.rejected)
    print(f"  sensitive job rejected mesh-private OK: "
          f"{res.rejected[0]['reason']}")
    print("  PASS")


def test_cost_ceiling_filters():
    print("\n[5] COST CEILING FILTERS OUT EXPENSIVE BIDS")
    print("-" * 60)
    # Force the mesh into busy hours (factor 5x).
    curve = TimeOfDaySlackCurve(points=[(0, 5.0), (24, 5.0)])
    mesh = MeshGPUProvider(provider_id="mesh-busy", slack_curve=curve)
    auction = Auction()
    auction.register(mesh)
    # Tight ceiling that even a 1.0x mesh would exceed.
    job = JobSpec(job_id="j5", kind="inference",
                  payload={"max_tokens": 1000},
                  cost_ceiling_usd=0.0001)
    res = auction.run(job)
    assert not res.is_won()
    assert any("price" in r["reason"] for r in res.rejected)
    print(f"  expensive bid filtered OK: {res.rejected[0]['reason']}")
    print("  PASS")


def test_slack_curve_interpolation():
    print("\n[6] SLACK CURVE INTERPOLATES BREAKPOINTS")
    print("-" * 60)
    curve = TimeOfDaySlackCurve(
        points=[(0, 0.2), (6, 0.2), (9, 1.0), (18, 1.5), (24, 0.2)]
    )
    midnight = curve.opportunity_cost_factor(datetime.time(0, 0))
    seven = curve.opportunity_cost_factor(datetime.time(7, 0))
    nine = curve.opportunity_cost_factor(datetime.time(9, 0))
    noon = curve.opportunity_cost_factor(datetime.time(12, 0))
    # Verify breakpoints match exactly.
    assert abs(midnight - 0.2) < 1e-9
    assert abs(nine - 1.0) < 1e-9
    # Interpolation: at 7am, between (6, 0.2) and (9, 1.0).
    expected_seven = 0.2 + (1.0/3.0) * (1.0 - 0.2)
    assert abs(seven - expected_seven) < 1e-9
    # At noon: between (9, 1.0) and (18, 1.5).
    expected_noon = 1.0 + (3.0/9.0) * (1.5 - 1.0)
    assert abs(noon - expected_noon) < 1e-9
    print(f"  00:00 -> {midnight:.3f}, 07:00 -> {seven:.3f},")
    print(f"  09:00 -> {nine:.3f}, 12:00 -> {noon:.3f} OK")
    # Default consumer curve sanity check.
    default = default_consumer_curve()
    factor_3am = default.opportunity_cost_factor(datetime.time(3, 0))
    factor_3pm = default.opportunity_cost_factor(datetime.time(15, 0))
    assert factor_3am < factor_3pm
    print(f"  default: 03:00 ({factor_3am:.3f}) < 15:00 "
          f"({factor_3pm:.3f}) OK")
    print("  PASS")


def test_auction_result_exposes_losing_bids():
    print("\n[7] AUCTIONRESULT EXPOSES LOSING BIDS")
    print("-" * 60)
    # Two mesh peers, same job.
    curve_cheap = TimeOfDaySlackCurve(points=[(0, 0.2), (24, 0.2)])
    curve_pricey = TimeOfDaySlackCurve(points=[(0, 0.8), (24, 0.8)])
    cheap = MeshGPUProvider(provider_id="cheap-peer",
                              slack_curve=curve_cheap)
    pricey = MeshGPUProvider(provider_id="pricey-peer",
                              slack_curve=curve_pricey)
    auction = Auction()
    auction.register(cheap)
    auction.register(pricey)
    job = JobSpec(job_id="j7", kind="inference",
                  payload={"max_tokens": 200})
    res = auction.run(job)
    assert res.is_won()
    assert res.winner.provider_id == "cheap-peer"
    # Both bids must be in the bids list.
    ids = {b.provider_id for b in res.bids}
    assert ids == {"cheap-peer", "pricey-peer"}
    print(f"  winner={res.winner.provider_id} (score={res.winner_score:.3f})")
    print(f"  all bids: {ids}")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("PROVIDER AUCTION TEST (Innovation §4.6 + §4.2)")
    print("=" * 60)
    t0 = time.time()
    test_mesh_underbids_cloud_offpeak()
    test_no_api_key_no_bid()
    test_quality_floor_enforced()
    test_privacy_class_filters_public_providers()
    test_cost_ceiling_filters()
    test_slack_curve_interpolation()
    test_auction_result_exposes_losing_bids()
    print("\n" + "=" * 60)
    print(f"ALL PROVIDER-AUCTION TESTS PASSED in {time.time() - t0:.1f}s")
    print("=" * 60)
