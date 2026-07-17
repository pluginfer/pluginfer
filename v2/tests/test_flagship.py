"""G4 — alpha-tier flagship registration + cost estimator."""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from api.jobs_service import JobsService  # noqa: E402
from core.flagship import (  # noqa: E402
    ALPHA_FLAGSHIPS,
    FlagshipModelSpec,
    FlagshipProvider,
    estimate_training_cost_usd,
    register_alpha_flagship,
)
from core.providers import Auction, JobSpec  # noqa: E402


# ---------------------------------------------------------------------------
# Catalogue + spec
# ---------------------------------------------------------------------------

def test_alpha_flagships_catalogue_present_and_licence_safe():
    assert len(ALPHA_FLAGSHIPS) >= 3
    # No GPL / AGPL — those would force Pluginfer to open-source
    # the inference stack which we're not ready for.
    for s in ALPHA_FLAGSHIPS:
        assert "GPL" not in s.licence.upper(), s.model_id


def test_flagship_spec_to_receipt_field_shape():
    s = ALPHA_FLAGSHIPS[0]
    d = s.to_receipt_model_field()
    assert d["id"] == s.model_id
    assert d["licence"] == s.licence
    assert d["parameter_count_b"] == s.parameter_count_b


# ---------------------------------------------------------------------------
# Registration + auction
# ---------------------------------------------------------------------------

def test_register_alpha_flagship_wins_an_llm_auction():
    auction = Auction()
    svc = JobsService(auction=auction)

    def runner(prompt: str, payload: dict) -> bytes:
        return f"flagship-reply: {prompt[:40]}".encode("utf-8")

    provider = register_alpha_flagship(
        jobs_service=svc,
        spec=ALPHA_FLAGSHIPS[0],
        runner_fn=runner,
    )

    spec = JobSpec(
        job_id="job1",
        kind="llm.completion",
        payload={"prompt": "hi", "max_tokens": 16},
        cost_ceiling_usd=1.0,
        latency_ceiling_ms=10_000,
        privacy_class="private",
        quality_floor=0.6,
    )
    result = auction.run(spec)
    assert result.is_won()
    assert result.winner.provider_id == provider.provider_id
    assert result.winner.evidence["tier"] == "alpha-flagship"


def test_register_alpha_flagship_does_not_bid_on_non_llm_kinds():
    auction = Auction()
    svc = JobsService(auction=auction)

    def runner(prompt: str, payload: dict) -> bytes:
        return b"x"
    register_alpha_flagship(jobs_service=svc, spec=ALPHA_FLAGSHIPS[0],
                            runner_fn=runner)

    spec = JobSpec(
        job_id="job1",
        kind="compute.test",
        payload={"prompt": "anything", "max_tokens": 16},
        cost_ceiling_usd=1.0,
        latency_ceiling_ms=10_000,
        privacy_class="public",
        quality_floor=0.5,
    )
    result = auction.run(spec)
    assert not result.is_won()


def test_flagship_execute_signs_result_hash():
    auction = Auction()
    svc = JobsService(auction=auction)

    def runner(prompt: str, payload: dict) -> bytes:
        return b"hello-from-flagship"
    p = register_alpha_flagship(
        jobs_service=svc, spec=ALPHA_FLAGSHIPS[0], runner_fn=runner,
    )

    spec = JobSpec(
        job_id="job-x",
        kind="llm.completion",
        payload={"prompt": "ping", "max_tokens": 8},
        cost_ceiling_usd=1.0,
        latency_ceiling_ms=10_000,
        privacy_class="private",
        quality_floor=0.5,
    )
    res = auction.run(spec)
    out = p.execute(spec, res.winner)
    assert out["status"] == "executed"
    assert out["model_id"] == ALPHA_FLAGSHIPS[0].model_id
    assert out["result_hash"]
    assert out["provider_sig"]                   # wallet signed it
    assert out["provider_pubkey_pem"]


# ---------------------------------------------------------------------------
# Cost estimator
# ---------------------------------------------------------------------------

def test_cost_estimator_returns_positive_savings():
    est = estimate_training_cost_usd(target_params_b=1.5)
    assert est["target_params_b"] == 1.5
    assert Decimal(est["public_cloud_usd"]) > 0
    assert Decimal(est["pluginfer_mesh_usd"]) > 0
    assert Decimal(est["pluginfer_mesh_usd"]) < Decimal(est["public_cloud_usd"])
    assert Decimal(est["pluginfer_savings_usd"]) > 0


def test_cost_estimator_scales_linearly_in_params():
    """Token count = params × ratio; gpu-hours = params × tokens; cost
    scales ~params^2 at fixed token-per-param ratio."""
    a = estimate_training_cost_usd(target_params_b=1.0)
    b = estimate_training_cost_usd(target_params_b=2.0)
    # cost should be ~4x at 2x params (since tokens also doubled).
    ratio = float(Decimal(b["public_cloud_usd"]) / Decimal(a["public_cloud_usd"]))
    assert 3.5 < ratio < 4.5, ratio
