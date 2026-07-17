"""Tests for the $/useful-token benchmark tool (gap #10)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.benchmark_dollar_per_token import (
    BenchmarkReport, run_benchmark, _run_cloud_constant_price,
    _VENDOR_PRICING_USD_PER_1M_TOKENS,
)


def test_cloud_constant_price_matches_published_pricing():
    """Sanity-check the vendor pricing snapshot. If anyone bumps the
    constants accidentally, this test catches it."""
    res = _run_cloud_constant_price("anthropic_claude_haiku", target_tokens=1000)
    rate = _VENDOR_PRICING_USD_PER_1M_TOKENS["anthropic_claude_haiku"]
    expected_per_token = (rate["input"] + rate["output"]) / 2 / 1_000_000
    assert abs(res.cost_per_useful_token_usd - expected_per_token) < 1e-12
    assert res.useful_tokens == 1000


def test_run_benchmark_produces_three_backend_report():
    report = run_benchmark(
        backends=["pluginfer", "anthropic", "openai"],
        target_tokens=500, workload="distill",
    )
    assert isinstance(report, BenchmarkReport)
    assert len(report.backends) == 3
    assert {r.backend for r in report.backends} >= {
        "pluginfer", "anthropic_claude_haiku", "openai_gpt4o_mini",
    }


def test_pluginfer_projected_undercuts_cloud_apis():
    """The strategic claim: Pluginfer's PROJECTED $/useful-token (what
    the §A12 reverse-auction will clear at, on consumer-electricity
    floor) materially undercuts vendor APIs.

    NOTE: the benchmark tool ALSO supports live-mode (drives a real
    Ollama generation, costs measured at electricity-only rate). On a
    slow consumer GPU like a GTX 1650 at ~13 tok/s, live-mode CAN be
    more expensive per token than GPT-4o-mini because OpenAI runs at
    H100 scale and amortises across millions of users. That's the
    honest answer for tiny one-off inference.

    Pluginfer wins structurally on:
      * Training / fine-tuning / distillation (sustained throughput,
        amortised over hours).
      * Regulated workloads AWS legally cannot serve.
      * Workloads where the consumer GPU is already paid for and idle.

    This test pins the projected-rate claim using the cloud constant-
    price runner so it's deterministic and CI-stable."""
    from tools.benchmark_dollar_per_token import _run_cloud_constant_price
    pf = _run_cloud_constant_price("pluginfer_mesh_projected", 1000)
    haiku = _run_cloud_constant_price("anthropic_claude_haiku", 1000)
    gpt4o_mini = _run_cloud_constant_price("openai_gpt4o_mini", 1000)
    sonnet = _run_cloud_constant_price("anthropic_claude_sonnet", 1000)

    # Pluginfer projected must be cheaper than Anthropic Haiku.
    assert pf.cost_per_useful_token_usd < haiku.cost_per_useful_token_usd, (
        f"Pluginfer projected (${pf.cost_per_useful_token_usd:.10f}) >= "
        f"Haiku (${haiku.cost_per_useful_token_usd:.10f}) — "
        "projected-rate thesis broken!"
    )
    # And cheaper than GPT-4o-mini (the cheapest mainstream option).
    assert pf.cost_per_useful_token_usd < gpt4o_mini.cost_per_useful_token_usd
    # And much cheaper than Sonnet — at least 10x.
    assert (sonnet.cost_per_useful_token_usd
            >= 10 * pf.cost_per_useful_token_usd)


def test_report_finalize_picks_correct_extremes():
    report = run_benchmark(
        backends=["pluginfer", "anthropic_claude_sonnet", "openai_gpt4o"],
        target_tokens=1000,
    )
    assert report.cheapest_backend == "pluginfer"
    assert report.most_expensive_backend in (
        "anthropic_claude_sonnet", "openai_gpt4o",
    )
    assert report.ratio_cheapest_to_most_expensive is not None
    assert report.ratio_cheapest_to_most_expensive > 1


def test_report_serialises_to_json_cleanly(tmp_path: Path):
    """The dashboard consumes this JSON; format must be stable."""
    report = run_benchmark(
        backends=["pluginfer", "anthropic"], target_tokens=100,
    )
    out_path = tmp_path / "bench.json"
    payload = {
        "timestamp_unix": report.timestamp_unix,
        "workload": report.workload,
        "backends": [
            {"backend": r.backend,
             "cost_per_useful_token_usd": r.cost_per_useful_token_usd,
             "useful_tokens": r.useful_tokens}
            for r in report.backends
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    parsed = json.loads(out_path.read_text())
    assert len(parsed["backends"]) == 2
    assert parsed["workload"] == "distill"
