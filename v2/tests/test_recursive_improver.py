"""Recursive self-improvement loop smoke tests.

These tests verify the *coordination* logic — that the loop:

* picks the largest unaddressed gap
* skips clusters that already have a registered adapter
* runs self-play, training, eval, registration, publish in order
* stops short when held-out loss is above threshold
* records every cycle to an audit log
* respects cooldown between cycles

The actual self-play / training / eval are pluggable — we use mocks
that exercise the wiring without depending on torch or APIs.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest


# --- mocks shaped like the production subsystems ----------------------------


@dataclass
class MockCluster:
    cluster_id: str
    label: str
    size: int
    avg_teacher_entropy: float
    mean_loss: float


class MockClusterDetector:
    def __init__(self, clusters):
        self._clusters = clusters

    def pending_clusters(self):
        return list(self._clusters)


class MockLoRAPool:
    def __init__(self, existing_cluster_ids=()):
        self._adapters = [
            type("A", (), {"cluster_id": c, "adapter_id": "lora-pre-" + c})()
            for c in existing_cluster_ids
        ]

    def list_adapters(self):
        return list(self._adapters)

    def register_adapter(self, *, adapter_id, cluster_id, label):
        self._adapters.append(
            type("A", (), {"cluster_id": cluster_id, "adapter_id": adapter_id})()
        )


class MockMarketplace:
    def __init__(self):
        self.listings: list = []

    def publish(self, *, adapter_id, cluster_id, label, held_out_loss):
        listing_id = f"listing-{len(self.listings) + 1}"
        self.listings.append({
            "listing_id": listing_id,
            "adapter_id": adapter_id,
            "cluster_id": cluster_id,
            "label": label,
            "held_out_loss": held_out_loss,
        })
        return listing_id


class MockSelfPlay:
    def __init__(self, samples_per_round=4):
        self._n = samples_per_round
        self.rounds_called = 0

    async def propose_round(self):
        self.rounds_called += 1
        return [f"sample_{self.rounds_called}_{i}" for i in range(self._n)]


class MockActiveSampler:
    """Stand-in; the improver doesn't strictly require it, but the
    signature wants something to pass."""
    pass


# --- tests -----------------------------------------------------------------


def test_identify_gap_picks_highest_weight(tmp_path: Path):
    from ai.filum.recursive_improver import (
        RecursiveImprover, RecursiveImproverConfig,
    )

    detector = MockClusterDetector([
        MockCluster("c1", "small", size=5, avg_teacher_entropy=2.0, mean_loss=1.0),
        MockCluster("c2", "big",   size=50, avg_teacher_entropy=3.0, mean_loss=2.5),
        MockCluster("c3", "med",   size=10, avg_teacher_entropy=2.5, mean_loss=1.5),
    ])
    imp = RecursiveImprover(
        active_sampler=MockActiveSampler(),
        cluster_detector=detector,
        self_play_generator=MockSelfPlay(),
        lora_pool=MockLoRAPool(),
        capability_marketplace=MockMarketplace(),
        config=RecursiveImproverConfig(
            work_dir=str(tmp_path), cycle_log=str(tmp_path / "cycles.jsonl"),
        ),
    )
    gap = imp.identify_gap()
    assert gap is not None
    # c2: weight = 2.5 * 50 = 125 (largest)
    assert gap.cluster_id == "c2"


def test_identify_gap_skips_existing_adapters(tmp_path: Path):
    from ai.filum.recursive_improver import RecursiveImprover, RecursiveImproverConfig

    detector = MockClusterDetector([
        MockCluster("c1", "x", size=100, avg_teacher_entropy=3.0, mean_loss=2.0),
        MockCluster("c2", "y", size=10,  avg_teacher_entropy=2.0, mean_loss=1.0),
    ])
    pool = MockLoRAPool(existing_cluster_ids=("c1",))
    imp = RecursiveImprover(
        active_sampler=MockActiveSampler(),
        cluster_detector=detector,
        self_play_generator=MockSelfPlay(),
        lora_pool=pool,
        capability_marketplace=MockMarketplace(),
        config=RecursiveImproverConfig(
            work_dir=str(tmp_path), cycle_log=str(tmp_path / "cycles.jsonl"),
        ),
    )
    gap = imp.identify_gap()
    assert gap is not None
    assert gap.cluster_id == "c2"


def test_full_cycle_publishes_and_logs(tmp_path: Path):
    from ai.filum.recursive_improver import RecursiveImprover, RecursiveImproverConfig

    detector = MockClusterDetector([
        MockCluster("c1", "regex-multiline", size=20,
                    avg_teacher_entropy=2.5, mean_loss=2.0),
    ])
    market = MockMarketplace()
    pool = MockLoRAPool()
    sp = MockSelfPlay(samples_per_round=8)

    log_path = tmp_path / "cycles.jsonl"
    imp = RecursiveImprover(
        active_sampler=MockActiveSampler(),
        cluster_detector=detector,
        self_play_generator=sp,
        lora_pool=pool,
        capability_marketplace=market,
        config=RecursiveImproverConfig(
            work_dir=str(tmp_path),
            cycle_log=str(log_path),
            samples_per_gap=8,
            cycle_cooldown_s=0.0,
            held_out_loss_max=100.0,   # well above stub eval's output
        ),
    )
    result = asyncio.run(imp.run_cycle())

    assert result.gap is not None
    assert result.gap.cluster_id == "c1"
    assert result.samples_generated >= 1
    assert result.adapter_id and result.adapter_id.startswith("lora-")
    assert result.adapter_held_out_loss is not None
    assert result.listing_id == "listing-1"
    assert len(market.listings) == 1
    assert market.listings[0]["cluster_id"] == "c1"

    # Audit log: one line.
    assert log_path.exists()
    lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["cycle_id"] == 1
    assert rec["adapter_id"] == result.adapter_id


def test_cycle_skips_publish_when_held_out_loss_above_threshold(tmp_path: Path):
    from ai.filum.recursive_improver import (
        RecursiveImprover, RecursiveImproverConfig, CapabilityGap,
    )

    detector = MockClusterDetector([
        MockCluster("c-bad", "hard",
                    size=100, avg_teacher_entropy=4.0, mean_loss=10.0),
    ])
    market = MockMarketplace()

    def bad_eval(adapter_id, gap):
        return 999.0   # never publishes

    imp = RecursiveImprover(
        active_sampler=MockActiveSampler(),
        cluster_detector=detector,
        self_play_generator=MockSelfPlay(),
        lora_pool=MockLoRAPool(),
        capability_marketplace=market,
        held_out_evaluator=bad_eval,
        config=RecursiveImproverConfig(
            work_dir=str(tmp_path),
            cycle_log=str(tmp_path / "cycles.jsonl"),
            samples_per_gap=4,
            held_out_loss_max=3.0,
            cycle_cooldown_s=0.0,
        ),
    )
    result = asyncio.run(imp.run_cycle())
    assert result.adapter_id is not None
    assert result.adapter_held_out_loss == 999.0
    assert result.listing_id is None
    assert market.listings == []
    assert "not publishing" in result.notes


def test_cycle_no_gap(tmp_path: Path):
    from ai.filum.recursive_improver import RecursiveImprover, RecursiveImproverConfig

    imp = RecursiveImprover(
        active_sampler=MockActiveSampler(),
        cluster_detector=MockClusterDetector([]),
        self_play_generator=MockSelfPlay(),
        lora_pool=MockLoRAPool(),
        capability_marketplace=MockMarketplace(),
        config=RecursiveImproverConfig(
            work_dir=str(tmp_path),
            cycle_log=str(tmp_path / "cycles.jsonl"),
            cycle_cooldown_s=0.0,
        ),
    )
    result = asyncio.run(imp.run_cycle())
    assert result.gap is None
    assert result.notes == "no gap identified"


def test_cooldown_blocks_back_to_back_cycles(tmp_path: Path):
    from ai.filum.recursive_improver import RecursiveImprover, RecursiveImproverConfig

    detector = MockClusterDetector([
        MockCluster("c1", "x", size=10, avg_teacher_entropy=2.0, mean_loss=1.0),
    ])
    imp = RecursiveImprover(
        active_sampler=MockActiveSampler(),
        cluster_detector=detector,
        self_play_generator=MockSelfPlay(),
        lora_pool=MockLoRAPool(),
        capability_marketplace=MockMarketplace(),
        config=RecursiveImproverConfig(
            work_dir=str(tmp_path),
            cycle_log=str(tmp_path / "cycles.jsonl"),
            cycle_cooldown_s=10.0,    # long cooldown
            held_out_loss_max=100.0,  # let the cycle complete
        ),
    )
    r1 = asyncio.run(imp.run_cycle())
    assert r1.gap is not None and r1.adapter_id is not None
    r2 = asyncio.run(imp.run_cycle())
    assert r2.notes == "cooldown"
