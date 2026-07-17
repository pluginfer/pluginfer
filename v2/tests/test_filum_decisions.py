"""Filum self-context + decision engine smoke tests.

Verifies:
* SelfContextIndex builds, queries, returns BM25-ranked chunks
* Index content_hash is stable
* DecisionEngine.gate_content wraps SafetyGate
* DecisionEngine.route_inference picks cheapest reliable
* DecisionEngine.should_promote_sun gates by stability
* DecisionEngine.accept_gradient respects staleness + pressure
* DecisionEngine.pick_next_capability_gap routes to improver
* DecisionEngine history is bounded
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest


# ---------- self-context ---------------------------------------------------

def test_self_context_indexes_a_repo(tmp_path: Path):
    from ai.filum.self_context import SelfContextIndex, IndexConfig

    # Build a tiny fake repo.
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "sun_election.py").write_text(
        "def elect_local_suns(self_view, peers):\n"
        "    '''Pick the K most stable peers as Suns.'''\n"
        "    return sorted(peers, key=lambda p: -p.stability_score)[:3]\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "# Pluginfer\nDecentralized AI compute mesh.\nSun election uses\n"
        "hardware pressure stability to pick mesh aggregators.\n",
        encoding="utf-8",
    )

    idx = SelfContextIndex.build(IndexConfig(repo_root=str(tmp_path)))
    assert idx.stats.n_docs >= 2
    res = idx.query("how does sun election work?", top_k=3)
    assert res, "query returned nothing"
    # Top result should be one of the two files we created.
    assert any("sun_election" in r.path or "README" in r.path for r in res)


def test_self_context_content_hash_stable(tmp_path: Path):
    from ai.filum.self_context import SelfContextIndex, IndexConfig

    (tmp_path / "f.py").write_text("x = 1\n", encoding="utf-8")
    a = SelfContextIndex.build(IndexConfig(repo_root=str(tmp_path)))
    b = SelfContextIndex.build(IndexConfig(repo_root=str(tmp_path)))
    assert a.content_hash() == b.content_hash()


def test_self_context_summary_includes_kinds(tmp_path: Path):
    from ai.filum.self_context import SelfContextIndex, IndexConfig

    (tmp_path / "src.py").write_text("def f(): pass\n", encoding="utf-8")
    (tmp_path / "INVENTIONS.md").write_text("## §X claim\n", encoding="utf-8")
    (tmp_path / "WORKLOG.md").write_text("## 2026-05-09 progress\n",
                                          encoding="utf-8")
    idx = SelfContextIndex.build(IndexConfig(repo_root=str(tmp_path)))
    summary = idx.stats_summary()
    assert summary["n_chunks"] >= 3
    assert "source" in summary["by_kind"]
    assert "inventions" in summary["by_kind"]
    assert "worklog" in summary["by_kind"]


# ---------- decision engine ------------------------------------------------

def test_gate_content_wraps_safety_gate():
    from ai.filum.decision_engine import DecisionEngine
    from ai.filum.hpa.safety import SafetyGate

    engine = DecisionEngine(safety_gate=SafetyGate())
    d = engine.gate_content(pubkey="alice", content="hello world")
    assert d.action == "allow"

    # Sanctioned region rejects.
    d2 = engine.gate_content(pubkey="bob", content="ok", region="IR")
    assert d2.action == "deny"
    assert "sanctioned" in d2.rationale.lower()


def test_gate_content_with_no_safety_bound_defaults_allow():
    from ai.filum.decision_engine import DecisionEngine

    engine = DecisionEngine()
    d = engine.gate_content(pubkey="x", content="anything")
    assert d.action == "allow"
    assert "no-safety-gate-bound" in d.rules_fired


def test_route_inference_picks_cheapest_reliable():
    from ai.filum.decision_engine import DecisionEngine

    engine = DecisionEngine()
    candidates = [
        {"id": "p1", "price": 0.20, "reliability": 0.95, "energy": "grid"},
        {"id": "p2", "price": 0.10, "reliability": 0.50, "energy": "grid"},
        {"id": "p3", "price": 0.15, "reliability": 0.90, "energy": "green"},
    ]
    d = engine.route_inference(candidates=candidates,
                                max_price=0.30, min_reliability=0.8)
    # Eligible: p1 (0.20), p3 (0.15). p3 is cheaper.
    assert d.action == "route"
    assert d.chosen_id == "p3"


def test_route_inference_no_match_when_thresholds_not_met():
    from ai.filum.decision_engine import DecisionEngine

    engine = DecisionEngine()
    candidates = [{"id": "p1", "price": 0.50, "reliability": 0.99}]
    d = engine.route_inference(candidates=candidates, max_price=0.10)
    assert d.action == "no_match"


def test_route_inference_green_preference_filters():
    from ai.filum.decision_engine import DecisionEngine

    engine = DecisionEngine()
    candidates = [
        {"id": "grid", "price": 0.05, "reliability": 0.9, "energy": "grid"},
        {"id": "green", "price": 0.20, "reliability": 0.9, "energy": "green"},
    ]
    d = engine.route_inference(candidates=candidates, max_price=1.0,
                                prefer_green=True)
    assert d.chosen_id == "green"


def test_should_promote_sun_threshold():
    from ai.filum.decision_engine import DecisionEngine
    from ai.filum.hpa.sun_election import NodeMembership

    engine = DecisionEngine()
    high = NodeMembership(node_id="hi", stability_score=0.9)
    low  = NodeMembership(node_id="lo", stability_score=0.4)

    d_hi = engine.should_promote_sun(high, current_suns=[])
    d_lo = engine.should_promote_sun(low,  current_suns=[])
    assert d_hi.action == "promote"
    assert d_lo.action == "keep_planet"


def test_should_promote_sun_full_set_keeps_unless_better():
    from ai.filum.decision_engine import DecisionEngine
    from ai.filum.hpa.sun_election import NodeMembership

    engine = DecisionEngine()
    full = [
        NodeMembership(node_id="s1", stability_score=0.95),
        NodeMembership(node_id="s2", stability_score=0.92),
        NodeMembership(node_id="s3", stability_score=0.85),
    ]
    candidate_better = NodeMembership(node_id="cand",
                                        stability_score=0.93)
    candidate_worse = NodeMembership(node_id="bad",
                                       stability_score=0.80)
    assert engine.should_promote_sun(candidate_better,
                                      current_suns=full).action == "promote"
    assert engine.should_promote_sun(candidate_worse,
                                      current_suns=full).action == "keep_planet"


def test_accept_gradient_rejects_overstale():
    from ai.filum.decision_engine import DecisionEngine
    from ai.filum.hpa.grain import Grain, GrainMeta

    engine = DecisionEngine()
    g = Grain(meta=GrainMeta(version_v=0))
    d = engine.accept_gradient(grain=g, current_version=10_000, tau=10.0)
    assert d.action == "reject"


def test_accept_gradient_accepts_fresh():
    from ai.filum.decision_engine import DecisionEngine
    from ai.filum.hpa.grain import Grain, GrainMeta

    engine = DecisionEngine()
    g = Grain(meta=GrainMeta(version_v=100, pressure_at_birth=0.2))
    d = engine.accept_gradient(grain=g, current_version=105, tau=200.0)
    assert d.action == "accept"
    assert 0.0 < d.confidence <= 1.0


def test_pick_capability_gap_no_improver_skips():
    from ai.filum.decision_engine import DecisionEngine

    d = DecisionEngine().pick_next_capability_gap()
    assert d.action == "skip"


def test_should_issue_receipt_user_request_always_issues():
    from ai.filum.decision_engine import DecisionEngine

    engine = DecisionEngine()
    d = engine.should_issue_receipt(content_class="general",
                                     user_requested=True)
    assert d.action == "issue"


def test_should_issue_receipt_non_general_always_issues():
    from ai.filum.decision_engine import DecisionEngine

    engine = DecisionEngine()
    d = engine.should_issue_receipt(content_class="medical")
    assert d.action == "issue"


def test_should_issue_receipt_general_defers():
    from ai.filum.decision_engine import DecisionEngine

    engine = DecisionEngine()
    d = engine.should_issue_receipt(content_class="general")
    assert d.action == "defer"


def test_decision_history_bounded():
    from ai.filum.decision_engine import DecisionEngine

    engine = DecisionEngine()
    for _ in range(2000):
        engine.gate_content(pubkey="x", content="ok")
    h = engine.history()
    assert len(h) <= 1024
