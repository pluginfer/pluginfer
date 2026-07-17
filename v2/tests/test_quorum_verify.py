"""§mesh-trust — quorum verification of untrusted compute.

Pins the honest mitigation for "you can't trust an anonymous node's
output": redundant execution + K-of-N agreement, with the failure
modes (single liar outvoted, split vote = dispute, only majority paid)
all exercised. Deterministic; no network.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

from core.quorum_verify import (
    QuorumPolicy, evaluate_quorum, run_quorum,
)


# ---------------------------------------------------------------------------
# Pure decision core
# ---------------------------------------------------------------------------

def test_unanimous_agreement_accepts():
    out = evaluate_quorum(
        [("a", "H"), ("b", "H"), ("c", "H")], quorum=2)
    assert out.accepted
    assert out.agreed_result_hash == "H"
    assert set(out.paid_providers()) == {"a", "b", "c"}
    assert out.dissenting_providers() == []


def test_single_liar_is_outvoted_and_unpaid():
    out = evaluate_quorum(
        [("a", "H"), ("b", "H"), ("liar", "WRONG")], quorum=2)
    assert out.accepted
    assert out.agreed_result_hash == "H"
    assert set(out.paid_providers()) == {"a", "b"}
    assert out.dissenting_providers() == ["liar"]      # fed to slashing


def test_no_majority_is_a_dispute_not_a_guess():
    out = evaluate_quorum(
        [("a", "X"), ("b", "Y"), ("c", "Z")], quorum=2)
    assert not out.accepted
    assert out.dispute
    assert out.paid_providers() == []


def test_split_tie_below_quorum_disputes():
    out = evaluate_quorum(
        [("a", "X"), ("b", "X"), ("c", "Y"), ("d", "Y")], quorum=3)
    assert not out.accepted
    assert out.dispute
    assert "split vote" in out.reason


def test_failed_providers_count_as_non_votes():
    # Two real results agree, one node failed (None) -> still accepted
    # at quorum 2, and the failed node is not paid.
    out = evaluate_quorum(
        [("a", "H"), ("b", "H"), ("dead", None)], quorum=2)
    assert out.accepted
    assert out.responded == 2
    assert "dead" not in out.paid_providers()


def test_all_failed_is_dispute():
    out = evaluate_quorum(
        [("a", None), ("b", None)], quorum=2)
    assert not out.accepted
    assert out.reason.startswith("no provider")


def test_quorum_not_reached_when_majority_too_thin():
    # 2 agree but quorum demands 3.
    out = evaluate_quorum(
        [("a", "H"), ("b", "H"), ("c", "WRONG")], quorum=3)
    assert not out.accepted
    assert out.agreement_count == 2


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def test_policy_default_is_no_redundancy():
    assert QuorumPolicy().enabled is False
    assert QuorumPolicy(n=3).enabled is True


def test_policy_disabled_for_nondeterministic_jobs():
    # Sampling jobs have no single correct hash — quorum must not apply.
    assert QuorumPolicy(n=3, deterministic=False).enabled is False


def test_policy_default_quorum_is_strict_majority():
    assert QuorumPolicy(n=3).required_quorum() == 2
    assert QuorumPolicy(n=5).required_quorum() == 3
    assert QuorumPolicy(n=4, quorum=4).required_quorum() == 4


# ---------------------------------------------------------------------------
# Async orchestration
# ---------------------------------------------------------------------------

def test_run_quorum_dispatches_and_accepts():
    async def dispatch(pid):
        return "GOOD" if pid != "liar" else "BAD"

    out = asyncio.run(run_quorum(
        dispatch, ["a", "b", "liar"], QuorumPolicy(n=3)))
    assert out.accepted
    assert out.agreed_result_hash == "GOOD"
    assert set(out.paid_providers()) == {"a", "b"}


def test_run_quorum_raising_provider_becomes_nonvote():
    async def dispatch(pid):
        if pid == "boom":
            raise RuntimeError("node crashed mid-job")
        return "GOOD"

    # Two good + one crashing, quorum 2 -> still accepted, crash absorbed.
    out = asyncio.run(run_quorum(
        dispatch, ["a", "b", "boom"], QuorumPolicy(n=3)))
    assert out.accepted
    assert "boom" not in out.paid_providers()
    assert out.responded == 2
