"""§RFC-3 BudgetLedger — Budget-as-Contract core semantics.

Pins the contract the enterprise sale rests on: caps bind fail-closed
BEFORE money moves, hierarchical envelopes all bind, windows roll over
deterministically, and every settled dollar is attributable via the
journal. Clock is injected — no sleeps, no flakes.
"""

from __future__ import annotations

import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

from governance.budget_ledger import BudgetLedger, RESERVATION_TTL_S


class _Clock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _ledger(**kw) -> "tuple[BudgetLedger, _Clock]":
    clk = _Clock()
    return BudgetLedger(None, clock=clk, **kw), clk


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------

def test_cap_binds_fail_closed():
    bl, _ = _ledger()
    bl.set_envelope("acme", 1.00, "month")
    assert bl.reserve("j1", "acme", 0.60) is None
    refusal = bl.reserve("j2", "acme", 0.60)
    assert refusal and "acme" in refusal and "headroom" in refusal


def test_hierarchy_every_prefix_binds():
    bl, _ = _ledger()
    bl.set_envelope("acme", 10.00, "month")
    bl.set_envelope("acme/support", 1.00, "month")
    # Child cap binds even though the org has room.
    assert bl.reserve("j1", "acme/support/bot", 0.50) is None
    refusal = bl.reserve("j2", "acme/support/bot", 0.60)
    assert refusal and "acme/support" in refusal
    # A sibling team under the same org is unaffected.
    assert bl.reserve("j3", "acme/eng/copilot", 5.00) is None


def test_org_cap_binds_across_teams():
    bl, _ = _ledger()
    bl.set_envelope("acme", 1.00, "month")
    assert bl.reserve("j1", "acme/a", 0.70) is None
    refusal = bl.reserve("j2", "acme/b", 0.70)
    assert refusal and "'acme'" in refusal


def test_unknown_path_allowed_by_default_but_journalled():
    bl, _ = _ledger()
    assert bl.reserve("j1", "nobody/configured/this", 5.0) is None
    bl.settle("j1", 5.0)
    rep = bl.report()
    assert rep["total_spend_usd"] == pytest.approx(5.0)
    assert rep["ungoverned_spend_usd"] == pytest.approx(5.0)


def test_require_envelope_refuses_uncovered_paths():
    bl, _ = _ledger(require_envelope=True)
    refusal = bl.reserve("j1", "unknown/app", 0.01)
    assert refusal and "no envelope covers" in refusal
    bl.set_envelope("unknown", 1.0, "month")
    assert bl.reserve("j2", "unknown/app", 0.01) is None


# ---------------------------------------------------------------------------
# Lifecycle: reserve → settle | release
# ---------------------------------------------------------------------------

def test_settle_records_actual_not_ceiling():
    bl, _ = _ledger()
    bl.set_envelope("acme", 1.00, "month")
    bl.reserve("j1", "acme", 0.90)        # worst-case hold
    bl.settle("j1", 0.05)                 # auction cleared much lower
    # The freed headroom is available again immediately.
    assert bl.reserve("j2", "acme", 0.90) is None
    assert bl.report()["total_spend_usd"] == pytest.approx(0.05)


def test_release_frees_the_hold():
    bl, _ = _ledger()
    bl.set_envelope("acme", 1.00, "month")
    bl.reserve("j1", "acme", 1.00)
    assert bl.reserve("j2", "acme", 0.50) is not None
    bl.release("j1")
    assert bl.reserve("j2", "acme", 0.50) is None


def test_settle_and_release_idempotent():
    bl, _ = _ledger()
    bl.set_envelope("acme", 1.00, "month")
    bl.reserve("j1", "acme", 0.40)
    bl.settle("j1", 0.40)
    bl.settle("j1", 0.40)                 # double-settle: no-op
    bl.release("j1")                      # after settle: no-op
    assert bl.report()["total_spend_usd"] == pytest.approx(0.40)
    # Re-reserving a settled job_id is a no-op success, not a new hold.
    assert bl.reserve("j1", "acme", 99.0) is None
    assert bl.reserve("probe", "acme", 0.60) is None


def test_stale_reservation_self_expires():
    bl, clk = _ledger()
    bl.set_envelope("acme", 1.00, "month")
    bl.reserve("crashed-job", "acme", 1.00)
    assert bl.reserve("j2", "acme", 0.50) is not None
    clk.t += RESERVATION_TTL_S + 1
    assert bl.reserve("j2", "acme", 0.50) is None


# ---------------------------------------------------------------------------
# Rollover
# ---------------------------------------------------------------------------

def test_day_window_rolls_over_and_resets_spend():
    bl, clk = _ledger()
    bl.set_envelope("acme", 1.00, "day")
    bl.reserve("j1", "acme", 1.00)
    bl.settle("j1", 1.00)
    assert bl.reserve("j2", "acme", 0.10) is not None
    clk.t += 86_400 + 1
    assert bl.reserve("j2", "acme", 0.10) is None
    # History survives the rollover in the journal.
    assert bl.report()["total_spend_usd"] == pytest.approx(1.00)


def test_total_period_never_rolls_over():
    bl, clk = _ledger()
    bl.set_envelope("acme", 1.00, "total")
    bl.reserve("j1", "acme", 1.00)
    bl.settle("j1", 1.00)
    clk.t += 365 * 86_400
    assert bl.reserve("j2", "acme", 0.01) is not None


def test_lowering_cap_below_spend_blocks_until_rollover():
    bl, _ = _ledger()
    bl.set_envelope("acme", 10.00, "month")
    bl.reserve("j1", "acme", 5.00)
    bl.settle("j1", 5.00)
    bl.set_envelope("acme", 1.00, "month")   # emergency clamp-down
    assert bl.reserve("j2", "acme", 0.01) is not None


# ---------------------------------------------------------------------------
# Chargeback + persistence
# ---------------------------------------------------------------------------

def test_report_groups_by_envelope_path():
    bl, _ = _ledger()
    bl.set_envelope("acme", 100.0, "month")
    for i, (path, cost) in enumerate([
        ("acme/support/bot", 1.0),
        ("acme/support/bot", 2.0),
        ("acme/eng/copilot", 4.0),
    ]):
        bl.reserve(f"j{i}", path, cost)
        bl.settle(f"j{i}", cost)
    rep = bl.report(prefix="acme/support")
    assert rep["total_spend_usd"] == pytest.approx(3.0)
    assert rep["by_envelope"]["acme/support/bot"]["jobs"] == 2
    full = bl.report()
    assert full["total_spend_usd"] == pytest.approx(7.0)
    assert full["ungoverned_spend_usd"] == 0.0


def test_state_persists_across_restart(tmp_path):
    d = str(tmp_path / "budget")
    clk = _Clock()
    bl = BudgetLedger(d, clock=clk)
    bl.set_envelope("acme", 1.00, "month")
    bl.reserve("j1", "acme", 0.75)
    bl.settle("j1", 0.75)
    # New process, same dir: spend and journal both survive.
    bl2 = BudgetLedger(d, clock=clk)
    assert bl2.reserve("j2", "acme", 0.50) is not None
    assert bl2.report()["total_spend_usd"] == pytest.approx(0.75)


def test_bad_inputs_rejected():
    bl, _ = _ledger()
    with pytest.raises(ValueError):
        bl.set_envelope("acme", 1.0, "fortnight")
    with pytest.raises(ValueError):
        bl.set_envelope("", 1.0)
    with pytest.raises(ValueError):
        bl.set_envelope("acme", -1.0)
