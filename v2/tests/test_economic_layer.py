"""HG21 — economic layer: stake + slashing + reputation over quorum.

Pins the money-safety bar:

  * every stake/unstake/slash leg keeps verify_balances GREEN — the
    audit invariants extend to the new bucket, they are not exempted,
  * slash proceeds go to the honest majority, NEVER the treasury,
  * dissenters are slashed only against an ACCEPTED majority — a
    dispute slashes nobody,
  * unbonding delay: funds stay slashable until maturity; a slash
    during unbonding shrinks the eventual claim,
  * quarantine trips on repeated dissent and gates the auction,
  * state survives restarts.

Deterministic: `now_fn` is injected; no sleeps, no network.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

from core.buyer_ledger import BuyerLedger, InsufficientFunds, \
    TREASURY_WALLET_ID
from core.economic_layer import EconomicLayer
from core.quorum_verify import evaluate_quorum

D = Decimal


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _funded_ledger(tmp_path, wallets=("p1", "p2", "p3"), amount="10"):
    led = BuyerLedger(str(tmp_path / "ledger"))
    for w in wallets:
        led.credit(w, D(amount), note="test funding")
    return led


def _econ(tmp_path, led=None, clock=None):
    led = led or _funded_ledger(tmp_path)
    clock = clock or Clock()
    return EconomicLayer(led, state_dir=str(tmp_path / "ledger"),
                         now_fn=clock), led, clock


def _accepted_outcome(majority=("p1", "p2"), dissenters=("p3",)):
    results = [(p, "good-hash") for p in majority]
    results += [(p, "bad-hash") for p in dissenters]
    out = evaluate_quorum(results, quorum=2)
    assert out.accepted
    return out


# ---------------------------------------------------------------------------
# Ledger legs
# ---------------------------------------------------------------------------

def test_stake_and_unstake_keep_verify_balances_green(tmp_path):
    led = _funded_ledger(tmp_path)
    led.stake("p1", D("3"))
    w = led.get_wallet("p1")
    assert w.available_usd == D("7") and w.staked_usd == D("3")
    led.unstake_release("p1", D("1"))
    w = led.get_wallet("p1")
    assert w.available_usd == D("8") and w.staked_usd == D("2")
    v = led.verify_balances()
    assert v["ok"], v


def test_stake_overdraw_and_bad_amounts_refused(tmp_path):
    led = _funded_ledger(tmp_path)
    with pytest.raises(InsufficientFunds):
        led.stake("p1", D("11"))
    with pytest.raises(ValueError):
        led.stake("p1", D("0"))
    with pytest.raises(InsufficientFunds):
        led.unstake_release("p1", D("1"))     # nothing staked


def test_slash_caps_at_stake_and_award_never_treasury(tmp_path):
    led = _funded_ledger(tmp_path)
    led.stake("p3", D("2"))
    taken = led.slash_stake("p3", D("5"), job_id="j1")
    assert taken == D("2")                    # capped at the bond
    assert led.get_wallet("p3").staked_usd == D("0")
    with pytest.raises(ValueError, match="treasury"):
        led.slash_award(TREASURY_WALLET_ID, D("1"), job_id="j1")
    led.slash_award("p1", D("2"), job_id="j1")
    assert led.get_wallet("p1").available_usd == D("12")
    assert led.verify_balances()["ok"]


def test_staked_balance_survives_ledger_reload(tmp_path):
    led = _funded_ledger(tmp_path)
    led.stake("p1", D("4"))
    led2 = BuyerLedger(str(tmp_path / "ledger"))
    w = led2.get_wallet("p1")
    assert w.staked_usd == D("4") and w.available_usd == D("6")
    assert led2.verify_balances()["ok"]


# ---------------------------------------------------------------------------
# Slashing on quorum outcomes
# ---------------------------------------------------------------------------

def test_dissenter_slashed_majority_awarded(tmp_path):
    econ, led, _ = _econ(tmp_path)
    for p in ("p1", "p2", "p3"):
        econ.stake(p, D("5"))
    out = _accepted_outcome()
    summary = econ.record_quorum_outcome(out, job_id="j1",
                                         job_price_usd=D("2"))
    # p3 slashed price × multiplier (1.0) = 2; split equally p1/p2.
    assert summary["slashed"] == {"p3": "2.00000000"}
    assert led.get_wallet("p3").staked_usd == D("3")
    assert led.get_wallet("p1").available_usd == D("6")   # 5 + 1
    assert led.get_wallet("p2").available_usd == D("6")
    assert led.treasury_balance() == D("0")               # never a cent
    assert led.verify_balances()["ok"]


def test_dispute_slashes_nobody(tmp_path):
    econ, led, _ = _econ(tmp_path)
    econ.stake("p1", D("5"))
    econ.stake("p2", D("5"))
    out = evaluate_quorum([("p1", "h1"), ("p2", "h2")], quorum=2)
    assert not out.accepted
    summary = econ.record_quorum_outcome(out, job_id="j1",
                                         job_price_usd=D("2"))
    assert summary["slashed"] == {}
    assert "nobody" in summary["action"]
    assert led.get_wallet("p1").staked_usd == D("5")
    assert led.get_wallet("p2").staked_usd == D("5")


def test_slash_pot_split_remainder_accounted(tmp_path):
    led = _funded_ledger(tmp_path, wallets=("p1", "p2", "p3", "p4"))
    econ, led, _ = _econ(tmp_path, led=led)
    for p in ("p1", "p2", "p3", "p4"):
        econ.stake(p, D("5"))            # available now 5 each
    results = [("p1", "good"), ("p2", "good"), ("p3", "good"),
               ("p4", "bad")]
    out = evaluate_quorum(results, quorum=2)
    econ.record_quorum_outcome(out, job_id="j1",
                               job_price_usd=D("1"))
    # Pot of 1 split across 3: shares sum EXACTLY to the pot.
    awards = sum(led.get_wallet(p).available_usd - D("5")
                 for p in ("p1", "p2", "p3"))
    assert awards == D("1")
    assert led.verify_balances()["ok"]


def test_unstaked_provider_slash_takes_nothing_but_reputation_hits(
        tmp_path):
    econ, led, _ = _econ(tmp_path)
    econ.stake("p1", D("5"))
    econ.stake("p2", D("5"))            # p3 never staked
    out = _accepted_outcome()
    summary = econ.record_quorum_outcome(out, job_id="j1",
                                         job_price_usd=D("2"))
    assert summary["slashed"] == {}     # no bond to take
    rep = econ.reputation("p3")
    assert rep["dissents"] == 1         # the record still lands


# ---------------------------------------------------------------------------
# Unbonding
# ---------------------------------------------------------------------------

def test_unbonding_delay_and_slash_during_unbonding(tmp_path):
    clock = Clock()
    econ, led, _ = _econ(tmp_path, clock=clock)
    econ.stake("p3", D("5"))
    econ.stake("p1", D("5"))
    econ.stake("p2", D("5"))
    econ.request_unstake("p3", D("5"))
    # Not matured: claim yields nothing, funds still staked.
    assert econ.claim_unstaked("p3") == D("0")
    assert led.get_wallet("p3").staked_usd == D("5")
    # Cheat while unbonding → the slash still bites the full bond.
    out = _accepted_outcome()
    econ.record_quorum_outcome(out, job_id="j1",
                               job_price_usd=D("2"))
    assert led.get_wallet("p3").staked_usd == D("3")
    # Mature the clock: the claim shrinks to what the slash left.
    clock.t += econ.unbonding_s + 1
    assert econ.claim_unstaked("p3") == D("3")
    assert led.get_wallet("p3").staked_usd == D("0")
    assert led.verify_balances()["ok"]


def test_cannot_unbond_more_than_staked(tmp_path):
    econ, _, _ = _econ(tmp_path)
    econ.stake("p1", D("3"))
    econ.request_unstake("p1", D("2"))
    with pytest.raises(ValueError, match="cannot unbond"):
        econ.request_unstake("p1", D("2"))    # only 1 not yet unbonding


# ---------------------------------------------------------------------------
# Reputation, quarantine, eligibility
# ---------------------------------------------------------------------------

def test_quarantine_trips_on_repeated_dissent_and_expires(tmp_path):
    clock = Clock()
    econ, led, _ = _econ(tmp_path, clock=clock)
    for p in ("p1", "p2", "p3"):
        econ.stake(p, D("9"))
    for i in range(3):
        out = _accepted_outcome()
        summary = econ.record_quorum_outcome(
            out, job_id=f"j{i}", job_price_usd=D("1"))
    assert "p3" in summary["quarantined"]
    ok, why = econ.is_eligible("p3")
    assert not ok and "quarantined" in why
    # Honest providers stay eligible throughout.
    assert econ.is_eligible("p1")[0]
    # Quarantine expires.
    clock.t += econ.quarantine_s + 1
    ok2, _ = econ.is_eligible("p3")
    assert ok2 or "stake" in _    # eligible again unless bond now short


def test_required_stake_rises_with_dissent_history(tmp_path):
    econ, _, _ = _econ(tmp_path)
    for p in ("p1", "p2", "p3"):
        econ.stake(p, D("5"))
    base = econ.required_stake("p3")
    econ.record_quorum_outcome(_accepted_outcome(), job_id="j1",
                               job_price_usd=D("1"))
    assert econ.required_stake("p3") == base * 2


def test_eligibility_requires_stake(tmp_path):
    econ, _, _ = _econ(tmp_path)
    ok, why = econ.is_eligible("p1")     # funded but never staked
    assert not ok and "stake" in why
    econ.stake("p1", econ.base_stake_usd)
    assert econ.is_eligible("p1")[0]


def test_state_survives_restart(tmp_path):
    clock = Clock()
    econ, led, _ = _econ(tmp_path, clock=clock)
    for p in ("p1", "p2", "p3"):
        econ.stake(p, D("9"))
    for i in range(3):
        econ.record_quorum_outcome(_accepted_outcome(),
                                   job_id=f"j{i}",
                                   job_price_usd=D("1"))
    econ.request_unstake("p1", D("2"))
    econ2 = EconomicLayer(led, state_dir=str(tmp_path / "ledger"),
                          now_fn=clock)
    rep = econ2.reputation("p3")
    assert rep["dissents"] == 3 and rep["quarantined"]
    assert econ2.reputation("p1")["unbonding_usd"] == "2"
    assert len(econ2.slash_journal()) == 3


# ---------------------------------------------------------------------------
# Auction gate
# ---------------------------------------------------------------------------

def test_stake_endpoints_full_lifecycle_on_real_node(tmp_path,
                                                     monkeypatch):
    """faucet → stake → status → unstake → claim, over the REAL node
    app (same fixture shape as test_swarm_auth)."""
    monkeypatch.delenv("PLUGINFER_SWARM_KEY", raising=False)
    monkeypatch.delenv("PLUGINFER_NODE_ADMIN_KEY", raising=False)
    monkeypatch.setenv("PLUGINFER_LEDGER_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from core.tokenomics import Wallet
    from tools.auto_mesh import build_node_app
    w = Wallet()
    app, svc = build_node_app(my_pubkey=w.public_key_pem, my_wallet=w,
                              node_id="econ-test")
    # Deterministic clock for the unbonding claim.
    clock = Clock()
    svc.economics.now = clock
    with TestClient(app) as c:
        assert c.post("/v1/testnet/faucet",
                      json={"wallet_id": "prov"}).status_code == 200
        r = c.post("/v1/stake", json={"wallet_id": "prov",
                                      "amount_usd": "5"})
        assert r.status_code == 200
        assert r.json()["staked_usd"] == "5"
        # Over-stake refused with the honest 402.
        assert c.post("/v1/stake", json={"wallet_id": "prov",
                                         "amount_usd": "100"}
                      ).status_code == 402
        status = c.get("/v1/stake/prov").json()
        assert status["staked_usd"] == "5" and status["eligible"]
        r2 = c.post("/v1/stake/unstake", json={"wallet_id": "prov",
                                               "amount_usd": "2"})
        assert r2.status_code == 200
        # Immature claim yields zero; matured claim releases.
        assert c.post("/v1/stake/claim", json={"wallet_id": "prov"}
                      ).json()["claimed_usd"] == "0"
        clock.t += svc.economics.unbonding_s + 1
        assert c.post("/v1/stake/claim", json={"wallet_id": "prov"}
                      ).json()["claimed_usd"] == "2"
        assert c.get("/v1/economics/slashes").json()["slashes"] == []
        # The whole dance kept the auditable ledger green.
        assert c.get("/v1/ledger/verify").json()["ok"]


def test_auction_excludes_ineligible_providers(tmp_path):
    from core.providers import Auction, Bid, JobSpec, Provider

    class FakeProvider(Provider):
        def __init__(self, pid):
            self.provider_id = pid

        def bid(self, job):
            return Bid(provider_id=self.provider_id,
                       price_usd=0.01, eta_ms=1000,
                       expected_quality=0.9, privacy_grade="public")

        def execute(self, job):          # pragma: no cover
            raise NotImplementedError

    auction = Auction()
    auction.register(FakeProvider("good"))
    auction.register(FakeProvider("banned"))
    auction.eligibility_fn = lambda pid: (
        (False, "quarantined") if pid == "banned" else (True, "ok"))
    res = auction.run(JobSpec(job_id="j1", kind="compute.echo",
                              payload={}))
    assert res.winner.provider_id == "good"
    reasons = {r["provider_id"]: r["reason"] for r in res.rejected}
    assert "ineligible: quarantined" in reasons["banned"]
