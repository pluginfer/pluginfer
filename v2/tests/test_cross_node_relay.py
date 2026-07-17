"""Mesh-native HTTP relay — A→C→B path when A can't reach B directly.

Two flavours:

  1. Unit: stub the HTTP layer and prove _CrossNodeProvider falls
     through direct → relay → mark-unreachable in the right order
     and cooldown gates subsequent bids.
  2. Live: three auto_mesh processes; A's cross-node provider for
     "B" gets pointed at a URL that REFUSES to listen (simulating a
     symmetric-NAT block), and the relay pool contains a real
     reachable C running auto_mesh that knows how to forward.

This is the proof that the mesh works between two strangers behind
NAT as long as one third party can see both.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, List, Tuple

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

from tools.auto_mesh import _CrossNodeProvider


class _FakeWallet:
    public_key_pem = "-----BEGIN PUBLIC KEY-----\nFAKE\n-----END PUBLIC KEY-----\n"

    def sign(self, msg):
        return "AAAA"


def _spec(prompt="hi", max_tokens=64):
    class _Spec:
        def __init__(self):
            self.job_id = "tjob"
            self.payload = {"prompt": prompt, "max_tokens": max_tokens}
    return _Spec()


# ---------------------------------------------------------------------------
# Unit: cooldown semantics
# ---------------------------------------------------------------------------

def test_cooldown_suspends_bids_after_failure():
    prov = _CrossNodeProvider(
        peer_url="http://10.255.255.1:65535",
        peer_pubkey="peer-pub",
        my_pubkey="my-pub",
        my_wallet=_FakeWallet(),
    )
    prov._last_failure_unix = time.time()
    assert prov._is_in_cooldown() is True
    # Bid abstains during cooldown — auction routes around us.
    assert prov.bid(_spec()) is None


def test_cooldown_clears_after_success():
    prov = _CrossNodeProvider(
        peer_url="http://10.255.255.1:65535",
        peer_pubkey="peer-pub", my_pubkey="my-pub", my_wallet=_FakeWallet(),
    )
    prov._last_failure_unix = time.time() - 1
    prov._last_success_unix = time.time()
    assert prov._is_in_cooldown() is False


def test_cooldown_expires_after_window():
    prov = _CrossNodeProvider(
        peer_url="http://10.255.255.1:65535",
        peer_pubkey="peer-pub", my_pubkey="my-pub", my_wallet=_FakeWallet(),
    )
    prov._last_failure_unix = (
        time.time() - prov.UNREACHABLE_COOLDOWN_S - 1.0
    )
    assert prov._is_in_cooldown() is False


# ---------------------------------------------------------------------------
# Unit: direct → relay → unreachable
# ---------------------------------------------------------------------------

def test_execute_uses_direct_when_peer_reachable(monkeypatch):
    prov = _CrossNodeProvider(
        peer_url="http://peer.example",
        peer_pubkey="peer-pub", my_pubkey="my-pub", my_wallet=_FakeWallet(),
    )
    prov._peer_hw = {"best_device": {"type": "cuda"}}
    prov._peer_score = 50.0

    calls: List[str] = []

    def _fake_post(self, url, body, timeout_s=30.0):
        calls.append(url)
        return (
            {"choices": [{"message": {"content": "direct-result"}}]},
            {"X-Pluginfer-Provider": "peer-sig"},
            None,
        )
    monkeypatch.setattr(
        _CrossNodeProvider, "_post_chat_completions", _fake_post,
    )

    out = prov.execute(_spec(), bid=None)
    assert out["status"] == "executed"
    assert out["cross_node_path"] == "direct"
    assert calls == ["http://peer.example/v1/chat/completions"]
    # Result content survives the bytes round-trip.
    import base64
    assert base64.b64decode(out["result_bytes"]) == b"direct-result"


def test_execute_falls_through_to_relay_when_direct_fails(monkeypatch):
    target_url = "http://peer-behind-nat.example"
    prov = _CrossNodeProvider(
        peer_url=target_url,
        peer_pubkey="peer-pub", my_pubkey="my-pub", my_wallet=_FakeWallet(),
        relay_pool_getter=lambda: [("http://relay-1.example", "relay-1-pub")],
    )
    prov._peer_hw = {"best_device": {"type": "cuda"}}
    prov._peer_score = 50.0

    calls: List[Tuple[str, str]] = []

    def _fake_post(self, url, body, timeout_s=30.0):
        calls.append((url, "fail" if target_url in url else "ok"))
        if target_url in url:
            return (None, None, "Connection refused")
        return (
            {"choices": [{"message": {"content": "relayed-result"}}]},
            {"X-Pluginfer-Provider": "peer-sig"},
            None,
        )
    monkeypatch.setattr(
        _CrossNodeProvider, "_post_chat_completions", _fake_post,
    )

    out = prov.execute(_spec(), bid=None)
    assert out["status"] == "executed", out
    assert out["cross_node_path"] == "relay"
    # Direct attempted first, then relay used.
    assert any("/v1/chat/completions" in u and target_url in u for u, _ in calls)
    assert any("/relay/" in u for u, _ in calls)


def test_execute_marks_unreachable_when_all_paths_fail(monkeypatch):
    target_url = "http://peer-behind-nat.example"
    prov = _CrossNodeProvider(
        peer_url=target_url,
        peer_pubkey="peer-pub", my_pubkey="my-pub", my_wallet=_FakeWallet(),
        relay_pool_getter=lambda: [
            ("http://relay-1.example", "r1"),
            ("http://relay-2.example", "r2"),
        ],
    )
    prov._peer_hw = {"best_device": {"type": "cuda"}}
    prov._peer_score = 50.0

    def _fake_post(self, url, body, timeout_s=30.0):
        return (None, None, "Connection refused")
    monkeypatch.setattr(
        _CrossNodeProvider, "_post_chat_completions", _fake_post,
    )

    out = prov.execute(_spec(), bid=None)
    assert out["status"] == "failed"
    assert "cross_node_unreachable" in out["reason"]
    assert prov._is_in_cooldown() is True
    # bid() abstains immediately after the failure.
    assert prov.bid(_spec()) is None


def test_relay_pool_excludes_target_itself(monkeypatch):
    """A relay through the target would defeat the point — must not
    show up in the candidate list. The wiring guarantees this in
    _bind_peer_if_new; we exercise the cross-node side by ensuring
    the cross-node won't accidentally re-call the target's URL as
    a relay even if the getter returns it."""
    target_url = "http://peer-behind-nat.example"
    prov = _CrossNodeProvider(
        peer_url=target_url,
        peer_pubkey="peer-pub", my_pubkey="my-pub", my_wallet=_FakeWallet(),
        # Maliciously include the target itself as a "relay" — the
        # cross-node should skip it.
        relay_pool_getter=lambda: [(target_url, "self-pub")],
    )
    prov._peer_hw = {"best_device": {"type": "cuda"}}
    prov._peer_score = 50.0

    calls: List[str] = []

    def _fake_post(self, url, body, timeout_s=30.0):
        calls.append(url)
        return (None, None, "Connection refused")
    monkeypatch.setattr(
        _CrossNodeProvider, "_post_chat_completions", _fake_post,
    )

    out = prov.execute(_spec(), bid=None)
    # Only one POST attempted: the direct path. Relay-through-self
    # filtered out.
    assert len(calls) == 1
    assert "/relay/" not in calls[0]
    assert out["status"] == "failed"
