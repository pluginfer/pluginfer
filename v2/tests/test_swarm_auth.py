"""Private-swarm authentication — strangers out, key holders in.

Covers the gate decision (unit), the middleware on the REAL node app
(integration), and the public-mode no-op that keeps today's behavior.
"""

import pytest


# ---------------------------------------------------------------------------
# Gate decision
# ---------------------------------------------------------------------------

def test_public_mode_allows_everything(monkeypatch):
    monkeypatch.delenv("PLUGINFER_SWARM_KEY", raising=False)
    from core.swarm_auth import is_authorized
    assert is_authorized({}, "203.0.113.9", path="/v1/jobs")


def test_private_mode_gate_matrix(monkeypatch):
    monkeypatch.setenv("PLUGINFER_SWARM_KEY", "dc-secret")
    from core.swarm_auth import is_authorized
    # Remote without key: refused.
    assert not is_authorized({}, "203.0.113.9", path="/v1/jobs")
    # Remote with wrong key: refused.
    assert not is_authorized({"x-pluginfer-swarm-key": "nope"},
                             "203.0.113.9", path="/v1/jobs")
    # Remote with the key: allowed.
    assert is_authorized({"x-pluginfer-swarm-key": "dc-secret"},
                         "203.0.113.9", path="/v1/jobs")
    # Local operator (loopback, no forwarding headers): allowed.
    assert is_authorized({}, "127.0.0.1", path="/v1/jobs")
    # Tunnel traffic reaches us FROM loopback but carries forwarding
    # headers — must NOT pass as local.
    assert not is_authorized({"x-forwarded-for": "203.0.113.9"},
                             "127.0.0.1", path="/v1/jobs")
    assert not is_authorized({"cf-connecting-ip": "203.0.113.9"},
                             "127.0.0.1", path="/v1/jobs")
    # Open paths: health for load balancers, the static panel shell.
    assert is_authorized({}, "203.0.113.9", path="/healthz")
    assert is_authorized({}, "203.0.113.9", path="/")


def test_auth_headers_follow_key(monkeypatch):
    from core import swarm_auth
    monkeypatch.delenv("PLUGINFER_SWARM_KEY", raising=False)
    assert swarm_auth.auth_headers() == {}
    monkeypatch.setenv("PLUGINFER_SWARM_KEY", "dc-secret")
    assert swarm_auth.auth_headers() == {"X-Pluginfer-Swarm-Key": "dc-secret"}


# ---------------------------------------------------------------------------
# Middleware on the real node app
# ---------------------------------------------------------------------------

@pytest.fixture()
def node_app(monkeypatch, tmp_path):
    monkeypatch.setenv("PLUGINFER_SWARM_KEY", "dc-secret")
    monkeypatch.setenv("PLUGINFER_LEDGER_DIR", str(tmp_path))
    from core.tokenomics import Wallet
    from tools.auto_mesh import build_node_app
    w = Wallet()
    app, _svc = build_node_app(
        my_pubkey=w.public_key_pem, my_wallet=w, node_id="swarm-test")
    return app


def test_node_refuses_mesh_traffic_without_key(node_app):
    from fastapi.testclient import TestClient
    with TestClient(node_app) as c:
        # TestClient's synthetic client host is "testclient" (not
        # loopback), so it exercises the REMOTE path.
        r = c.get("/peers")
        assert r.status_code == 401
        r = c.post("/v1/chat/completions", json={
            "model": "pluginfer-alpha",
            "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 401
        # With the key, the same calls pass the gate.
        h = {"X-Pluginfer-Swarm-Key": "dc-secret"}
        assert c.get("/peers", headers=h).status_code == 200
        # Open paths stay open for load balancers / panel shell.
        assert c.get("/healthz").status_code == 200
        assert c.get("/").status_code == 200


def test_outbound_gossip_carries_key(monkeypatch):
    monkeypatch.setenv("PLUGINFER_SWARM_KEY", "dc-secret")
    captured = {}

    class _Resp:
        status = 200

        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        captured["headers"] = dict(req.header_items())
        return _Resp()

    import core.gossip_discovery as gd
    monkeypatch.setattr(gd.urllib.request, "urlopen", fake_urlopen)
    gd._http_get_json("http://peer.example:8100/peers")
    assert captured["headers"].get("X-pluginfer-swarm-key") == "dc-secret"
