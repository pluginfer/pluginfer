"""HG22 (transport leg) — first-class TLS for gateway + node.

Pins: half-configured TLS refuses startup (never silent plaintext),
gencert mints a loadable self-signed pair, and — decisively — a real
uvicorn server started with those kwargs answers over HTTPS.
"""

from __future__ import annotations

import json
import ssl
import sys
import threading
import time
import urllib.request
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

from governance.tls import (TLSConfigError, generate_self_signed,
                            tls_kwargs)


def test_tls_kwargs_unset_means_plain_http(monkeypatch):
    monkeypatch.delenv("PLUGINFER_GW_TLS_CERT", raising=False)
    monkeypatch.delenv("PLUGINFER_GW_TLS_KEY", raising=False)
    assert tls_kwargs("PLUGINFER_GW") == {}


def test_half_configured_tls_refuses(monkeypatch, tmp_path):
    cert = tmp_path / "c.pem"
    cert.write_text("x")
    monkeypatch.setenv("PLUGINFER_GW_TLS_CERT", str(cert))
    monkeypatch.delenv("PLUGINFER_GW_TLS_KEY", raising=False)
    with pytest.raises(TLSConfigError, match="half-configured"):
        tls_kwargs("PLUGINFER_GW")
    monkeypatch.setenv("PLUGINFER_GW_TLS_KEY", str(tmp_path / "no.pem"))
    with pytest.raises(TLSConfigError, match="not found"):
        tls_kwargs("PLUGINFER_GW")


def test_gencert_produces_loadable_pair(tmp_path):
    paths = generate_self_signed(tmp_path, common_name="test-node")
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
    cert = x509.load_pem_x509_certificate(
        Path(paths["cert"]).read_bytes())
    assert "test-node" in cert.subject.rfc4514_string()
    key = serialization.load_pem_private_key(
        Path(paths["key"]).read_bytes(), password=None)
    assert key is not None


def test_real_https_round_trip(tmp_path, monkeypatch):
    """Decisive: a live uvicorn with the minted cert answers HTTPS."""
    import uvicorn
    from fastapi import FastAPI

    paths = generate_self_signed(tmp_path)
    monkeypatch.setenv("PLUGINFER_NODE_TLS_CERT", paths["cert"])
    monkeypatch.setenv("PLUGINFER_NODE_TLS_KEY", paths["key"])
    ssl_kw = tls_kwargs("PLUGINFER_NODE")
    assert set(ssl_kw) == {"ssl_certfile", "ssl_keyfile"}

    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"tls": True}

    config = uvicorn.Config(app, host="127.0.0.1", port=0,
                            log_level="error", **ssl_kw)
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        deadline = time.time() + 10
        port = None
        while time.time() < deadline and port is None:
            if server.started and server.servers:
                socks = server.servers[0].sockets
                if socks:
                    port = socks[0].getsockname()[1]
                    break
            time.sleep(0.05)
        assert port, "server never started"
        # Self-signed: the client pins-or-skips verification KNOWINGLY
        # (this is a test client talking to its own just-minted cert).
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(f"https://127.0.0.1:{port}/ping",
                                    context=ctx, timeout=5) as r:
            assert json.loads(r.read()) == {"tls": True}
    finally:
        server.should_exit = True
        t.join(timeout=10)
        assert not t.is_alive()
