"""Structured-logging tests."""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from core.structured_logging import (  # noqa: E402
    JSONFormatter,
    configure,
    get_logger,
)


def test_json_formatter_emits_one_object_per_record():
    f = JSONFormatter(node_id="node-A")
    rec = logging.LogRecord(
        name="core.test", level=logging.INFO, pathname="x", lineno=1,
        msg="peer_connected", args=(), exc_info=None,
    )
    out = f.format(rec)
    d = json.loads(out)
    assert d["level"] == "INFO"
    assert d["component"] == "core.test"
    assert d["event"] == "peer_connected"
    assert d["node_id"] == "node-A"


def test_extra_fields_propagate():
    sink = io.StringIO()
    configure(level="INFO", json=True, sink=sink, node_id="n-1")
    log = get_logger("test.extra")
    log.info("auction_won", extra={"job_id": "abc", "price_usd": 0.01})
    line = sink.getvalue().strip()
    d = json.loads(line)
    assert d["job_id"] == "abc"
    assert d["price_usd"] == 0.01


def test_sensitive_fields_redacted():
    """Wallet private key / API key / passphrase MUST be redacted."""
    sink = io.StringIO()
    configure(level="INFO", json=True, sink=sink)
    log = get_logger("test.redact")
    log.info("auth_event", extra={
        "wallet_priv_key": "abcdef1234567890abcdef",
        "session_id": "sess-supersecretvalue1234",
        "user_email": "ok@example.com",
    })
    d = json.loads(sink.getvalue().strip().splitlines()[-1])
    # The key prefix is preserved, the bulk is redacted.
    assert "abcdef1234567890abcdef" not in d["wallet_priv_key"]
    assert "sess-supersecretvalue1234" not in d["session_id"]
    # Non-sensitive fields untouched.
    assert d["user_email"] == "ok@example.com"


def test_configure_is_idempotent():
    sink1 = io.StringIO()
    sink2 = io.StringIO()
    configure(level="INFO", json=True, sink=sink1)
    configure(level="INFO", json=True, sink=sink2)
    log = get_logger("test.idem")
    log.info("hello")
    # Only the second sink received the log line; the first handler
    # was removed by the second configure() call.
    assert sink1.getvalue() == ""
    assert sink2.getvalue() != ""
