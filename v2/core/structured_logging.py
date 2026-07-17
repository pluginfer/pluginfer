"""Structured JSON logging.

Stdlib logging already speaks JSON if we hand it a properly-shaped
formatter. We keep the dep tree minimal -- no `structlog` -- because
the seed-node Docker image must stay <50MB.

Usage:

    from core.structured_logging import configure, get_logger
    configure(level="INFO", json=True, sink=sys.stderr)
    log = get_logger(__name__)
    log.info("peer_connected", extra={"peer_id": "abc", "rtt_ms": 23})

Every record carries: timestamp, level, component (logger name),
event (msg) plus any `extra=` fields. Sensitive fields (wallet
private key, env tokens) MUST never be passed in `extra` -- the
formatter has a small allowlist as defense in depth.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional, TextIO

# Fields we will redact from any log record because they are CWE-532
# vectors. Add more here as the surface grows.
_REDACT_FIELD_PATTERNS = (
    "private_key", "secret", "passphrase", "api_key", "session_id",
    "wallet_priv", "raw_token",
)


def _redact_value(v: Any) -> Any:
    if isinstance(v, str) and len(v) > 8:
        return v[:4] + "..." + v[-2:]
    return "<redacted>"


def _scrub(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        kl = k.lower()
        if any(pat in kl for pat in _REDACT_FIELD_PATTERNS):
            out[k] = _redact_value(v)
        else:
            out[k] = v
    return out


class JSONFormatter(logging.Formatter):
    """Emits one JSON object per log record."""

    def __init__(self, *, node_id: Optional[str] = None) -> None:
        super().__init__()
        self.node_id = node_id

    def format(self, record: logging.LogRecord) -> str:
        body: Dict[str, Any] = {
            "ts": time.time(),
            "level": record.levelname,
            "component": record.name,
            "event": record.getMessage(),
        }
        if self.node_id:
            body["node_id"] = self.node_id
        # Pull anything passed via extra= into the JSON body.
        for k, v in record.__dict__.items():
            if k in (
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "levelname", "levelno",
                "lineno", "message", "module", "msecs", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName",
                "taskName",
            ):
                continue
            body[k] = v
        if record.exc_info:
            body["exc"] = self.formatException(record.exc_info)
        return json.dumps(_scrub(body), default=str)


def configure(
    *,
    level: str = "INFO",
    json: bool = True,
    sink: Optional[TextIO] = None,
    node_id: Optional[str] = None,
) -> None:
    """Reset the root logger with a JSON handler.

    Idempotent: clears existing handlers so calling twice doesn't
    multiply log lines.
    """
    sink = sink or sys.stderr
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    h = logging.StreamHandler(sink)
    h.setFormatter(JSONFormatter(node_id=node_id) if json else logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    ))
    root.addHandler(h)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_from_env() -> None:
    """Convenience: read PLUGINFER_LOG_LEVEL + PLUGINFER_LOG_JSON env."""
    configure(
        level=os.environ.get("PLUGINFER_LOG_LEVEL", "INFO"),
        json=os.environ.get("PLUGINFER_LOG_JSON", "1") not in ("0", "false", "False"),
        node_id=os.environ.get("PLUGINFER_NODE_ID"),
    )
