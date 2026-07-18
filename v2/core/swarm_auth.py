"""Private-swarm authentication — the mesh's front door lock.

Set ``PLUGINFER_SWARM_KEY`` on every node of a private mesh (e.g. one
company's datacenters) and the node refuses ALL mesh traffic that does
not present the key — joins, job submissions, ledger reads, everything.
Unset (the default), nothing changes: the public mesh stays open.

Honest scope, stated plainly:

* This is a **shared symmetric key** — the API-key trust model. It keeps
  strangers out; it does not give per-node identity or revocation (that
  is a later milestone: per-node certs). One key, one swarm.
* The key travels as a request header, so the transport MUST be TLS —
  which every supported path already is (`--share` tunnels are https;
  a reverse proxy terminates TLS in front of a datacenter node). Running
  plain HTTP across the internet would expose the key on the wire; we
  say so rather than pretend otherwise.
* Local operator exception: a request from loopback with no forwarding
  headers (X-Forwarded-For / CF-Connecting-IP / X-Real-IP / Forwarded)
  is the operator on the machine itself and passes without the key, so
  `pluginfer up` and the control panel keep working. Tunnels and proxies
  add forwarding headers, so remote traffic cannot masquerade as local
  by connecting to the tunnel's loopback leg.
"""

from __future__ import annotations

import hmac
import os
from typing import Dict, Mapping, Optional

SWARM_KEY_ENV = "PLUGINFER_SWARM_KEY"
SWARM_KEY_HEADER = "x-pluginfer-swarm-key"

# Endpoints that stay open in private mode:
#   /healthz — liveness for load balancers / peers' dead-node sweep.
#   /        — the control panel's static HTML (contains no data; every
#              piece of data it shows comes from gated API calls, which
#              the panel retries with the key the operator types in).
OPEN_PATHS = ("/healthz", "/")

_FORWARD_HEADERS = ("x-forwarded-for", "cf-connecting-ip",
                    "x-real-ip", "forwarded")

_LOOPBACK = ("127.0.0.1", "::1", "localhost")


def swarm_key() -> Optional[str]:
    """The configured swarm key, or None when the mesh is public."""
    k = os.environ.get(SWARM_KEY_ENV, "").strip()
    return k or None


def auth_headers() -> Dict[str, str]:
    """Headers every OUTBOUND mesh call must carry in private mode."""
    k = swarm_key()
    return {SWARM_KEY_HEADER.title(): k} if k else {}


def is_authorized(headers: Mapping[str, str],
                  client_host: Optional[str],
                  path: str = "") -> bool:
    """Gate decision for one inbound request. Case-insensitive headers
    expected lowercase (Starlette provides that)."""
    key = swarm_key()
    if key is None:
        return True
    if path in OPEN_PATHS:
        return True
    presented = headers.get(SWARM_KEY_HEADER, "")
    if presented and hmac.compare_digest(presented, key):
        return True
    # Local operator on the node's own machine.
    if client_host in _LOOPBACK and not any(
            h in headers for h in _FORWARD_HEADERS):
        return True
    return False


__all__ = ["swarm_key", "auth_headers", "is_authorized",
           "SWARM_KEY_ENV", "SWARM_KEY_HEADER", "OPEN_PATHS"]
