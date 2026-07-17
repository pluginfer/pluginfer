"""DHT bootstrap glue.

`bootstrap_dht_from_seeds` is the high-level entry point that:
  1. Pulls a peer list from the seed_node infrastructure.
  2. Returns the peers in a format ready for `core.kademlia.KademliaNode.bootstrap`.

The actual DHT join / find_node walk is done by the existing
`core.kademlia` module (already substantial; reviewed in WORKLOG W6
batch 1 and 6). This file just glues seed bootstrap to that.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def bootstrap_dht_from_seeds(
    bootstrap_seeds: list[dict],
    *,
    pubkey_pem: str,
    sign_fn,
    local_ip: str,
    local_port: int,
    node_version: str,
) -> list[dict]:
    """Register self with each seed; pull peer lists; return aggregated peers.

    `bootstrap_seeds` is the BOOTSTRAP_SEEDS list from
    `core.complete_mesh_controller`. Returns a deduplicated list of
    {ip, port, pubkey_pem} dicts ready for further DHT walking.
    """
    from infrastructure.seed_node import seed_client as _sc

    seen: set[tuple[str, int]] = set()
    out: list[dict] = []

    for entry in bootstrap_seeds:
        seed = _sc.SeedAddress(host=entry["host"], port=entry.get("port", 9000))
        try:
            _sc.register_sync(
                seed,
                pubkey_pem=pubkey_pem,
                sign_fn=sign_fn,
                ip=local_ip,
                port=local_port,
                node_version=node_version,
            )
            peers = _sc.fetch_peers_sync(seed, max_n=50)
        except Exception as e:
            logger.warning(
                "[DHT-BOOTSTRAP] seed %s:%s unreachable: %s",
                seed.host, seed.port, e,
            )
            continue
        for p in peers:
            key = (p.get("ip"), int(p.get("port", 0)))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "ip": p["ip"],
                    "port": p["port"],
                    "pubkey_pem": p.get("pubkey_pem"),
                }
            )
    return out
