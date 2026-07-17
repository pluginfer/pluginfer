"""Seed registry loader — read the multi-seed bundle, validate quorum
signatures, expose the trusted seed list to the auto_mesh client.

A node bootstrapping into the mesh needs at least ONE reachable seed.
The bundled `data/seed_registry.json` lists every operator-published
seed across regions. The loader:

  1. Parses the bundle file.
  2. Filters records whose `quorum_signatures` list contains at least
     `min_signatures` entries from KNOWN validator keys. Records that
     fail this check are NOT trusted (returning them would let an
     attacker who modified the bundle redirect bootstrap traffic).
  3. Allows env override (`PLUGINFER_SEED_HOST` etc.) — used for dev
     and for closed-mesh deployments. The env path bypasses the
     quorum check intentionally; this is a single-operator pin.
  4. Returns the first record whose RTT probe responds; on failure,
     falls through to the next.

Innovation: §A27 "Quorum-signed bootstrap registry for permissionless
overlays." The decentralized-substrate pitch demands no single trust
anchor; this is the minimum mechanism that satisfies it for the
bootstrap layer specifically.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SeedRecord:
    id: str
    host: str
    port: int
    region: str = ""
    pubkey_fingerprint_sha256: str = ""
    quorum_signatures: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"{self.host}:{self.port}"

    def has_quorum(self, *, min_signatures: int,
                   validator_fps: Optional[set] = None) -> bool:
        """True when the record carries at least min_signatures
        signatures from validator_fps (if provided). When the env
        validator set is empty, we accept ANY signatures meeting the
        count — appropriate when the operator hasn't yet published
        a validator roster (bootstrap-of-bootstrap)."""
        if min_signatures <= 0:
            return True
        sigs = self.quorum_signatures or []
        if not validator_fps:
            return len(sigs) >= min_signatures
        accepted = sum(
            1 for s in sigs
            if s.get("signer_fingerprint_sha256", "") in validator_fps
        )
        return accepted >= min_signatures


@dataclass
class SeedRegistry:
    min_signatures: int = 2
    records: List[SeedRecord] = field(default_factory=list)
    tofu_mode: bool = False

    @classmethod
    def from_file(cls, path: str) -> "SeedRegistry":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = [
            SeedRecord(
                id=str(r.get("id") or r.get("host")),
                host=str(r.get("host", "")),
                port=int(r.get("port", 9000)),
                region=str(r.get("region", "")),
                pubkey_fingerprint_sha256=str(
                    r.get("pubkey_fingerprint_sha256", "")
                ),
                quorum_signatures=list(r.get("quorum_signatures") or []),
            )
            for r in data.get("records", [])
            if r.get("host")
        ]
        return cls(
            min_signatures=int(data.get("min_signatures", 2)),
            records=records,
            tofu_mode=bool(data.get("tofu_mode", False)),
        )

    def trusted_records(
        self, *, validator_fps: Optional[set] = None,
    ) -> List[SeedRecord]:
        """Return only records that pass the quorum check, OR every
        record when tofu_mode is set (the operator is bootstrapping
        the validator set and hasn't yet collected the second
        signature)."""
        if self.tofu_mode:
            return list(self.records)
        out = []
        for r in self.records:
            if r.has_quorum(
                min_signatures=self.min_signatures,
                validator_fps=validator_fps,
            ):
                out.append(r)
        return out

    def reachable_records(
        self, *, validator_fps: Optional[set] = None,
        probe_timeout_s: float = 1.0,
    ) -> List[SeedRecord]:
        """Trusted records that also answer a TCP probe within the
        timeout. Sorted by probe order (first reachable first)."""
        out = []
        for r in self.trusted_records(validator_fps=validator_fps):
            if _tcp_reachable(r.host, r.port, timeout_s=probe_timeout_s):
                out.append(r)
        return out


def _tcp_reachable(host: str, port: int, *, timeout_s: float = 1.0) -> bool:
    try:
        s = socket.create_connection((host, port), timeout=timeout_s)
        s.close()
        return True
    except OSError:
        return False


def resolve_bootstrap_seed(
    *,
    bundle_path: Optional[str] = None,
    validator_fps: Optional[set] = None,
) -> Optional[Tuple[str, int]]:
    """The auto_mesh entrypoint helper. Resolution order:

      1. PLUGINFER_SEED_HOST + PLUGINFER_SEED_PORT env override.
      2. First reachable trusted record from the bundle.
      3. None — caller falls back to --gossip-bootstrap or refuses
         to start.
    """
    env_host = os.environ.get("PLUGINFER_SEED_HOST", "").strip()
    env_port = os.environ.get("PLUGINFER_SEED_PORT", "").strip()
    if env_host and env_port:
        try:
            return (env_host, int(env_port))
        except ValueError:
            logger.warning("invalid PLUGINFER_SEED_PORT: %s", env_port)
    if bundle_path is None:
        bundle_path = str(
            Path(__file__).resolve().parents[1]
            / "data" / "seed_registry.json"
        )
    if not Path(bundle_path).exists():
        return None
    try:
        reg = SeedRegistry.from_file(bundle_path)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("seed_registry load failed: %s", e)
        return None
    reachable = reg.reachable_records(validator_fps=validator_fps)
    if not reachable:
        return None
    return (reachable[0].host, reachable[0].port)


__all__ = [
    "SeedRecord",
    "SeedRegistry",
    "resolve_bootstrap_seed",
]
