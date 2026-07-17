"""Liquid State grain — the unit of work in the §C protocol.

A grain is one portable, signed, version-tagged training fragment.
It is what flows between Pluginfer nodes the way packets flow
through a network. Everything in §C1-§C8 is expressed in terms of
grains.

A grain carries:
* model_shard_id     - which layer-shard this gradient is for
* version_v          - integer step number; defines staleness
* contributor_id     - public-key fingerprint of the producer
* low_rank_grad      - rank-r gradient as a flat byte buffer
                       (r-by-k float32 in the canonical embodiment)
* shape_meta         - (m, n, r) so receivers can reconstruct
* optimizer_seed     - so a recipient can deterministically resume
* pressure_at_birth  - producer's §B1 pressure scalar at flush time
* signature          - Ed25519 over canonical bytes

Why a single message format?
* Gradients (§C1, §C5) are grains.
* Migrated in-flight work (§C4) is a grain.
* Move-compute-to-data results (§C6) are grains.
* Sun-aggregated regional updates (§C2) are grains.

The signature scheme is intentionally minimal — Ed25519, no
ceremony. Verification is O(1) constant; aggregation cost is
dominated by tensor merge, not crypto.

Stdlib-only: ``hashlib`` + ``hmac`` for the canonical hash; the
signature is a placeholder HMAC under a node-local secret unless
``cryptography`` is available, in which case real Ed25519 is used.
This keeps unit tests CPU-pure and dependency-free.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import struct
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey, Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization
    _HAS_ED25519 = True
except Exception:                                                  # pragma: no cover
    _HAS_ED25519 = False


# Grain protocol version. Bumped when the wire format breaks back-compat.
GRAIN_PROTOCOL_VERSION = 1


@dataclass
class GrainMeta:
    """Everything except the gradient bytes themselves. Hashed for signature."""
    protocol: int = GRAIN_PROTOCOL_VERSION
    grain_id: str = ""           # sha256-prefix of full payload
    model_shard_id: str = ""
    version_v: int = 0
    contributor_id: str = ""
    optimizer_seed: int = 0
    pressure_at_birth: float = 0.0
    shape_m: int = 0
    shape_n: int = 0
    shape_r: int = 0
    dp_epsilon: float = 0.0      # 0.0 = no DP; >0 = noise added before signing
    created_ts: float = 0.0


@dataclass
class Grain:
    """A signed, version-tagged training fragment.

    The gradient is stored as raw bytes to avoid pulling torch into
    the grain layer (so grains can be inspected, gossiped, and
    merged on nodes that don't have torch).
    """
    meta: GrainMeta = field(default_factory=GrainMeta)
    grad_bytes: bytes = b""
    signature: bytes = b""

    # ---- canonicalisation ------------------------------------------------

    def canonical_payload(self) -> bytes:
        """Bytes that the signature attests. Excludes the signature itself
        and the meta.grain_id (which is *derived* from the payload)."""
        d = asdict(self.meta)
        d.pop("grain_id", None)
        head = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
        return head + b"\x00" + self.grad_bytes

    def compute_grain_id(self) -> str:
        return hashlib.sha256(self.canonical_payload()).hexdigest()[:32]

    # ---- signing ---------------------------------------------------------

    def sign(self, key_material: bytes) -> "Grain":
        """Sign with Ed25519 if available, else HMAC-SHA256.

        ``key_material`` is interpreted as the Ed25519 private-key seed
        (32 bytes) when ed25519 is available, else as an HMAC secret.
        """
        self.meta.grain_id = self.compute_grain_id()
        payload = self.canonical_payload()
        if _HAS_ED25519 and len(key_material) >= 32:
            priv = Ed25519PrivateKey.from_private_bytes(key_material[:32])
            self.signature = priv.sign(payload)
        else:
            self.signature = hmac.new(
                key_material, payload, hashlib.sha256,
            ).digest()
        return self

    def verify(self, public_key: bytes) -> bool:
        """Returns True iff signature is valid under public_key.

        With Ed25519: ``public_key`` is the 32-byte raw Ed25519 public
        key. With HMAC: ``public_key`` is the same secret used to sign
        (HMAC has no public/private distinction; this is a soft mode
        for tests).
        """
        if not self.signature:
            return False
        try:
            payload = self.canonical_payload()
            if _HAS_ED25519 and len(public_key) == 32 and len(self.signature) == 64:
                pub = Ed25519PublicKey.from_public_bytes(public_key)
                pub.verify(self.signature, payload)
                return True
            # HMAC fallback: constant-time compare.
            expected = hmac.new(public_key, payload, hashlib.sha256).digest()
            return hmac.compare_digest(expected, self.signature)
        except Exception:
            return False

    # ---- serialisation ---------------------------------------------------

    def to_bytes(self) -> bytes:
        head = json.dumps(asdict(self.meta), sort_keys=True,
                          separators=(",", ":")).encode()
        return (
            struct.pack(">I", len(head)) + head
            + struct.pack(">I", len(self.grad_bytes)) + self.grad_bytes
            + struct.pack(">I", len(self.signature)) + self.signature
        )

    @classmethod
    def from_bytes(cls, blob: bytes) -> "Grain":
        off = 0
        (hl,) = struct.unpack(">I", blob[off:off + 4]); off += 4
        head = blob[off:off + hl]; off += hl
        meta = GrainMeta(**json.loads(head.decode()))
        (gl,) = struct.unpack(">I", blob[off:off + 4]); off += 4
        grad = blob[off:off + gl]; off += gl
        (sl,) = struct.unpack(">I", blob[off:off + 4]); off += 4
        sig = blob[off:off + sl]
        return cls(meta=meta, grad_bytes=grad, signature=sig)

    # ---- migration (§C4) -------------------------------------------------

    def staleness(self, current_v: int) -> int:
        """Number of versions this grain is behind the current one."""
        return max(0, current_v - self.meta.version_v)

    def decay_weight(self, current_v: int, tau: float = 200.0) -> float:
        """Exponential staleness decay used by §C5 NBGGA."""
        import math
        d = self.staleness(current_v)
        return math.exp(-d / max(1.0, tau))


# ----------------------------------------------------------------------------
# Convenience helpers
# ----------------------------------------------------------------------------

def make_grain(
    *,
    model_shard_id: str,
    version_v: int,
    contributor_id: str,
    optimizer_seed: int,
    pressure: float,
    grad_low_rank,                # shape (m, r) or (r, n); see callers
    full_shape: tuple[int, int],
    dp_epsilon: float = 0.0,
) -> Grain:
    """Build an unsigned grain from a low-rank gradient tensor.

    ``grad_low_rank`` may be a numpy array, a torch tensor, or any
    object with ``.tobytes()``. We always serialise as flat float32.
    """
    # Lazy float32 byte conversion. Avoids hard torch/numpy dep here.
    flat = _to_float32_bytes(grad_low_rank)
    m, n = full_shape
    # We accept either (r,n) or (m,r) shape; r is whichever dim isn't m or n.
    try:
        rows, cols = grad_low_rank.shape
    except Exception:
        # 1-D fallback: treat as (1, len)
        rows, cols = 1, len(flat) // 4
    if rows == m:
        r = cols
    elif cols == n:
        r = rows
    else:
        # Generic: r = min(rows, cols)
        r = min(rows, cols)
    meta = GrainMeta(
        model_shard_id=model_shard_id,
        version_v=version_v,
        contributor_id=contributor_id,
        optimizer_seed=optimizer_seed,
        pressure_at_birth=float(pressure),
        shape_m=int(m),
        shape_n=int(n),
        shape_r=int(r),
        dp_epsilon=float(dp_epsilon),
        created_ts=time.time(),
    )
    g = Grain(meta=meta, grad_bytes=flat)
    g.meta.grain_id = g.compute_grain_id()
    return g


def _to_float32_bytes(arr) -> bytes:
    """Accept numpy / torch / list-of-lists, return float32 little-endian bytes."""
    # Try torch first (most common in this repo).
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            return arr.detach().to(torch.float32).cpu().numpy().tobytes()
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(arr, np.ndarray):
            return arr.astype("float32").tobytes()
        return np.asarray(arr, dtype="float32").tobytes()
    except Exception:
        # Stdlib fallback: list of lists of floats.
        flat = []
        for row in arr:
            try:
                flat.extend(row)
            except TypeError:
                flat.append(row)
        return struct.pack(f"<{len(flat)}f", *(float(x) for x in flat))


def grad_from_grain(grain: Grain):
    """Reconstruct a numpy array from grain.grad_bytes.

    Returns a 2-D array shaped (rows, cols) where rows*cols == numel.
    Caller knows whether to treat as (m, r) or (r, n) from the shape
    metadata.
    """
    import numpy as np
    flat = np.frombuffer(grain.grad_bytes, dtype="<f4")
    m, n, r = grain.meta.shape_m, grain.meta.shape_n, grain.meta.shape_r
    # Decide orientation: if m*r == numel, it's (m, r); else (r, n).
    if m * r == flat.size:
        return flat.reshape(m, r)
    if r * n == flat.size:
        return flat.reshape(r, n)
    # Fall back to most-square reshape.
    side = int(round(flat.size ** 0.5))
    if side * side == flat.size:
        return flat.reshape(side, side)
    return flat.reshape(1, flat.size)


def fresh_keypair() -> tuple[bytes, bytes]:
    """Return (private_seed, public_key). Random Ed25519 if available."""
    if _HAS_ED25519:
        priv = Ed25519PrivateKey.generate()
        pub = priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        seed = priv.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return (seed, pub)
    # HMAC fallback: secret == public.
    s = os.urandom(32)
    return (s, s)
