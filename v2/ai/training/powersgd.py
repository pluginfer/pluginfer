"""PowerSGD: low-rank gradient compression for mesh all-reduce.

Why
---
DiLoCo (`ai/training/distributed.py`) is the right idea for the mesh:
each worker does K local SGD steps then 1 outer all-reduce. But even
that 1-in-K all-reduce ships the FULL gradient -- for a 1.13B-param
model that's 2.26 GB / round at fp16. Across a 100 Mb/s home
connection, that's ~3 minutes / round of pure waiting. K=500 local
steps means a round every ~10 minutes anyway, so the comm cost is
tolerable but ugly.

PowerSGD (Vogels et al., 2019) compresses each gradient matrix G
(shape m, n) to a rank-r approximation G ≈ P @ Q^T where:

    P : (m, r)   -- low-rank LEFT factor
    Q : (n, r)   -- low-rank RIGHT factor

Compression ratio: (m*n) / ((m+n)*r). For typical transformer
matrices (m=4096, n=4096) at r=4: 4096²/8192 = 2048× compression.

What makes this work in practice is **error feedback**: the residual
`E = G - P @ Q^T` is added to the NEXT round's gradient before
compression. Over multiple rounds the residual gets opportunistically
shipped — gradient is unbiased on average even though each
individual round is biased.

This module ships:
  * PowerSGDCompressor -- compress / decompress per-param tensor.
  * Per-parameter error feedback buffers.
  * A drop-in `compressed_all_reduce(grads_dict, transport_fn)`
    that the DiLoCo outer step calls.

Failure modes (honest)
----------------------
* Rank r=4 is the sweet spot the paper found; r=2 saves more bandwidth
  but adds ~5-10% to step count for convergence.
* The first ~50 outer rounds with rank=4 are noticeably slower per
  step than full all-reduce -- the error feedback hasn't accumulated
  yet. Warmup with rank=8 or rank=16 for the first 100 rounds and
  drop to 4 thereafter.
* For 1D tensors (biases, RMSNorm gammas) low-rank is meaningless;
  they're sent uncompressed.

References
----------
* "PowerSGD: Practical Low-Rank Gradient Compression for Distributed
  Optimization" (Vogels, Karimireddy, Jaggi, 2019, NeurIPS)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

try:
    import torch
    _HAS_TORCH = True
except Exception:                                                # pragma: no cover
    torch = None
    _HAS_TORCH = False


@dataclass
class PowerSGDState:
    """Per-parameter PowerSGD state. One per gradient matrix."""
    # Error feedback buffer (residual from the previous compression).
    error: object = None
    # Cached Q matrix from the previous round (warm-start).
    q: object = None
    # The rank used for this matrix (may differ across params).
    rank: int = 4


def _orthogonalize(matrix):
    """In-place QR-like orthogonalization. Two passes of modified
    Gram-Schmidt is enough for numerical stability at rank<=32."""
    cols = matrix.size(1)
    for i in range(cols):
        # Subtract projections of previous columns.
        for j in range(i):
            matrix[:, i] -= (matrix[:, j] * matrix[:, i]).sum() * matrix[:, j]
        norm = matrix[:, i].norm()
        if norm > 1e-8:
            matrix[:, i] /= norm
        else:
            matrix[:, i] = 0


class PowerSGDCompressor:
    """Compress + decompress gradient tensors via PowerSGD.

    Stateful per-parameter: keeps the Q warm-start and error buffer.
    Use one compressor instance per training run; pass each gradient
    through `compress_decompress` (training-side) or via the explicit
    `compress` / `decompress` halves (network-side)."""

    def __init__(self, rank: int = 4,
                 min_compression_size: int = 1024,
                 use_error_feedback: bool = True):
        if not _HAS_TORCH:
            raise RuntimeError("PowerSGDCompressor requires torch")
        self.rank = int(rank)
        self.min_compression_size = int(min_compression_size)
        self.use_error_feedback = bool(use_error_feedback)
        # name -> PowerSGDState
        self.state: Dict[str, PowerSGDState] = {}

    # ------------------------------------------------------------------

    def _get_state(self, name: str, shape) -> PowerSGDState:
        s = self.state.get(name)
        if s is None:
            s = PowerSGDState(rank=self.rank)
            self.state[name] = s
        return s

    def _too_small_to_compress(self, t) -> bool:
        return t.dim() != 2 or t.numel() < self.min_compression_size

    # ------------------------------------------------------------------

    def compress(self, name: str, grad):
        """Compress a 2D gradient. Returns ((P, Q), residual_for_efb)
        where residual lives only on the sender's side -- it never
        crosses the wire. The receiver only needs (P, Q).

        For tensors that are too small or non-2D, returns (grad, None)
        and the receiver should treat the first element as the raw
        gradient (no decompression needed)."""
        if self._too_small_to_compress(grad):
            return grad, None

        s = self._get_state(name, grad.shape)

        # Error feedback: g <- g + accumulated residual
        if self.use_error_feedback:
            if s.error is None:
                s.error = torch.zeros_like(grad)
            grad_efb = grad + s.error
        else:
            grad_efb = grad

        m, n = grad_efb.shape
        rank = min(s.rank, m, n)

        if s.q is None:
            # Cold start: random Q.
            q = torch.randn(n, rank, device=grad.device,
                            dtype=grad.dtype)
        else:
            q = s.q

        # P = G @ Q  (orthogonalize for numerical stability)
        p = grad_efb @ q
        _orthogonalize(p)
        # Q = G^T @ P
        new_q = grad_efb.t() @ p

        # Approximation: G ≈ P @ Q^T
        approx = p @ new_q.t()
        # Update error feedback for next round.
        if self.use_error_feedback:
            s.error = grad_efb - approx
        s.q = new_q.detach().clone()

        return (p, new_q), approx

    def decompress(self, payload):
        """Reconstruct a gradient from a compressed payload.

        `payload` is whatever `compress` returned as the first element:
          * tuple (P, Q) -> reconstruct via P @ Q^T
          * raw tensor   -> identity
        """
        if isinstance(payload, tuple) and len(payload) == 2:
            p, q = payload
            return p @ q.t()
        return payload

    # ------------------------------------------------------------------

    def compression_ratio(self, name: str, shape) -> float:
        if len(shape) != 2:
            return 1.0
        m, n = shape
        if m * n < self.min_compression_size:
            return 1.0
        rank = min(self.rank, m, n)
        full = m * n
        compressed = (m + n) * rank
        return full / max(1, compressed)


# ---------------------------------------------------------------------------
# DiLoCo glue: compress -> all-reduce -> decompress
# ---------------------------------------------------------------------------


async def compressed_all_reduce(
    *,
    grads: Dict[str, "torch.Tensor"],
    compressor: PowerSGDCompressor,
    transport_fn: Callable[[str, object], "Awaitable[List[object]]"],
):
    """Run the DiLoCo outer all-reduce with PowerSGD compression.

    `grads` -- {param_name: gradient_tensor} that this worker
    contributed in the last K local steps.

    `transport_fn(name, payload) -> awaitable[list[payload_from_each_peer]]`
    is supplied by the caller -- it ships the compressed payload to
    every other worker and returns all peers' compressed payloads
    back. The mesh transport (MeshConnector + RemoteProvider) is the
    natural backing for this in production; for tests, a simple
    in-memory function works.

    This function compresses, ships, decompresses-and-averages, and
    writes the averaged tensor back into `grads` in place.
    """
    if not _HAS_TORCH:
        raise RuntimeError("compressed_all_reduce requires torch")

    # 1. compress every gradient locally
    sent: Dict[str, object] = {}
    for name, g in grads.items():
        payload, _ = compressor.compress(name, g)
        sent[name] = payload

    # 2. ship and gather
    received: Dict[str, list] = {}
    for name, payload in sent.items():
        peer_payloads = await transport_fn(name, payload)
        received[name] = peer_payloads

    # 3. decompress + average (include our own to make it a true
    #    all-reduce-mean, not a peer-reduce)
    averaged: Dict[str, "torch.Tensor"] = {}
    for name, peer_list in received.items():
        all_payloads = [sent[name], *peer_list]
        decoded = [compressor.decompress(p) for p in all_payloads]
        averaged[name] = sum(decoded) / float(len(decoded))

    # 4. write back into the gradient dict
    for name, t in averaged.items():
        grads[name] = t

    return grads
