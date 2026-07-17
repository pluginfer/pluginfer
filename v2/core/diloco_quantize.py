"""
DiLoCo Delta Quantization
=========================
Per-tensor symmetric 8-bit quantization for gradient deltas.

Why
---
A 7B-param model has ~28GB of fp32 weights. Sending raw deltas every
DiLoCo round across consumer internet (~100 Mbps) takes 38 minutes.
Symmetric int8 cuts that to 7GB → ~10 minutes. Combined with DiLoCo's
500x reduction in sync frequency, this is what makes consumer-WAN
training actually feasible — proven by INTELLECT-1 (Nov 2024).

Algorithm (per tensor)
----------------------
    scale = max(|x|) / 127.0
    q     = round(x / scale).clamp(-127, 127).to(int8)
    x_hat = q.to(float) * scale

The quantization error is unbiased (E[x_hat] = x for symmetric data)
and bounded by scale, so it composes safely with DiLoCo's outer SGD.

For tensors that are exactly zero (no update), we ship a single byte
of metadata and skip the payload — handles sparse layers cheaply.
"""

from __future__ import annotations

import hashlib
import io
import struct
from typing import Dict, Tuple

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception as _torch_err:                      # pragma: no cover
    torch = None                                     # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = _torch_err


_QUANT_MAGIC = b"PLGQ"
_QUANT_VERSION = 1


def quantize_delta(delta: Dict[str, torch.Tensor]) -> bytes:
    """Serialize a state-dict-shaped delta as int8 + per-tensor fp32 scales."""
    buf = io.BytesIO()
    buf.write(_QUANT_MAGIC)
    buf.write(struct.pack("<I", _QUANT_VERSION))
    buf.write(struct.pack("<I", len(delta)))

    for name, tensor in delta.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Delta entry '{name}' is not a Tensor")
        cpu = tensor.detach().cpu().contiguous().to(torch.float32)
        name_b = name.encode("utf-8")
        if len(name_b) > 0xFFFF:
            raise ValueError("Parameter name too long")
        buf.write(struct.pack("<H", len(name_b)))
        buf.write(name_b)
        buf.write(struct.pack("<B", cpu.dim()))
        for d in cpu.shape:
            buf.write(struct.pack("<I", int(d)))

        max_abs = float(cpu.abs().max().item()) if cpu.numel() > 0 else 0.0
        if max_abs == 0.0:
            buf.write(struct.pack("<f", 0.0))   # scale
            buf.write(struct.pack("<B", 1))     # all_zero flag
            continue
        scale = max_abs / 127.0
        q = torch.round(cpu / scale).clamp(-127, 127).to(torch.int8)
        buf.write(struct.pack("<f", scale))
        buf.write(struct.pack("<B", 0))         # all_zero flag
        buf.write(q.numpy().tobytes())

    body = buf.getvalue()
    digest = hashlib.sha256(body).digest()
    return body + digest


def dequantize_delta(payload: bytes,
                     expected_shapes: Dict[str, Tuple[int, ...]] | None = None,
                     ) -> Dict[str, torch.Tensor]:
    """Inverse of `quantize_delta`. Returns fp32 tensors."""
    if len(payload) < 4 + 4 + 4 + 32:
        raise ValueError("Quantized payload too short")
    body, digest = payload[:-32], payload[-32:]
    if hashlib.sha256(body).digest() != digest:
        raise ValueError("SHA-256 mismatch on quantized delta")

    pos = 0
    if body[pos:pos + 4] != _QUANT_MAGIC:
        raise ValueError("Bad magic on quantized delta")
    pos += 4
    version = struct.unpack_from("<I", body, pos)[0]
    pos += 4
    if version != _QUANT_VERSION:
        raise ValueError(f"Unsupported quant version: {version}")
    n = struct.unpack_from("<I", body, pos)[0]
    pos += 4

    out: Dict[str, torch.Tensor] = {}
    for _ in range(n):
        name_len = struct.unpack_from("<H", body, pos)[0]
        pos += 2
        name = body[pos:pos + name_len].decode("utf-8")
        pos += name_len
        ndim = body[pos]
        pos += 1
        shape = struct.unpack_from(f"<{ndim}I", body, pos)
        pos += ndim * 4
        scale = struct.unpack_from("<f", body, pos)[0]
        pos += 4
        all_zero = body[pos]
        pos += 1

        if expected_shapes and name in expected_shapes and tuple(shape) != tuple(expected_shapes[name]):
            raise ValueError(
                f"Shape mismatch on quantized '{name}': got {tuple(shape)}, "
                f"expected {tuple(expected_shapes[name])}"
            )

        n_elem = 1
        for d in shape:
            n_elem *= int(d)

        if all_zero:
            out[name] = torch.zeros(shape, dtype=torch.float32)
            continue

        if pos + n_elem > len(body):
            raise ValueError(f"Truncated quantized payload for '{name}'")
        raw = body[pos:pos + n_elem]
        pos += n_elem
        q = torch.frombuffer(bytearray(raw), dtype=torch.int8).reshape(shape).clone()
        out[name] = q.to(torch.float32) * scale

    return out


def estimate_compression_ratio(delta: Dict[str, torch.Tensor]) -> float:
    """For logs / dashboards. ~4x for typical fp32 → int8."""
    fp32_bytes = sum(t.numel() * 4 for t in delta.values())
    quant_bytes = len(quantize_delta(delta))
    return fp32_bytes / max(quant_bytes, 1)
