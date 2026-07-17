"""
DiLoCo Safe Serialization
=========================
Serialize / deserialize PyTorch state_dicts WITHOUT pickle.

Why this matters
----------------
torch.save / torch.load use pickle, which is a remote-code-execution
primitive when the file comes from an untrusted source. In a network
where workers send weights to each other, a malicious worker could
ship a poisoned checkpoint that owns every other node on load.

This module ships a pure-bytes, self-describing format:

    [ 4-byte magic 'PLGW' ]
    [ 4-byte version       (uint32 LE) ]
    [ 4-byte num_tensors   (uint32 LE) ]
    For each tensor:
      [ 2-byte name_len    (uint16 LE) ]
      [ name (utf-8 bytes) ]
      [ 1-byte dtype_code ]
      [ 1-byte ndim ]
      [ ndim × 4-byte shape (uint32 LE) ]
      [ tensor bytes (little-endian, contiguous) ]
    [ 32-byte SHA-256 of everything above ]

Reads validate magic, version, the SHA-256 digest, and the dtype/shape
of each tensor before allocating memory. Any mismatch raises before
control flows back to model loading.

Supported dtypes: float32, float16, bfloat16, int64, int32, int8, uint8.
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

MAGIC = b"PLGW"
FORMAT_VERSION = 1

# dtype_code <-> torch dtype tables (only populated when torch is available;
# functions below check `_TORCH_AVAILABLE` and raise a clean error otherwise).
if _TORCH_AVAILABLE:
    _DTYPE_CODES = {
        torch.float32: 1,
        torch.float16: 2,
        torch.bfloat16: 3,
        torch.int64: 4,
        torch.int32: 5,
        torch.int8: 6,
        torch.uint8: 7,
    }
    _CODE_TO_DTYPE = {v: k for k, v in _DTYPE_CODES.items()}
    _DTYPE_BYTES = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int64: 8,
        torch.int32: 4,
        torch.int8: 1,
        torch.uint8: 1,
    }
else:
    _DTYPE_CODES = {}
    _CODE_TO_DTYPE = {}
    _DTYPE_BYTES = {}


class DeserializationError(ValueError):
    """Raised when a payload fails any validation step."""


def serialize_state_dict(state_dict: Dict[str, torch.Tensor]) -> bytes:
    """Encode a state_dict to a self-describing, hash-protected byte string."""
    buf = io.BytesIO()
    buf.write(MAGIC)
    buf.write(struct.pack("<I", FORMAT_VERSION))
    buf.write(struct.pack("<I", len(state_dict)))

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Entry '{name}' is not a Tensor")
        if tensor.dtype not in _DTYPE_CODES:
            raise TypeError(f"Unsupported dtype for '{name}': {tensor.dtype}")
        # Move to CPU and make contiguous; bytes are little-endian on x86/ARM
        cpu_tensor = tensor.detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        if len(name_bytes) > 0xFFFF:
            raise ValueError(f"Parameter name too long (>{0xFFFF} bytes): {name[:40]}...")
        buf.write(struct.pack("<H", len(name_bytes)))
        buf.write(name_bytes)
        buf.write(struct.pack("<B", _DTYPE_CODES[cpu_tensor.dtype]))
        buf.write(struct.pack("<B", cpu_tensor.dim()))
        for dim in cpu_tensor.shape:
            buf.write(struct.pack("<I", int(dim)))
        # numpy view to get bytes; for bfloat16, route via int16 view
        if cpu_tensor.dtype == torch.bfloat16:
            buf.write(cpu_tensor.view(torch.int16).numpy().tobytes())
        else:
            buf.write(cpu_tensor.numpy().tobytes())

    body = buf.getvalue()
    digest = hashlib.sha256(body).digest()
    return body + digest


def deserialize_state_dict(payload: bytes,
                           expected_keys: Tuple[str, ...] | None = None,
                           expected_shapes: Dict[str, Tuple[int, ...]] | None = None,
                           ) -> Dict[str, torch.Tensor]:
    """
    Decode a payload produced by `serialize_state_dict`.

    Optional `expected_keys` and `expected_shapes` perform pre-allocation
    schema validation — catches a malicious or buggy peer before tensor
    memory is allocated.
    """
    if len(payload) < 4 + 4 + 4 + 32:
        raise DeserializationError("Payload too short")

    body = payload[:-32]
    declared_digest = payload[-32:]
    actual_digest = hashlib.sha256(body).digest()
    if declared_digest != actual_digest:
        raise DeserializationError("SHA-256 mismatch — payload corrupted or tampered")

    pos = 0
    if body[pos:pos + 4] != MAGIC:
        raise DeserializationError("Bad magic — not a Pluginfer weight blob")
    pos += 4
    version = struct.unpack_from("<I", body, pos)[0]
    pos += 4
    if version != FORMAT_VERSION:
        raise DeserializationError(f"Unsupported format version: {version}")
    num_tensors = struct.unpack_from("<I", body, pos)[0]
    pos += 4

    state: Dict[str, torch.Tensor] = {}
    for _ in range(num_tensors):
        name_len = struct.unpack_from("<H", body, pos)[0]
        pos += 2
        name = body[pos:pos + name_len].decode("utf-8")
        pos += name_len
        dtype_code = body[pos]
        pos += 1
        ndim = body[pos]
        pos += 1
        shape: Tuple[int, ...] = struct.unpack_from(f"<{ndim}I", body, pos)
        pos += ndim * 4
        if dtype_code not in _CODE_TO_DTYPE:
            raise DeserializationError(f"Unknown dtype_code: {dtype_code}")
        dtype = _CODE_TO_DTYPE[dtype_code]
        elem_bytes = _DTYPE_BYTES[dtype]
        n_elem = 1
        for d in shape:
            n_elem *= int(d)
        n_bytes = n_elem * elem_bytes
        if pos + n_bytes > len(body):
            raise DeserializationError(f"Truncated tensor payload for '{name}'")

        if expected_shapes and name in expected_shapes:
            if tuple(shape) != tuple(expected_shapes[name]):
                raise DeserializationError(
                    f"Shape mismatch for '{name}': got {tuple(shape)}, "
                    f"expected {tuple(expected_shapes[name])}"
                )

        raw = body[pos:pos + n_bytes]
        pos += n_bytes

        if dtype == torch.bfloat16:
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.int16).view(torch.bfloat16)
        else:
            tensor = torch.frombuffer(bytearray(raw), dtype=dtype)
        tensor = tensor.reshape(shape).clone()  # clone -> own memory
        state[name] = tensor

    if pos != len(body):
        raise DeserializationError(f"Trailing bytes after parse: {len(body) - pos} extra")

    if expected_keys is not None:
        missing = set(expected_keys) - set(state)
        extra = set(state) - set(expected_keys)
        if missing or extra:
            raise DeserializationError(
                f"Schema mismatch — missing={sorted(missing)} extra={sorted(extra)}"
            )

    return state


def state_dict_hash(state_dict: Dict[str, torch.Tensor]) -> str:
    """Stable SHA-256 of a state_dict, for verification logs and audit trails."""
    return hashlib.sha256(serialize_state_dict(state_dict)).hexdigest()
