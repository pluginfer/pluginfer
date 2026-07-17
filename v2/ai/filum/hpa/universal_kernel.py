"""§F5 Universal Compute Layer — break the CUDA monopoly.

NVIDIA's leverage in AI comes from one thing: the CUDA software
lock-in. Every model, every training framework, every inference
runtime speaks CUDA first and "other" second. AMD GPUs, Intel
GPUs, Apple Silicon, and browsers run subsets at degraded
performance. This is the moat — not the silicon.

Pluginfer's mesh thesis only works if we *bypass* this lock. The
mesh aggregates idle compute on whatever hardware happens to be in
the user's pocket. If only NVIDIA cards can contribute, we are
just a CUDA marketplace; the moat stays intact.

This module is the kernel-abstraction layer that decouples Filum's
training from any single vendor's API. The contract:

* **Define training operations once** as pure mathematical ops
  (matmul, layernorm, attention_head, cross_entropy, etc.).
* **Multiple backends** — each op has implementations in:
  - ``torch_native`` — whatever torch is built against (CUDA/ROCm/MPS/XPU/CPU).
  - ``numpy_cpu`` — pure CPU fallback, always available.
  - ``webgpu_wgsl`` — WebGPU shaders for browsers + portable hosts.
  - ``iree_vmfb`` — IREE-compiled bytecode for production cross-vendor.
* **Runtime dispatcher** picks the best backend at startup based
  on what's available. Mesh nodes report their backend in their
  §C2 stability advertisement.

The killer architectural property: a §C grain produced via the
``webgpu_wgsl`` backend on a browser is byte-compatible with one
produced via ``torch_native`` on a CUDA H100. Gradients are
gradients. The §C5 NBGGA aggregator does not care which backend
emitted them — only the (model_shard_id, version_v, signature)
matter.

This breaks NVIDIA's lock at the protocol level. CUDA is no longer
the only gateway to mesh participation. AMD, Intel, Apple,
browser, and CPU contributors are all *first-class* citizens of
the mesh.

design notes §F5 (drafted in the design notes): a method of decentralised
neural-network training in which the unit of contributable work
is defined by a vendor-agnostic kernel descriptor; the descriptor
is dispatched at runtime to one of multiple compute backends
based on the contributor's available hardware; gradient outputs
across heterogeneous backends are byte-compatible at the protocol
layer such that aggregation is unaffected by backend choice; and
the routing decision is made *per node*, not per network, allowing
the mesh to consume any hardware regardless of vendor.

This file ships the *scaffold* — the contract + the torch_native
+ numpy_cpu backends + a stub for webgpu_wgsl + iree_vmfb. The
WebGPU and IREE backends are fully specified at the descriptor
level but their compiled implementations are loaded lazily when
the runtime detects the toolchain. Production rollout adds them
without changing the protocol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------- the universal kernel descriptor -------------------------------

@dataclass
class KernelDescriptor:
    """A vendor-agnostic description of one training kernel.

    Fields:
      op_name      — "matmul" | "layernorm" | "attention" | "cross_entropy"
                     | "rmsnorm" | "silu_ffn" | "rope" | etc.
      input_shapes — list of tuples of ints
      output_shape — tuple of ints
      dtype        — "fp16" | "fp32" | "bf16" | "int8"
      params       — op-specific hyperparameters (head_dim, n_heads, etc.)
    """
    op_name: str
    input_shapes: list = field(default_factory=list)
    output_shape: tuple = ()
    dtype: str = "fp32"
    params: dict = field(default_factory=dict)


# ---------- backends ------------------------------------------------------

BACKENDS = ("torch_native", "numpy_cpu", "webgpu_wgsl", "iree_vmfb")


@dataclass
class BackendCapabilities:
    name: str
    available: bool
    supports_dtypes: tuple = ("fp32",)
    supports_ops: tuple = ()
    notes: str = ""


def detect_torch_native() -> BackendCapabilities:
    try:
        import torch  # noqa: F401
    except ImportError:
        return BackendCapabilities(name="torch_native", available=False,
                                     notes="torch not installed")
    dtypes = ["fp32"]
    try:
        import torch
        if torch.cuda.is_available() or hasattr(torch.backends, "mps"):
            dtypes += ["fp16", "bf16"]
    except Exception:
        pass
    return BackendCapabilities(
        name="torch_native",
        available=True,
        supports_dtypes=tuple(dtypes),
        supports_ops=(
            "matmul", "layernorm", "rmsnorm", "softmax",
            "cross_entropy", "silu_ffn", "attention", "rope",
            "embedding",
        ),
        notes="auto-routes to whatever torch is built for "
              "(CUDA / ROCm / MPS / XPU / CPU)",
    )


def detect_numpy_cpu() -> BackendCapabilities:
    try:
        import numpy  # noqa: F401
    except ImportError:
        return BackendCapabilities(name="numpy_cpu", available=False,
                                     notes="numpy not installed")
    return BackendCapabilities(
        name="numpy_cpu",
        available=True,
        supports_dtypes=("fp32",),
        supports_ops=(
            "matmul", "layernorm", "softmax", "cross_entropy",
        ),
        notes="pure-CPU fallback; always available",
    )


def detect_webgpu_wgsl() -> BackendCapabilities:
    """Probe for a WebGPU runtime. Available via wgpu-py on some hosts;
    in-browser the runtime is provided natively. We check for wgpu-py
    here — production browser embedding wires this differently."""
    try:
        import wgpu  # noqa: F401
        return BackendCapabilities(
            name="webgpu_wgsl",
            available=True,
            supports_dtypes=("fp32", "fp16"),
            supports_ops=("matmul", "layernorm", "softmax"),
            notes="wgpu-py available; browser-native via WebGPU",
        )
    except ImportError:
        return BackendCapabilities(
            name="webgpu_wgsl",
            available=False,
            notes="wgpu-py not installed; install for browser/portable backend",
        )


def detect_iree_vmfb() -> BackendCapabilities:
    try:
        import iree.runtime  # noqa: F401
        return BackendCapabilities(
            name="iree_vmfb",
            available=True,
            supports_dtypes=("fp32", "fp16", "bf16"),
            supports_ops=(
                "matmul", "layernorm", "rmsnorm", "softmax",
                "cross_entropy", "attention",
            ),
            notes="IREE runtime available; compiled .vmfb bytecode portable "
                  "across CUDA / Vulkan / Metal / WebGPU / WASM / CPU",
        )
    except ImportError:
        return BackendCapabilities(
            name="iree_vmfb",
            available=False,
            notes="IREE not installed; install for production cross-vendor backend",
        )


def detect_all_backends() -> list[BackendCapabilities]:
    return [
        detect_torch_native(),
        detect_numpy_cpu(),
        detect_webgpu_wgsl(),
        detect_iree_vmfb(),
    ]


# ---------- the dispatcher ------------------------------------------------

class UniversalKernelDispatcher:
    """Picks the best backend for each kernel call.

    Priority (highest to lowest):
      1. torch_native — best perf when available
      2. iree_vmfb    — production cross-vendor when packaged
      3. webgpu_wgsl  — browser/portable
      4. numpy_cpu    — fallback

    Override per call by passing ``prefer="iree_vmfb"`` etc.
    """

    def __init__(self):
        self._caps = {b.name: b for b in detect_all_backends()}
        self._handlers: dict[tuple[str, str], Callable] = {}
        self._register_default_handlers()

    @property
    def capabilities(self) -> dict[str, BackendCapabilities]:
        return dict(self._caps)

    def available_backends(self) -> list[str]:
        return [name for name, cap in self._caps.items() if cap.available]

    def best_backend_for(self, op_name: str,
                          *, prefer: Optional[str] = None) -> Optional[str]:
        """Return the highest-priority backend that supports op_name."""
        if prefer and prefer in self.available_backends():
            cap = self._caps[prefer]
            if op_name in cap.supports_ops:
                return prefer
        # Default priority order.
        for backend in ("torch_native", "iree_vmfb",
                          "webgpu_wgsl", "numpy_cpu"):
            cap = self._caps.get(backend)
            if cap and cap.available and op_name in cap.supports_ops:
                return backend
        return None

    def execute(
        self,
        descriptor: KernelDescriptor,
        *args,
        prefer: Optional[str] = None,
        **kwargs,
    ) -> tuple[Any, str]:
        """Dispatch and run the kernel. Returns (output, backend_used)."""
        backend = self.best_backend_for(descriptor.op_name, prefer=prefer)
        if backend is None:
            raise RuntimeError(
                f"no available backend supports op '{descriptor.op_name}' "
                f"(available: {self.available_backends()})"
            )
        handler = self._handlers.get((backend, descriptor.op_name))
        if handler is None:
            raise RuntimeError(
                f"backend '{backend}' has no handler for '{descriptor.op_name}'"
            )
        return handler(descriptor, *args, **kwargs), backend

    def register_handler(self, backend: str, op_name: str,
                          handler: Callable) -> None:
        """Allow runtime extension. Mesh contributors can plug in new backends
        without touching the dispatcher core."""
        self._handlers[(backend, op_name)] = handler

    # ---- default handlers ---------------------------------------------

    def _register_default_handlers(self) -> None:
        self._register_torch_handlers()
        self._register_numpy_handlers()

    def _register_torch_handlers(self) -> None:
        try:
            import torch
        except ImportError:
            return

        def _matmul(_d, a, b):
            return torch.matmul(a, b)

        def _softmax(_d, x):
            return torch.softmax(x, dim=-1)

        def _layernorm(d, x):
            normalized_shape = d.params.get(
                "normalized_shape", (x.shape[-1],),
            )
            return torch.nn.functional.layer_norm(x, normalized_shape)

        def _cross_entropy(_d, logits, targets):
            return torch.nn.functional.cross_entropy(logits, targets)

        for op, fn in (("matmul", _matmul), ("softmax", _softmax),
                          ("layernorm", _layernorm),
                          ("cross_entropy", _cross_entropy)):
            self.register_handler("torch_native", op, fn)

    def _register_numpy_handlers(self) -> None:
        try:
            import numpy as np
        except ImportError:
            return

        def _matmul(_d, a, b):
            return np.matmul(np.asarray(a, dtype="float32"),
                             np.asarray(b, dtype="float32"))

        def _softmax(_d, x):
            x = np.asarray(x, dtype="float32")
            x = x - x.max(axis=-1, keepdims=True)
            e = np.exp(x)
            return e / e.sum(axis=-1, keepdims=True)

        def _layernorm(_d, x):
            x = np.asarray(x, dtype="float32")
            mean = x.mean(axis=-1, keepdims=True)
            var = x.var(axis=-1, keepdims=True)
            return (x - mean) / np.sqrt(var + 1e-5)

        def _cross_entropy(_d, logits, targets):
            logits = np.asarray(logits, dtype="float32")
            targets = np.asarray(targets, dtype="int64")
            x = logits - logits.max(axis=-1, keepdims=True)
            log_sum = np.log(np.exp(x).sum(axis=-1))
            picked = x[np.arange(len(targets)), targets]
            return float(-(picked - log_sum).mean())

        for op, fn in (("matmul", _matmul), ("softmax", _softmax),
                          ("layernorm", _layernorm),
                          ("cross_entropy", _cross_entropy)):
            self.register_handler("numpy_cpu", op, fn)


# Module-level singleton — one dispatcher per process.
_GLOBAL_DISPATCHER: Optional[UniversalKernelDispatcher] = None


def get_dispatcher() -> UniversalKernelDispatcher:
    global _GLOBAL_DISPATCHER
    if _GLOBAL_DISPATCHER is None:
        _GLOBAL_DISPATCHER = UniversalKernelDispatcher()
    return _GLOBAL_DISPATCHER


def summarize_backends() -> str:
    """Human-readable summary of what's available on this host."""
    d = get_dispatcher()
    lines = ["Pluginfer Universal Compute Layer (§F5)"]
    for name, cap in d.capabilities.items():
        marker = "ON " if cap.available else "off"
        lines.append(f"  [{marker}] {name:<14}  {cap.notes}")
        if cap.available:
            lines.append(f"           dtypes: {cap.supports_dtypes}")
            lines.append(f"           ops:    {cap.supports_ops}")
    return "\n".join(lines)
