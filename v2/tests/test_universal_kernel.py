"""§F5 Universal Compute Layer tests."""

from __future__ import annotations

import numpy as np
import pytest


def test_detect_all_backends_returns_list():
    from ai.filum.hpa.universal_kernel import detect_all_backends, BACKENDS
    caps = detect_all_backends()
    assert len(caps) == len(BACKENDS)
    names = {c.name for c in caps}
    assert names == set(BACKENDS)


def test_dispatcher_finds_at_least_one_backend():
    """numpy_cpu is always-available, so this never returns no backends."""
    from ai.filum.hpa.universal_kernel import get_dispatcher
    d = get_dispatcher()
    available = d.available_backends()
    assert "numpy_cpu" in available


def test_best_backend_prefers_torch_when_available():
    from ai.filum.hpa.universal_kernel import get_dispatcher
    d = get_dispatcher()
    avail = d.available_backends()
    best = d.best_backend_for("matmul")
    if "torch_native" in avail:
        assert best == "torch_native"
    else:
        assert best in ("iree_vmfb", "webgpu_wgsl", "numpy_cpu")


def test_dispatch_matmul_numpy_backend():
    from ai.filum.hpa.universal_kernel import (
        UniversalKernelDispatcher, KernelDescriptor,
    )
    d = UniversalKernelDispatcher()
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    descriptor = KernelDescriptor(op_name="matmul")
    out, backend_used = d.execute(descriptor, a, b, prefer="numpy_cpu")
    assert backend_used == "numpy_cpu"
    expected = np.matmul(a, b)
    assert np.allclose(out, expected)


def test_dispatch_softmax_numpy_backend():
    from ai.filum.hpa.universal_kernel import (
        UniversalKernelDispatcher, KernelDescriptor,
    )
    d = UniversalKernelDispatcher()
    x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    descriptor = KernelDescriptor(op_name="softmax")
    out, _ = d.execute(descriptor, x, prefer="numpy_cpu")
    # Each row sums to ~1.
    sums = out.sum(axis=-1)
    assert np.allclose(sums, [1.0, 1.0])


def test_dispatch_unknown_op_raises():
    from ai.filum.hpa.universal_kernel import (
        UniversalKernelDispatcher, KernelDescriptor,
    )
    d = UniversalKernelDispatcher()
    descriptor = KernelDescriptor(op_name="totally-fake-op-xyz")
    with pytest.raises(RuntimeError):
        d.execute(descriptor, np.zeros(4))


def test_dispatch_torch_when_available():
    """If torch is installed, matmul through torch_native must work."""
    pytest.importorskip("torch")
    import torch
    from ai.filum.hpa.universal_kernel import (
        UniversalKernelDispatcher, KernelDescriptor,
    )
    d = UniversalKernelDispatcher()
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    descriptor = KernelDescriptor(op_name="matmul")
    out, backend_used = d.execute(descriptor, a, b, prefer="torch_native")
    assert backend_used == "torch_native"
    expected = torch.matmul(a, b)
    assert torch.allclose(out, expected)


def test_register_custom_handler():
    from ai.filum.hpa.universal_kernel import (
        UniversalKernelDispatcher, KernelDescriptor,
    )
    d = UniversalKernelDispatcher()
    called: list = []

    def my_handler(descriptor, x):
        called.append(descriptor.op_name)
        return x * 2

    d.register_handler("numpy_cpu", "double", my_handler)
    descriptor = KernelDescriptor(op_name="double")
    # Update the internal capabilities so the dispatcher will route here.
    cur = d._caps["numpy_cpu"]
    d._caps["numpy_cpu"] = type(cur)(
        name="numpy_cpu", available=True,
        supports_dtypes=("fp32",),
        supports_ops=(*cur.supports_ops, "double"),
    )
    out, backend_used = d.execute(descriptor, 5.0, prefer="numpy_cpu")
    assert backend_used == "numpy_cpu"
    assert out == 10.0
    assert called == ["double"]


def test_summarize_backends_mentions_universal_layer():
    from ai.filum.hpa.universal_kernel import summarize_backends
    s = summarize_backends()
    assert "Universal Compute Layer" in s
    assert "numpy_cpu" in s
