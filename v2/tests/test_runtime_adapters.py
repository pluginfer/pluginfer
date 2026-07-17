"""Runtime adapter contract tests.

The adapters wrap third-party LLM runtimes. We test the SHAPE of the
contract (probe path, autodetect order, RuntimeAdapterUnavailable
on missing deps) — NOT that real models work on this CI machine,
because Ollama/llama-cpp/transformers aren't necessarily installed.

What we pin:
  * `register_adapter` populates the registry at import time.
  * `list_available_adapters` returns the subset whose probe
    succeeds — empty list is allowed on a bare CI box.
  * `autodetect_runner` raises `RuntimeAdapterUnavailable` cleanly
    when nothing is reachable.
  * Each adapter raises `RuntimeAdapterUnavailable` (not a different
    exception type) when its dependencies aren't met — the
    refuse-rather-than-lie discipline.
"""

from __future__ import annotations

import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

from core.runtime_adapters.base import (  # noqa: E402
    RunnerFn,
    RuntimeAdapterUnavailable,
    autodetect_runner,
    list_available_adapters,
    register_adapter,
)
from core.runtime_adapters import (  # noqa: E402
    llama_cpp_adapter, ollama_adapter, transformers_adapter,
)


def test_registry_lists_at_least_three_adapters():
    """Importing the package registers all three built-in adapters."""
    # The probe path may filter to fewer; we check the unfiltered
    # list is populated by inspecting the private registry.
    from core.runtime_adapters.base import _REGISTRY
    names = [n for n, _ in _REGISTRY]
    assert "ollama" in names
    assert "llama-cpp" in names
    assert "transformers" in names


def test_list_available_adapters_handles_missing_deps_cleanly():
    """The list MUST not raise — adapters whose deps aren't installed
    just don't appear. On THIS dev machine that's any subset."""
    available = list_available_adapters()
    assert isinstance(available, list)
    for name in available:
        assert isinstance(name, str)


def test_autodetect_raises_runtime_adapter_unavailable_when_all_fail():
    """If every adapter has been monkey-patched to fail, the
    autodetect raises the canonical exception type (not a generic
    RuntimeError or AssertionError)."""
    import core.runtime_adapters.base as base_mod
    saved = list(base_mod._REGISTRY)
    base_mod._REGISTRY.clear()

    def _fail(*, model_id, _probe=False, **_kw):
        raise RuntimeAdapterUnavailable("test stub")
    base_mod._REGISTRY.append(("stub", _fail))

    try:
        with pytest.raises(RuntimeAdapterUnavailable):
            autodetect_runner(model_id="anything")
    finally:
        base_mod._REGISTRY[:] = saved


def test_autodetect_picks_first_available():
    import core.runtime_adapters.base as base_mod
    saved = list(base_mod._REGISTRY)
    base_mod._REGISTRY.clear()

    def _fail_first(*, model_id, _probe=False, **_kw):
        raise RuntimeAdapterUnavailable("nope")

    def _ok_second(*, model_id, _probe=False, **_kw):
        return lambda prompt, payload: b"second-adapter-output"

    base_mod._REGISTRY.append(("first", _fail_first))
    base_mod._REGISTRY.append(("second", _ok_second))
    try:
        runner = autodetect_runner(model_id="x")
        assert callable(runner)
        assert runner("hello", {}) == b"second-adapter-output"
    finally:
        base_mod._REGISTRY[:] = saved


def test_prefer_kw_reorders_trial_list():
    import core.runtime_adapters.base as base_mod
    saved = list(base_mod._REGISTRY)
    base_mod._REGISTRY.clear()

    def _a(*, model_id, _probe=False, **_kw):
        return lambda prompt, payload: b"a-output"

    def _b(*, model_id, _probe=False, **_kw):
        return lambda prompt, payload: b"b-output"

    base_mod._REGISTRY.append(("a", _a))
    base_mod._REGISTRY.append(("b", _b))
    try:
        # No preference: a wins.
        assert autodetect_runner(model_id="x")("p", {}) == b"a-output"
        # Prefer b: b wins.
        assert autodetect_runner(
            model_id="x", prefer=["b"],
        )("p", {}) == b"b-output"
    finally:
        base_mod._REGISTRY[:] = saved


def test_ollama_adapter_raises_unavailable_when_not_running():
    """When Ollama isn't running on the default host, the probe
    raises RuntimeAdapterUnavailable (not a different exception type)
    so autodetect can fall through cleanly."""
    import os
    saved = os.environ.get("OLLAMA_HOST")
    # Point at a port that can't have anything listening.
    os.environ["OLLAMA_HOST"] = "http://127.0.0.1:1"
    try:
        with pytest.raises(RuntimeAdapterUnavailable):
            ollama_adapter.make_ollama_runner(
                model_id="probe", _probe=True,
                host="http://127.0.0.1:1",
            )
    finally:
        if saved is None:
            os.environ.pop("OLLAMA_HOST", None)
        else:
            os.environ["OLLAMA_HOST"] = saved


def test_llama_cpp_adapter_raises_unavailable_without_models_dir(tmp_path, monkeypatch):
    """No .gguf in models dir -> probe raises RuntimeAdapterUnavailable."""
    monkeypatch.setenv("PLUGINFER_GGUF_DIR", str(tmp_path))
    # Re-import to pick up the env var.
    import importlib
    import core.runtime_adapters.llama_cpp_adapter as mod
    importlib.reload(mod)
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        # If llama_cpp itself isn't installed, the probe raises
        # at the import line — also RuntimeAdapterUnavailable.
        with pytest.raises(RuntimeAdapterUnavailable):
            mod.make_llama_cpp_runner(model_id="x", _probe=True)
        return
    with pytest.raises(RuntimeAdapterUnavailable):
        mod.make_llama_cpp_runner(model_id="x", _probe=True)


def test_runner_signature_returns_bytes():
    """A custom adapter returns bytes from runner — the flagship
    pipeline depends on this exact shape (it base64-encodes + hashes
    for the PNIS receipt)."""
    def factory(*, model_id, _probe=False, **_kw):
        if _probe:
            return lambda prompt, payload: b""
        return lambda prompt, payload: f"reply-to:{prompt}".encode("utf-8")

    runner = factory(model_id="x")
    out = runner("hello", {"max_tokens": 16})
    assert isinstance(out, bytes)
    assert out == b"reply-to:hello"
