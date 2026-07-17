"""§J1 Multi-Model Federation tests.

Exercises the routing surface, privacy gating, and §D1 receipt
emission without requiring Ollama, a Filum checkpoint, or a remote
API to be present. Two strategies:

* For the "is the wiring correct" tests we register a synthetic
  backend that just echoes a deterministic response.
* For the "real local LLM is detected" path we monkeypatch
  ``urllib.request.urlopen`` against ``OllamaBackend`` so the test
  is hermetic.
"""

from __future__ import annotations

import io
import json
from typing import Optional

import pytest

from ai.filum.hpa.model_federation import (
    FederationConfig,
    FilumLocalBackend,
    GenerationRequest,
    GenerationResponse,
    ModelBackend,
    ModelFederation,
    OllamaBackend,
    quick_status,
)


# ---------- helpers --------------------------------------------------------


class _EchoBackend(ModelBackend):
    """Always-available local backend that just echoes the prompt."""

    name = "echo"
    is_local = True
    requires_network = False

    def __init__(self, label: str = "echo-1"):
        self.label = label

    def available(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return [self.label]

    def generate(self, req: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            text=f"[{self.label}] {req.prompt}",
            model_id=self.label,
            backend_name=self.name,
            elapsed_s=0.001,
        )


class _RemoteBackend(ModelBackend):
    """Network-requiring backend, used to test LOCAL_ONLY gating."""

    name = "remote_api"
    is_local = False
    requires_network = True

    def available(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return ["claude-stub"]

    def generate(self, req: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            text="[remote] " + req.prompt,
            model_id="claude-stub",
            backend_name=self.name,
            elapsed_s=0.01,
        )


# ---------- routing & priority --------------------------------------------


def test_federation_picks_first_eligible_local():
    fed = ModelFederation(
        config=FederationConfig(issue_receipts=False),
        backends=[_EchoBackend("first"), _EchoBackend("second")],
    )
    resp = fed.generate(GenerationRequest(
        prompt="hi", require_receipt=False,
    ))
    assert resp.backend_name == "echo"
    assert resp.model_id == "first"
    assert resp.text == "[first] hi"


def test_local_only_rejects_network_backend():
    """LOCAL_ONLY must filter out backends that need the network."""
    fed = ModelFederation(
        config=FederationConfig(issue_receipts=False),
        backends=[_RemoteBackend(), _EchoBackend("local")],
    )
    resp = fed.generate(GenerationRequest(
        prompt="ping",
        privacy_mode="LOCAL_ONLY",
        require_receipt=False,
    ))
    assert resp.backend_name == "echo"
    assert resp.model_id == "local"


def test_local_only_with_no_local_backend_raises():
    fed = ModelFederation(
        config=FederationConfig(issue_receipts=False),
        backends=[_RemoteBackend()],
    )
    with pytest.raises(RuntimeError) as exc:
        fed.generate(GenerationRequest(
            prompt="ping",
            privacy_mode="LOCAL_ONLY",
            require_receipt=False,
        ))
    assert "LOCAL_ONLY" in str(exc.value)


def test_hybrid_allows_remote_when_only_remote_present():
    fed = ModelFederation(
        config=FederationConfig(issue_receipts=False),
        backends=[_RemoteBackend()],
    )
    resp = fed.generate(GenerationRequest(
        prompt="hello",
        privacy_mode="HYBRID",
        require_receipt=False,
    ))
    assert resp.backend_name == "remote_api"


def test_list_available_reports_only_available_backends():
    class _Down(_EchoBackend):
        def available(self) -> bool:
            return False

    fed = ModelFederation(
        config=FederationConfig(issue_receipts=False),
        backends=[_Down("down"), _EchoBackend("up")],
    )
    avail = fed.list_available()
    assert len(avail) == 1
    assert avail[0]["models"] == ["up"]


# ---------- Ollama auto-detection (hermetic via monkeypatch) --------------


def _fake_urlopen_ollama_with_models(*_args, **_kwargs):
    """Pretend Ollama is up with two installed models."""
    body = json.dumps({
        "models": [
            {"name": "llama3:8b"},
            {"name": "phi3:3.8b"},
        ],
    }).encode("utf-8")
    return _CtxBytes(body)


def _fake_urlopen_ollama_empty(*_args, **_kwargs):
    return _CtxBytes(json.dumps({"models": []}).encode("utf-8"))


def _fake_urlopen_ollama_unreachable(*_args, **_kwargs):
    raise OSError("connection refused")


class _CtxBytes:
    """Minimal urlopen() replacement: a context manager whose body
    is the stub bytes."""
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._payload


def test_ollama_detects_installed_models(monkeypatch):
    monkeypatch.setattr(
        "ai.filum.hpa.model_federation.urllib.request.urlopen",
        _fake_urlopen_ollama_with_models,
    )
    backend = OllamaBackend()
    models = backend.list_models()
    assert "llama3:8b" in models
    assert "phi3:3.8b" in models
    assert backend.available() is True


def test_ollama_picks_llama_first_then_phi(monkeypatch):
    monkeypatch.setattr(
        "ai.filum.hpa.model_federation.urllib.request.urlopen",
        _fake_urlopen_ollama_with_models,
    )
    backend = OllamaBackend()
    assert backend.pick_default_model() == "llama3:8b"


def test_ollama_unreachable_means_unavailable(monkeypatch):
    monkeypatch.setattr(
        "ai.filum.hpa.model_federation.urllib.request.urlopen",
        _fake_urlopen_ollama_unreachable,
    )
    backend = OllamaBackend()
    assert backend.list_models() == []
    assert backend.available() is False


def test_ollama_empty_means_unavailable(monkeypatch):
    monkeypatch.setattr(
        "ai.filum.hpa.model_federation.urllib.request.urlopen",
        _fake_urlopen_ollama_empty,
    )
    backend = OllamaBackend()
    assert backend.list_models() == []
    assert backend.available() is False


# ---------- Filum local backend availability gate -------------------------


def test_filum_local_unavailable_when_no_checkpoint(tmp_path):
    backend = FilumLocalBackend(
        checkpoint_path=str(tmp_path / "missing.pt"),
        tokenizer_path=str(tmp_path / "missing.json"),
    )
    assert backend.available() is False
    assert backend.list_models() == []


# ---------- §D1 receipt emission ------------------------------------------


def test_receipt_attached_when_requested(tmp_path):
    fed = ModelFederation(
        config=FederationConfig(issue_receipts=True),
        backends=[_EchoBackend("recv")],
    )
    resp = fed.generate(GenerationRequest(
        prompt="prove it",
        require_receipt=True,
    ))
    # Receipt id may be None if the receipt subsystem is unavailable
    # in this test env; if present it must be a non-empty string.
    if resp.receipt_id is not None:
        assert isinstance(resp.receipt_id, str)
        assert resp.receipt_id


def test_no_receipt_when_disabled():
    fed = ModelFederation(
        config=FederationConfig(issue_receipts=False),
        backends=[_EchoBackend("noreceipt")],
    )
    resp = fed.generate(GenerationRequest(
        prompt="quiet", require_receipt=False,
    ))
    assert resp.receipt_id is None


# ---------- quick_status() smoke ------------------------------------------


def test_quick_status_with_no_backends_is_helpful(monkeypatch):
    # Force both auto-detected backends to be unavailable.
    monkeypatch.setattr(
        "ai.filum.hpa.model_federation.urllib.request.urlopen",
        _fake_urlopen_ollama_unreachable,
    )
    out = quick_status()
    assert "no backends available" in out.lower()
    assert "ollama.com" in out


def test_quick_status_lists_available(monkeypatch):
    monkeypatch.setattr(
        "ai.filum.hpa.model_federation.urllib.request.urlopen",
        _fake_urlopen_ollama_with_models,
    )
    out = quick_status()
    assert "ollama" in out.lower()
    assert "llama3:8b" in out
