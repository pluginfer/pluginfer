"""CP-1 / Task 1.3: full task-dispatch wire test.

Exercises auction layer -> MeshGPUProvider.execute -> result-hashing +
wallet signing. Also covers requester-signature validation and the
explicit ProviderConfigurationError path (no silent stubs allowed).

A direct mesh round-trip would need a running CompleteMeshController;
we use the `local_executor` injection so the test stays in-process and
exercises the full code path end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import base64  # noqa: E402
import hashlib  # noqa: E402

import pytest  # noqa: E402

from core.providers import (  # noqa: E402
    Bid,
    JobSpec,
    MeshGPUProvider,
    ProviderConfigurationError,
    _CloudHttpError,
    _CloudLLMProvider,
)
from core.tokenomics import Wallet  # noqa: E402


def _make_job(payload: dict | None = None, **kw) -> JobSpec:
    return JobSpec(
        job_id="job_test_1",
        kind="inference",
        payload=payload or {"prompt": "hello"},
        cost_ceiling_usd=kw.get("cost_ceiling_usd", 1.0),
        latency_ceiling_ms=kw.get("latency_ceiling_ms", 5000),
    )


def _make_bid(provider_id: str = "peer_1", price: float = 0.001) -> Bid:
    return Bid(
        provider_id=provider_id,
        price_usd=price,
        eta_ms=1000,
        expected_quality=0.85,
        privacy_grade="private",
    )


# ---------------------------------------------------------------------------
# Happy path: local_executor + wallet
# ---------------------------------------------------------------------------

def test_mesh_provider_executes_locally_and_signs_result() -> None:
    wallet = Wallet()
    fixed_output = b"the actual result bytes"

    def executor(payload):
        assert payload["prompt"] == "hello"
        return fixed_output

    provider = MeshGPUProvider(
        provider_id="peer_local_1",
        wallet=wallet,
        local_executor=executor,
    )
    bid = _make_bid()
    out = provider.execute(_make_job(), bid)
    assert out["status"] == "executed"
    assert out["job_id"] == "job_test_1"
    assert out["provider_id"] == "peer_local_1"
    # Hash is sha256 of raw bytes
    assert out["result_hash"] == hashlib.sha256(fixed_output).hexdigest()
    # b64 round-trips
    assert base64.b64decode(out["result_bytes"]) == fixed_output
    # Provider signature verifies under the wallet pubkey
    assert Wallet.verify(
        out["provider_pubkey_pem"], out["result_hash"], out["provider_sig"]
    )
    assert out["execution_ms"] >= 0


# ---------------------------------------------------------------------------
# Configuration error: no executor AND no task_router
# ---------------------------------------------------------------------------

def test_mesh_provider_raises_when_unconfigured() -> None:
    wallet = Wallet()
    provider = MeshGPUProvider(provider_id="peer_unconfigured", wallet=wallet)
    with pytest.raises(ProviderConfigurationError):
        provider.execute(_make_job(), _make_bid())


def test_mesh_provider_raises_when_no_wallet() -> None:
    provider = MeshGPUProvider(
        provider_id="peer_no_wallet",
        local_executor=lambda _p: b"x",
    )
    with pytest.raises(ProviderConfigurationError):
        provider.execute(_make_job(), _make_bid())


# ---------------------------------------------------------------------------
# Local executor errors -> structured timeout / error response
# ---------------------------------------------------------------------------

def test_local_executor_exception_returns_error_dict() -> None:
    wallet = Wallet()

    def boom(_payload):
        raise RuntimeError("disk full")

    provider = MeshGPUProvider(
        provider_id="peer_err",
        wallet=wallet,
        local_executor=boom,
    )
    out = provider.execute(_make_job(), _make_bid())
    assert out["status"] == "error"
    assert out["code"] == "execution_error"
    assert out["refund_eligible"] is True
    assert "disk full" in out["reason"]


def test_local_executor_wrong_return_type_returns_error_dict() -> None:
    wallet = Wallet()
    provider = MeshGPUProvider(
        provider_id="peer_bad_return",
        wallet=wallet,
        local_executor=lambda _p: "string not bytes",
    )
    out = provider.execute(_make_job(), _make_bid())
    assert out["status"] == "error"
    assert "must return bytes" in out["reason"]


# ---------------------------------------------------------------------------
# Mesh path via TaskRouter timeout (synthetic)
# ---------------------------------------------------------------------------

class _FakeTaskRouter:
    """Simulates submit_and_wait with configurable behaviour."""

    def __init__(self, behaviour: str) -> None:
        self.behaviour = behaviour

    def submit_and_wait(self, requirements, input_data, timeout_s):
        if self.behaviour == "timeout":
            return None
        if self.behaviour == "error":
            return {"status": "error", "reason": "no peer accepted"}
        if self.behaviour == "ok":
            return {"output": "mesh-side computed"}
        raise AssertionError("unknown behaviour")


def test_mesh_path_timeout_returns_refund_eligible() -> None:
    wallet = Wallet()
    provider = MeshGPUProvider(
        provider_id="peer_mesh_to",
        wallet=wallet,
        task_router=_FakeTaskRouter("timeout"),
    )
    out = provider.execute(_make_job(), _make_bid())
    assert out["status"] == "timeout"
    assert out["refund_eligible"] is True
    assert out["deadline_ms"] >= 1000


def test_mesh_path_error_returns_refund_eligible() -> None:
    wallet = Wallet()
    provider = MeshGPUProvider(
        provider_id="peer_mesh_err",
        wallet=wallet,
        task_router=_FakeTaskRouter("error"),
    )
    out = provider.execute(_make_job(), _make_bid())
    assert out["status"] == "error"
    assert out["code"] == "execution_error"
    assert "no peer accepted" in out["reason"]


def test_mesh_path_ok_signs_result() -> None:
    wallet = Wallet()
    provider = MeshGPUProvider(
        provider_id="peer_mesh_ok",
        wallet=wallet,
        task_router=_FakeTaskRouter("ok"),
    )
    out = provider.execute(_make_job(), _make_bid())
    assert out["status"] == "executed"
    assert Wallet.verify(
        out["provider_pubkey_pem"], out["result_hash"], out["provider_sig"]
    )


# ---------------------------------------------------------------------------
# Requester-signature gate
# ---------------------------------------------------------------------------

def test_requester_sig_gate_rejects_missing_sig() -> None:
    wallet = Wallet()
    provider = MeshGPUProvider(
        provider_id="peer_sig",
        wallet=wallet,
        local_executor=lambda _p: b"x",
        require_signed_requests=True,
    )
    out = provider.execute(_make_job(), _make_bid())
    assert out["status"] == "error"
    assert out["code"] == "requester_sig_invalid"
    assert "no requester_pubkey_pem" in out["reason"]


def test_requester_sig_gate_rejects_bad_sig() -> None:
    wallet = Wallet()
    requester = Wallet()
    provider = MeshGPUProvider(
        provider_id="peer_sig_bad",
        wallet=wallet,
        local_executor=lambda _p: b"x",
        require_signed_requests=True,
    )
    job = _make_job()
    job.requester_pubkey_pem = requester.public_key_pem
    job.request_signature = requester.sign("not the right message")
    out = provider.execute(job, _make_bid())
    assert out["status"] == "error"
    assert out["code"] == "requester_sig_invalid"


def test_requester_sig_gate_accepts_valid_sig() -> None:
    wallet = Wallet()
    requester = Wallet()
    provider = MeshGPUProvider(
        provider_id="peer_sig_ok",
        wallet=wallet,
        local_executor=lambda _p: b"signed-output",
        require_signed_requests=True,
    )
    job = _make_job()
    job.requester_pubkey_pem = requester.public_key_pem
    job.request_signature = requester.sign(job.signing_message())
    out = provider.execute(job, _make_bid())
    assert out["status"] == "executed"
    assert out["result_hash"] == hashlib.sha256(b"signed-output").hexdigest()


# ---------------------------------------------------------------------------
# Cloud LLM provider: schema dispatch, fail-closed paths
# ---------------------------------------------------------------------------

def test_cloud_provider_returns_error_with_no_api_key() -> None:
    """No keychain entry -> structured error, no silent fake response."""
    p = _CloudLLMProvider(
        provider_id="openai",
        keychain_service="pluginfer-openai-test-no-key",
        keychain_user="default",
        base_price_per_1k_tok_usd=0.005,
        base_eta_ms=1000,
        base_quality=0.9,
        enabled=True,
    )
    out = p.execute(_make_job({"prompt": "hello"}), _make_bid())
    assert out["status"] == "error"
    assert out["code"] == "no_api_key"
    assert out["refund_eligible"] is True


def test_cloud_provider_unknown_schema_raises_cloud_http_error() -> None:
    """An unknown provider_id should surface as a clean error path."""
    p = _CloudLLMProvider(
        provider_id="some-future-vendor",
        keychain_service="pluginfer-some-future-vendor-test",
        keychain_user="default",
        base_price_per_1k_tok_usd=0.005,
        base_eta_ms=1000,
        base_quality=0.9,
        enabled=True,
    )
    # Bypass the no-key guard so we reach _dispatch_upstream
    with pytest.raises(_CloudHttpError, match="unknown_provider_schema"):
        p._dispatch_upstream("hi", 8, 1.0)
