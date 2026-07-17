"""Tests for A4: fine_tune SDK."""

import sys
from pathlib import Path

import pytest

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))
SDK_PATH = V2 / "sdk" / "python"
if str(SDK_PATH) not in sys.path:
    sys.path.insert(0, str(SDK_PATH))

from pluginfer.fine_tune import (  # noqa: E402
    FineTuneError,
    FineTuneSpec,
    fine_tune,
    fine_tune_blocking,
)


# ---------------------------------------------------------------------------
# Stub client for testing without httpx / live API
# ---------------------------------------------------------------------------


class _StubJobs:
    def __init__(self, *, submit_response=None, get_responses=None):
        self._submit_response = submit_response or {
            "job_id": "ft-job-1", "state": "submitted",
        }
        self._get_responses = list(get_responses or [
            {"job_id": "ft-job-1", "state": "running", "checkpoints": []},
            {"job_id": "ft-job-1", "state": "completed",
             "checkpoints": [{"seq": 0, "uri": "ipfs://abc",
                              "sha256": "00", "produced_at_ns": 1}],
             "final_model_uri": "ipfs://final-checkpoint"},
        ])
        self.submitted_kwargs = None
        self._calls = 0

    def submit(self, **kwargs):
        self.submitted_kwargs = kwargs
        return self._submit_response

    def get(self, job_id):
        if self._calls >= len(self._get_responses):
            return self._get_responses[-1]
        r = self._get_responses[self._calls]
        self._calls += 1
        return r


class _StubClient:
    def __init__(self, jobs=None):
        self.jobs = jobs or _StubJobs()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_zero_epochs_rejected():
    with pytest.raises(FineTuneError, match="epochs"):
        fine_tune(client=_StubClient(), model="m",
                  dataset_uri="hf://d", epochs=0)


def test_excessive_peer_count_rejected():
    with pytest.raises(FineTuneError, match="peers="):
        fine_tune(client=_StubClient(), model="m",
                  dataset_uri="hf://d", peers=999)


def test_invalid_privacy_class_rejected():
    with pytest.raises(FineTuneError, match="privacy_class"):
        fine_tune(client=_StubClient(), model="m",
                  dataset_uri="hf://d", privacy_class="WHATEVER")


def test_zero_cost_ceiling_rejected():
    with pytest.raises(FineTuneError, match="cost_ceiling_usd"):
        fine_tune(client=_StubClient(), model="m",
                  dataset_uri="hf://d", cost_ceiling_usd=0)


def test_empty_model_rejected():
    with pytest.raises(FineTuneError, match="model"):
        fine_tune(client=_StubClient(), model="",
                  dataset_uri="hf://d")


def test_empty_dataset_rejected():
    with pytest.raises(FineTuneError, match="dataset_uri"):
        fine_tune(client=_StubClient(), model="m",
                  dataset_uri="")


def test_client_without_jobs_rejected():
    class Bare: ...
    with pytest.raises(FineTuneError, match=".jobs.submit"):
        fine_tune(client=Bare(), model="m", dataset_uri="hf://d")


# ---------------------------------------------------------------------------
# Submission flow
# ---------------------------------------------------------------------------


def test_successful_submit_returns_handle_with_job_id():
    client = _StubClient()
    job = fine_tune(client=client, model="hf/llama-3-8b",
                    dataset_uri="hf://x/y")
    assert job.job_id == "ft-job-1"
    assert job.state == "submitted"
    assert job.spec.epochs == 3                  # default
    assert client.jobs.submitted_kwargs is not None
    assert client.jobs.submitted_kwargs["kind"] == "training"


def test_submit_passes_privacy_to_jobspec():
    client = _StubClient()
    fine_tune(client=client, model="m", dataset_uri="hf://d",
              privacy_class="sensitive")
    assert client.jobs.submitted_kwargs["privacy_class"] == "sensitive"


def test_submit_passes_cost_ceiling_to_jobspec():
    client = _StubClient()
    fine_tune(client=client, model="m", dataset_uri="hf://d",
              cost_ceiling_usd=12.5)
    assert client.jobs.submitted_kwargs["cost_ceiling_usd"] == 12.5


def test_canonical_hash_is_deterministic():
    spec = FineTuneSpec(model="m", dataset_uri="hf://d")
    assert spec.canonical_hash() == FineTuneSpec(
        model="m", dataset_uri="hf://d"
    ).canonical_hash()


# ---------------------------------------------------------------------------
# Blocking flow
# ---------------------------------------------------------------------------


def test_blocking_completes_when_state_reaches_completed():
    client = _StubClient()
    job = fine_tune_blocking(
        client=client,
        model="m", dataset_uri="hf://d",
        poll_interval_s=0.0, timeout_s=2.0,
    )
    assert job.is_complete()
    assert job.final_model_uri == "ipfs://final-checkpoint"
    assert len(job.checkpoints) == 1


def test_blocking_returns_failed_on_timeout():
    """If state never advances out of running before timeout, return
    job marked failed (not infinite hang)."""
    client = _StubClient(jobs=_StubJobs(get_responses=[
        {"job_id": "ft-job-1", "state": "running", "checkpoints": []}
        for _ in range(100)
    ]))
    job = fine_tune_blocking(
        client=client,
        model="m", dataset_uri="hf://d",
        poll_interval_s=0.0, timeout_s=0.05,
    )
    assert job.is_failed()
    assert job.error == "timeout"
