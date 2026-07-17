"""Fine-tune-on-the-mesh SDK (PNIS §A4).

One call:

    from pluginfer import fine_tune

    job = fine_tune(
        client=Pluginfer(api_key="pf_live_..."),
        model="hf/llama-3-8b",
        dataset_uri="hf://my-org/customer-support-dialogues",
        epochs=3,
        peers=10,
        gradient_provenance=True,
    )

The SDK marshals a training spec into a regular Pluginfer JobSpec
(kind="training"), submits it to the auction layer, and returns a
TrainingJob handle whose `.state` and `.checkpoints` track progress
across the §A14 hot-migration boundaries.

Why this design is novel
----------------------
Hugging Face exposes "click to inference"; AWS exposes
"click-to-train" but only on AWS-controlled hardware. Pluginfer's
contribution:

  "A high-level SDK call submitting a parameterized fine-tune job to
   a permissionless decentralised compute mesh, with K-redundant
   gradient dispatch, optional zero-knowledge gradient-provenance
   attestation, and on-chain settlement of partial-step receipts that
   compose into a final attested checkpoint."

Honesty
-------
This SDK function is the SUBMIT side; the actual training execution
happens on the mesh providers and is wired via the existing
`core/diloco_*` and `core/redundant_dispatcher` machinery. The SDK
returns immediately with a TrainingJob that the caller polls; full
training cost / time depends on the mesh's available capacity, the
dataset size, and the rented peer count.

The function refuses to run if the configured client lacks an API
key OR if the requested peer count is higher than the SDK's safety
ceiling -- preventing a typo from accidentally renting 10000 GPUs.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Spec + handle types
# ---------------------------------------------------------------------------


@dataclass
class FineTuneSpec:
    """Validated training spec ready for submission."""
    model: str                               # e.g. "hf/llama-3-8b"
    dataset_uri: str                         # e.g. "hf://org/dataset"
    epochs: int = 3
    learning_rate: float = 3e-4
    batch_size: int = 8
    peers: int = 4                           # K-redundant dispatch fan-out
    gradient_provenance: bool = True         # §1 ZK attestation on each grad
    quorum_threshold: float = 2.0 / 3.0      # consensus fraction for accept
    privacy_class: str = "private"           # "public"|"private"|"sensitive"
    cost_ceiling_usd: float = 50.0           # hard ceiling for the WHOLE run
    deadline_hours: int = 24

    def canonical_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True,
                       separators=(",", ":")).encode()
        ).hexdigest()

    def to_job_payload(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingCheckpoint:
    """A reference to one signed checkpoint produced during the run."""
    seq: int
    checkpoint_uri: str
    sha256: str
    produced_at_ns: int


@dataclass
class TrainingJob:
    """Handle returned by fine_tune(). Polls the API for progress."""
    job_id: str
    spec: FineTuneSpec
    state: str = "submitted"                 # submitted|running|completed|failed
    submitted_at_ns: int = 0
    checkpoints: List[TrainingCheckpoint] = field(default_factory=list)
    final_model_uri: Optional[str] = None
    error: Optional[str] = None

    def is_complete(self) -> bool:
        return self.state == "completed"

    def is_failed(self) -> bool:
        return self.state == "failed"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


_MAX_PEER_FANOUT = 64                        # safety ceiling


class FineTuneError(RuntimeError):
    pass


def _validate(spec: FineTuneSpec) -> None:
    if spec.epochs < 1:
        raise FineTuneError(f"epochs must be >= 1, got {spec.epochs}")
    if spec.peers < 1:
        raise FineTuneError(f"peers must be >= 1, got {spec.peers}")
    if spec.peers > _MAX_PEER_FANOUT:
        raise FineTuneError(
            f"peers={spec.peers} exceeds SDK safety ceiling "
            f"(_MAX_PEER_FANOUT={_MAX_PEER_FANOUT}). Set "
            f"client.confirm_high_fanout=True or split into multiple jobs.")
    if not (0.0 < spec.quorum_threshold <= 1.0):
        raise FineTuneError("quorum_threshold must be in (0, 1]")
    if spec.cost_ceiling_usd <= 0:
        raise FineTuneError("cost_ceiling_usd must be > 0")
    if spec.deadline_hours < 1:
        raise FineTuneError("deadline_hours must be >= 1")
    if spec.privacy_class not in ("public", "private", "sensitive"):
        raise FineTuneError(
            f"privacy_class must be one of public|private|sensitive, "
            f"got '{spec.privacy_class}'"
        )
    if not spec.model:
        raise FineTuneError("model must be a non-empty identifier")
    if not spec.dataset_uri:
        raise FineTuneError("dataset_uri must be a non-empty URI")


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def fine_tune(
    *,
    client: Any,                             # PluginferClient or test stub
    model: str,
    dataset_uri: str,
    epochs: int = 3,
    learning_rate: float = 3e-4,
    batch_size: int = 8,
    peers: int = 4,
    gradient_provenance: bool = True,
    quorum_threshold: float = 2.0 / 3.0,
    privacy_class: str = "private",
    cost_ceiling_usd: float = 50.0,
    deadline_hours: int = 24,
) -> TrainingJob:
    """Submit a fine-tune job to the Pluginfer mesh and return a handle.

    The handle's `.job_id`, `.state` etc. are populated from the API
    response. `.checkpoints` is empty at submit time; the caller polls
    via `client.jobs.get(job.job_id)` (or the websocket stream) and
    updates the local handle as new checkpoints arrive.

    For a full end-to-end runner that polls + resolves to the final
    checkpoint URI, use `fine_tune_blocking()`.
    """
    spec = FineTuneSpec(
        model=model,
        dataset_uri=dataset_uri,
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        peers=int(peers),
        gradient_provenance=bool(gradient_provenance),
        quorum_threshold=float(quorum_threshold),
        privacy_class=privacy_class,
        cost_ceiling_usd=float(cost_ceiling_usd),
        deadline_hours=int(deadline_hours),
    )
    _validate(spec)

    # The submit side is whatever JobsClient the user gave us. The
    # tests inject a stub. Production wires this to client.jobs.submit().
    submit = getattr(client, "jobs", None)
    if submit is None or not hasattr(submit, "submit"):
        raise FineTuneError(
            "client lacks a .jobs.submit(...) method -- pass a Pluginfer SDK "
            "client or test stub.")
    resp = submit.submit(
        kind="training",
        payload=spec.to_job_payload(),
        cost_ceiling_usd=spec.cost_ceiling_usd,
        latency_ceiling_ms=spec.deadline_hours * 3600 * 1000,
        privacy_class=spec.privacy_class,
        quality_floor=spec.quorum_threshold,
    )
    job_id = getattr(resp, "job_id", None) or resp["job_id"]

    return TrainingJob(
        job_id=str(job_id),
        spec=spec,
        state=getattr(resp, "state", None) or resp.get("state", "submitted"),
        submitted_at_ns=time.time_ns(),
    )


def fine_tune_blocking(
    *,
    client: Any,
    poll_interval_s: float = 5.0,
    timeout_s: float = 24 * 3600,
    **fine_tune_kwargs: Any,
) -> TrainingJob:
    """Submit + poll + return when complete or failed. Suitable for
    notebook / CLI usage where the caller wants the final checkpoint.

    The SDK does NOT block forever: timeout_s caps the wait so an
    infinite hang is impossible.
    """
    handle = fine_tune(client=client, **fine_tune_kwargs)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if handle.is_complete() or handle.is_failed():
            return handle
        time.sleep(poll_interval_s)
        try:
            update = client.jobs.get(handle.job_id)
        except Exception as e:
            handle.state = "failed"
            handle.error = f"poll failed: {e!r}"
            return handle
        handle.state = (
            getattr(update, "state", None)
            or update.get("state", handle.state)
        )
        cps = (getattr(update, "checkpoints", None)
               or update.get("checkpoints", []) or [])
        handle.checkpoints = [
            TrainingCheckpoint(
                seq=int(c.get("seq", i)),
                checkpoint_uri=str(c.get("uri", "")),
                sha256=str(c.get("sha256", "")),
                produced_at_ns=int(c.get("produced_at_ns", 0)),
            )
            for i, c in enumerate(cps)
        ]
        handle.final_model_uri = (
            getattr(update, "final_model_uri", None)
            or update.get("final_model_uri")
        )
    handle.state = "failed"
    handle.error = "timeout"
    return handle


__all__ = [
    "FineTuneSpec",
    "FineTuneError",
    "TrainingJob",
    "TrainingCheckpoint",
    "fine_tune",
    "fine_tune_blocking",
]
