"""Weekly fine-tuner - honest stub.

The intent is: every Sunday 00:00 UTC, take the labelled flywheel
dataset and run a fine-tune of the live model on the Pluginfer mesh.
A/B-evaluate the new checkpoint against the current production
checkpoint; if eval metrics improve, promote.

Requires:
  - A real OutcomeLabeler (see labeler.py)
  - Mesh-distributed training (see ai/training/mesh_trainer.py - also
    a stub)
  - Compute budget allocation policy (validators decide which mesh
    slices are reserved for in-house training)
  - Versioning / blue-green for the production checkpoint

None of these fit in the current session.
"""

from __future__ import annotations


class FineTuningNotImplementedError(NotImplementedError):
    pass


class WeeklyFineTuner:
    def __init__(self, *args, **kwargs) -> None:
        raise FineTuningNotImplementedError(
            "WeeklyFineTuner requires the OutcomeLabeler + MeshTrainer + "
            "compute-budget policy + blue-green checkpoint promotion to be "
            "implemented. See ai/flywheel/labeler.py, ai/training/mesh_trainer.py, "
            "and TODO §7 / DEEP-W19 for the integration roadmap."
        )

    def run(self) -> None:
        raise FineTuningNotImplementedError("WeeklyFineTuner.run not implemented")
