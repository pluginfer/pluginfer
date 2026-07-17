"""DDP wrapper - honest stub.

Real Data-Parallel training across the Pluginfer mesh requires:
  - Process-group bootstrap over the mesh communication layer (gossip /
    task_router envelopes carrying NCCL / Gloo handshake)
  - Ring-allreduce of gradients with bandwidth-optimal scheduling
  - Health monitoring + elastic add/drop of nodes
  - Checkpoint broadcast + dataloader sharding

None of that fits in a single-node coding session. Per project
discipline (`core/privacy.py` template), we expose the surface as an
honest stub rather than ship a mock.

The single-process Trainer already supports `device='cuda'` for single-
GPU runs; DDP wraps it for multi-GPU/multi-node.
"""

from __future__ import annotations


class DDPNotImplementedError(NotImplementedError):
    """Raised when callers try to use the DDP path before it's wired."""


def wrap_ddp(model, world_size: int, rank: int):
    raise DDPNotImplementedError(
        "DDP wrapper not yet implemented. Single-process Trainer is the "
        "current supported path. See PNIS roadmap CP-AI-FINAL+ for the "
        "mesh-trainer integration."
    )


def init_process_group(*, backend: str = "gloo", init_method: str = "") -> None:
    raise DDPNotImplementedError(
        "init_process_group not yet implemented. The Pluginfer mesh "
        "communication layer is what would carry the NCCL/Gloo handshake; "
        "see WORKLOG DEEP-W19 for the bootstrap protocol prerequisites."
    )
