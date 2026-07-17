"""MeshTrainer - honest stub for the Pluginfer-dogfooded training path.

The vision is: PluginferLM trains on the very GPU mesh it helps
orchestrate. Each provider node holds the full model, processes a
shard of the batch, and gradients are averaged via the mesh's
communication layer (signed gossip / task_router envelopes carry
all-reduce messages).

This file currently stubs the surface. The mesh transport itself,
the trust model for gradient aggregation (which is itself a research
problem - a malicious provider can submit poisoned gradients), and
the elastic add/drop of nodes during a long run are all multi-week
work that depends on DEEP-W19 (mesh-on-install protocol) and W32
(BFT slash evidence) being shipped first.

When implementing for real, integrate:
  - core/gradient_provenance.py: Pedersen proof binding gradient to
    (data shard, model checkpoint) - prevents free-riding and
    poisoning attacks
  - core/task_router.py: K-redundant dispatch + brain-trust filtering
  - core/bft_consensus.py: validator set decides which gradient
    aggregations are admitted into the chain
"""

from __future__ import annotations

from .distributed import DDPNotImplementedError


class MeshTrainerNotImplementedError(DDPNotImplementedError):
    pass


class MeshTrainer:
    def __init__(self, *args, **kwargs) -> None:
        raise MeshTrainerNotImplementedError(
            "MeshTrainer requires DEEP-W19 (mesh-on-install protocol), "
            "the gradient-provenance ZK proof to be wired into the "
            "all-reduce path, and a stake-weighted aggregator selection. "
            "Single-process Trainer is the current supported path."
        )

    def all_reduce_gradients(self) -> None:
        raise MeshTrainerNotImplementedError(
            "all_reduce_gradients not yet implemented; see roadmap."
        )

    def elastic_add_node(self, new_node_pubkey: str) -> None:
        raise MeshTrainerNotImplementedError(
            "elastic_add_node not yet implemented; see roadmap."
        )

    def handle_node_failure(self, failed_pubkey: str) -> None:
        raise MeshTrainerNotImplementedError(
            "handle_node_failure not yet implemented; see roadmap."
        )
