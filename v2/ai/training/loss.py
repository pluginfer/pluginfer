"""Loss functions for the LM backbone and the task heads."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def lm_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """Standard next-token-prediction cross-entropy.

    `logits` shape: (B, T, V); `labels` shape: (B, T) with -100 for
    ignored positions (PAD or end-of-sequence).
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


class MultiTaskLoss:
    """Weighted sum of per-task losses.

    Used by the Trainer when training the backbone jointly with task
    heads. The weights are typically driven by `CurriculumScheduler`.
    """

    def __init__(self, weights: Optional[dict[str, float]] = None) -> None:
        self.weights = weights or {}

    def __call__(self, losses: dict[str, Tensor]) -> Tensor:
        if not losses:
            raise ValueError("MultiTaskLoss called with empty losses dict")
        total: Tensor | None = None
        for name, loss in losses.items():
            w = self.weights.get(name, 1.0)
            if w == 0.0:
                continue
            term = loss * w
            total = term if total is None else total + term
        if total is None:
            # Every weight was zero - return a 0 grad-able tensor in the right
            # device/dtype.
            example = next(iter(losses.values()))
            total = torch.zeros((), device=example.device, dtype=example.dtype)
        return total
