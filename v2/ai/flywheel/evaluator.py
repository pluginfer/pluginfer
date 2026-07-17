"""A/B evaluator: compare two model checkpoints on a held-out set.

Given two trained PluginferLM checkpoints, run both on the same val
DataLoader and report mean cross-entropy and perplexity. The promote
decision is the caller's; this module returns the metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch

from ai.model.transformer import PluginferLM
from ai.training.loss import lm_cross_entropy


@dataclass
class CheckpointEvalResult:
    name: str
    mean_loss: float
    mean_ppl: float
    n_batches: int


class ABEvaluator:
    def __init__(self, val_loader: Iterable[dict], device: str = "cpu") -> None:
        # Materialise the val_loader once so both checkpoints see the same data.
        self.val_batches: list[dict] = list(val_loader)
        self.device = device

    @torch.no_grad()
    def _eval_one(
        self, model: PluginferLM, name: str
    ) -> CheckpointEvalResult:
        model.to(self.device).eval()
        losses: list[float] = []
        for batch in self.val_batches:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            logits = model(input_ids)
            losses.append(float(lm_cross_entropy(logits, labels).item()))
        if not losses:
            raise ValueError("ABEvaluator val_loader is empty")
        mean = sum(losses) / len(losses)
        return CheckpointEvalResult(
            name=name,
            mean_loss=mean,
            mean_ppl=math.exp(min(mean, 50.0)),
            n_batches=len(losses),
        )

    def compare(
        self,
        model_a: PluginferLM,
        model_b: PluginferLM,
        name_a: str = "A",
        name_b: str = "B",
    ) -> dict:
        a = self._eval_one(model_a, name_a)
        b = self._eval_one(model_b, name_b)
        return {
            "a": a,
            "b": b,
            "winner": name_a if a.mean_loss <= b.mean_loss else name_b,
            "delta_loss": b.mean_loss - a.mean_loss,
        }
