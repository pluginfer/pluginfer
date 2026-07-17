"""Trainer: the main training loop for PluginferLM.

Designed to run on:
  1. Single CPU (unit tests, the 100-step CP-AI-4 gate)
  2. Single GPU (development)
  3. Multi-GPU DDP (mesh_trainer.py is the wrapper)

Returns a `TrainMetrics` dict: initial_loss, final_loss, initial_ppl,
final_ppl, max_grad_norm, n_steps, n_evals, best_val_loss. The CP-AI-4
gate asserts final_loss < initial_loss on this object.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader

from ai.model.transformer import PluginferLM

from .checkpointing import Checkpointer, load_checkpoint
from .loss import lm_cross_entropy
from .optimizer import AdamW, CosineSchedulerWithWarmup


@dataclass
class TrainingConfig:
    max_steps: int = 100
    eval_every: int = 50
    log_every: int = 10
    checkpoint_every: int = 0  # 0 = never (CP-AI-4 default)

    max_lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 10
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip_norm: float = 1.0
    grad_accum_steps: int = 1

    device: str = "cpu"
    amp_dtype: str = "none"   # "bf16" / "fp16" / "none"
    seed: int = 0

    checkpoint_dir: str = ""  # if set, Checkpointer is enabled


@dataclass
class TrainMetrics:
    initial_loss: float = 0.0
    final_loss: float = 0.0
    initial_ppl: float = 0.0
    final_ppl: float = 0.0
    max_grad_norm: float = 0.0
    n_steps: int = 0
    n_evals: int = 0
    best_val_loss: float = float("inf")
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _amp_context(dtype: str, device: str):
    if dtype == "none" or device == "cpu":
        # CPU bf16 is technically supported by torch but adds noise on
        # tiny debug models; default to no autocast.
        from contextlib import nullcontext

        return nullcontext()
    if dtype == "bf16":
        return torch.autocast(device_type=device, dtype=torch.bfloat16)
    if dtype == "fp16":
        return torch.autocast(device_type=device, dtype=torch.float16)
    raise ValueError(f"unknown amp_dtype: {dtype}")


class Trainer:
    """Single-process trainer for PluginferLM.

    Public entry points:
      - train(train_loader, val_loader=None) -> TrainMetrics
      - train_step(batch) -> dict
      - eval_step(batch) -> dict
      - save_checkpoint(path) / load_state(checkpoint_dict)
      - classmethod from_checkpoint(path, config)

    State exposed:
      - global_step
      - optimizer
      - scheduler
    """

    def __init__(self, model: PluginferLM, training_config: TrainingConfig) -> None:
        self.model = model
        self.config = training_config
        torch.manual_seed(training_config.seed)
        self.model.to(training_config.device)

        self.optimizer = AdamW(
            self._param_groups(),
            lr=training_config.max_lr,
            betas=training_config.betas,
            eps=training_config.eps,
            weight_decay=0.0,  # weight decay applied per-group below
        )
        # AdamW reads weight_decay per-group; set on the decay group only:
        for g in self.optimizer.param_groups:
            g["weight_decay"] = g.get("weight_decay", training_config.weight_decay)

        self.scheduler = CosineSchedulerWithWarmup(
            self.optimizer,
            max_lr=training_config.max_lr,
            warmup_steps=training_config.warmup_steps,
            max_steps=training_config.max_steps,
            min_lr_ratio=training_config.min_lr_ratio,
        )
        self.global_step: int = 0
        self.checkpointer: Optional[Checkpointer] = (
            Checkpointer(training_config.checkpoint_dir)
            if training_config.checkpoint_dir
            else None
        )

    # ------------------------------------------------------------------
    # Param-group split
    # ------------------------------------------------------------------

    def _param_groups(self) -> list[dict]:
        """Decay 2D+ weights; no decay for biases / norms / 1D params.

        This matches the GPT-3 / Llama practice and is critical for
        avoiding accidental L2 on RMSNorm gamma (which causes a slow
        drift toward zero and degrades training).
        """
        decay, no_decay = [], []
        for _name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay.append(p)
            else:
                no_decay.append(p)
        return [
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.train()
        input_ids = batch["input_ids"].to(self.config.device)
        labels = batch["labels"].to(self.config.device)

        with _amp_context(self.config.amp_dtype, self.config.device):
            logits = self.model(input_ids)
            loss = lm_cross_entropy(logits, labels) / self.config.grad_accum_steps

        loss.backward()
        grad_norm_val = float("nan")

        if (self.global_step + 1) % self.config.grad_accum_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.grad_clip_norm
            )
            grad_norm_val = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
            self.optimizer.step()
            self.scheduler.step(self.global_step)
            self.optimizer.zero_grad(set_to_none=True)

        scaled_loss = float(loss.item() * self.config.grad_accum_steps)
        return {
            "loss": scaled_loss,
            "ppl": math.exp(min(scaled_loss, 50.0)),  # clamp to avoid overflow
            "lr": self.scheduler.current_lr,
            "grad_norm": grad_norm_val,
        }

    @torch.no_grad()
    def eval_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.eval()
        input_ids = batch["input_ids"].to(self.config.device)
        labels = batch["labels"].to(self.config.device)
        logits = self.model(input_ids)
        loss = lm_cross_entropy(logits, labels)
        loss_val = float(loss.item())
        return {"loss": loss_val, "ppl": math.exp(min(loss_val, 50.0))}

    @torch.no_grad()
    def evaluate(self, val_loader: Iterable) -> dict[str, float]:
        losses: list[float] = []
        for batch in val_loader:
            m = self.eval_step(batch)
            losses.append(m["loss"])
        if not losses:
            return {"loss": float("nan"), "ppl": float("nan")}
        mean = sum(losses) / len(losses)
        return {"loss": mean, "ppl": math.exp(min(mean, 50.0))}

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> TrainMetrics:
        metrics = TrainMetrics()
        max_grad_norm = 0.0
        # Prime: read the first batch and capture initial loss BEFORE any step.
        train_iter = self._cycle(train_loader)
        first_batch = next(train_iter)
        with torch.no_grad():
            initial = self.eval_step(first_batch)
        metrics.initial_loss = initial["loss"]
        metrics.initial_ppl = initial["ppl"]

        # Now run training. We fold the first batch back into the iteration.
        seen_first = False

        for step in range(self.config.max_steps):
            if not seen_first:
                batch = first_batch
                seen_first = True
            else:
                batch = next(train_iter)

            step_metrics = self.train_step(batch)
            self.global_step += 1
            if math.isfinite(step_metrics["grad_norm"]):
                max_grad_norm = max(max_grad_norm, step_metrics["grad_norm"])

            if (step + 1) % self.config.log_every == 0:
                metrics.history.append(
                    {"step": self.global_step, **step_metrics}
                )

            if (
                val_loader is not None
                and self.config.eval_every > 0
                and (step + 1) % self.config.eval_every == 0
            ):
                eval_metrics = self.evaluate(val_loader)
                metrics.n_evals += 1
                metrics.history.append(
                    {"step": self.global_step, "eval": eval_metrics}
                )
                if eval_metrics["loss"] < metrics.best_val_loss:
                    metrics.best_val_loss = eval_metrics["loss"]

            if (
                self.checkpointer is not None
                and self.config.checkpoint_every > 0
                and (step + 1) % self.config.checkpoint_every == 0
            ):
                self.checkpointer.save(
                    self.global_step,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler_state={"current_lr": self.scheduler.current_lr},
                    config=self.model.config,
                    training_config=self.config,
                )

        # Final loss = mean of last `eval_every` train steps (or last step if
        # too short)
        last_train_losses = [
            h["loss"] for h in metrics.history if "loss" in h
        ]
        if last_train_losses:
            tail = last_train_losses[-max(1, self.config.eval_every // self.config.log_every) :]
            metrics.final_loss = sum(tail) / len(tail)
            metrics.final_ppl = math.exp(min(metrics.final_loss, 50.0))
        else:
            # Fall back to the final step value if log_every > max_steps
            metrics.final_loss = step_metrics["loss"]
            metrics.final_ppl = step_metrics["ppl"]

        metrics.max_grad_norm = max_grad_norm
        metrics.n_steps = self.global_step
        return metrics

    @staticmethod
    def _cycle(loader: Iterable) -> Iterable:
        while True:
            for batch in loader:
                yield batch

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        from .checkpointing import save_checkpoint

        save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler_state={"current_lr": self.scheduler.current_lr},
            global_step=self.global_step,
            config=self.model.config,
            training_config=self.config,
        )

    def load_state(self, body: dict) -> None:
        self.model.load_state_dict(body["model_state_dict"])
        self.optimizer.load_state_dict(body["optimizer_state_dict"])
        self.global_step = int(body.get("global_step", 0))
        sched = body.get("scheduler_state") or {}
        if "current_lr" in sched:
            self.scheduler.current_lr = float(sched["current_lr"])
            for g in self.optimizer.param_groups:
                g["lr"] = float(sched["current_lr"])

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, training_config: TrainingConfig
    ) -> "Trainer":
        body = load_checkpoint(path)
        from ai.model.config import ModelConfig

        model_cfg = ModelConfig(**body["config"])
        model = PluginferLM(model_cfg)
        trainer = cls(model, training_config)
        trainer.load_state(body)
        return trainer
