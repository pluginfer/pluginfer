"""Checkpoint save / load helpers.

Saved file is a single .pt produced by `torch.save`. Layout:

    {
        "format_version": 1,
        "model_state_dict": ...,
        "optimizer_state_dict": ...,
        "scheduler_state": {"current_lr": float},
        "global_step": int,
        "config": dict (ModelConfig as dict),
        "training_config": dict,
    }
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_FORMAT_VERSION: int = 1


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler_state: dict,
    global_step: int,
    config: Any,
    training_config: Any,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state": scheduler_state,
        "global_step": global_step,
        "config": dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(config),
        "training_config": (
            dataclasses.asdict(training_config)
            if dataclasses.is_dataclass(training_config)
            else dict(training_config)
        ),
    }
    torch.save(body, path)


def load_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    body = torch.load(path, map_location="cpu", weights_only=False)
    if body.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format_version "
            f"{body.get('format_version')!r}; expected {CHECKPOINT_FORMAT_VERSION}"
        )
    return body


class Checkpointer:
    """Periodic checkpoint manager. Keeps the last `keep_n` checkpoints."""

    def __init__(self, dir_path: str | Path, keep_n: int = 3) -> None:
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_n = keep_n

    def save(
        self,
        global_step: int,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler_state: dict,
        config: Any,
        training_config: Any,
    ) -> Path:
        path = self.dir / f"ckpt_step_{global_step:08d}.pt"
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler_state=scheduler_state,
            global_step=global_step,
            config=config,
            training_config=training_config,
        )
        self._prune()
        return path

    def _prune(self) -> None:
        ckpts = sorted(self.dir.glob("ckpt_step_*.pt"))
        for old in ckpts[: -self.keep_n] if self.keep_n > 0 else []:
            old.unlink(missing_ok=True)

    def latest(self) -> Path | None:
        ckpts = sorted(self.dir.glob("ckpt_step_*.pt"))
        return ckpts[-1] if ckpts else None
