"""
DiLoCo Worker (Inner Loop)
==========================
Runs the local SGD inner loop of DiLoCo (Distributed Low-Communication
training, Douillard et al., 2023). One pass = one "round":

   1. Load global weights from the aggregator (or initialize fresh).
   2. Run K steps of inner-loop SGD/AdamW on local data.
   3. Compute delta = θ_local - θ_global.
   4. Ship the delta back to the aggregator (quantized for WAN).

Crucially: the worker only ever sees its own data. The bytes that leave
the machine are gradient *deltas*, never raw inputs. That's the privacy
property real federated training was supposed to give — and what the
old `plugins/federated_trainer.py` only pretended to do via `time.sleep`.

Production note
---------------
For 7B+ models the inner loop should use AdamW + gradient checkpointing
+ optionally LoRA-only training to keep memory in budget on consumer
GPUs. The interface below is fully decoupled from architecture so the
same worker handles MLP, CNN, Transformer, or LLM.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except Exception as _torch_err:                      # pragma: no cover
    torch = None                                     # type: ignore[assignment]
    nn = None                                        # type: ignore[assignment]
    DataLoader = None                                # type: ignore[assignment]
    TensorDataset = None                             # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = _torch_err

from .diloco_models import build_model, loss_fn_for, count_parameters
from .diloco_serialize import deserialize_state_dict, state_dict_hash
from .diloco_quantize import quantize_delta, estimate_compression_ratio

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """What a worker reports after finishing one DiLoCo round."""
    quantized_delta: bytes
    initial_loss: float
    final_loss: float
    examples_seen: int
    inner_steps: int
    wall_time: float
    delta_norm: float
    compression_ratio: float
    base_weights_hash: str
    final_weights_hash: str
    device: str
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class InnerLoopConfig:
    """Hyperparameters for one DiLoCo inner loop."""
    inner_steps: int = 50           # K in the paper; typical 50-500
    inner_lr: float = 1e-2
    inner_momentum: float = 0.9
    weight_decay: float = 0.0
    batch_size: int = 32
    grad_clip_norm: Optional[float] = 1.0
    optimizer: str = "sgd"          # 'sgd' | 'adamw'
    quantize: bool = True


def _select_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def _build_optimizer(params, cfg: InnerLoopConfig) -> torch.optim.Optimizer:
    if cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(params, lr=cfg.inner_lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(
        params, lr=cfg.inner_lr,
        momentum=cfg.inner_momentum,
        weight_decay=cfg.weight_decay,
    )


class DiLoCoWorker:
    """
    Stateless worker. One instance == one inference engine.
    Multiple rounds reuse the same model object to avoid reallocation.
    """

    def __init__(self, model_spec: Dict[str, object], device_pref: str = "auto"):
        self.model_spec = dict(model_spec)
        self.device = _select_device(device_pref)
        self.model: nn.Module = build_model(self.model_spec).to(self.device)
        self.loss_fn: nn.Module = loss_fn_for(self.model_spec).to(self.device)
        logger.info(
            "DiLoCoWorker built: arch=%s params=%d device=%s",
            self.model_spec.get("arch"), count_parameters(self.model), self.device,
        )

    # ---- weight transfer ----
    def load_global_weights(self, payload: Optional[bytes]) -> None:
        """Replace model weights from a serialized aggregator checkpoint."""
        if payload is None:
            return  # keep init weights for round 0
        expected = {k: tuple(v.shape) for k, v in self.model.state_dict().items()}
        new_state = deserialize_state_dict(
            payload,
            expected_keys=tuple(expected.keys()),
            expected_shapes=expected,
        )
        new_state = {k: v.to(self.device) for k, v in new_state.items()}
        self.model.load_state_dict(new_state, strict=True)

    def export_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    # ---- one DiLoCo round ----
    def run_round(self,
                  data_iter: Callable[[int], Tuple[torch.Tensor, torch.Tensor]],
                  cfg: InnerLoopConfig,
                  global_weights_payload: Optional[bytes] = None,
                  ) -> TrainResult:
        """
        data_iter(batch_size) -> (x, y) callable.
        Lets callers stream local data without exposing it on the wire.
        """
        # 1. Snapshot global weights pre-training
        self.load_global_weights(global_weights_payload)
        global_state = self.export_weights()
        global_hash = state_dict_hash(global_state)

        # 2. Inner loop
        opt = _build_optimizer(self.model.parameters(), cfg)
        self.model.train()

        initial_loss: Optional[float] = None
        last_loss: float = 0.0
        examples_seen = 0
        steps_done = 0
        t0 = time.time()

        for step in range(cfg.inner_steps):
            x, y = data_iter(cfg.batch_size)
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            loss.backward()

            if cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_norm)
            opt.step()

            loss_val = float(loss.detach().item())
            if initial_loss is None:
                initial_loss = loss_val
            last_loss = loss_val
            examples_seen += int(x.shape[0])
            steps_done += 1

        wall_time = time.time() - t0
        if initial_loss is None:
            initial_loss = last_loss

        # 3. Compute delta = local - global, on CPU fp32
        local_state = self.export_weights()
        local_hash = state_dict_hash(local_state)

        delta: Dict[str, torch.Tensor] = {}
        for k, local_t in local_state.items():
            delta[k] = (local_t.detach().cpu().to(torch.float32)
                        - global_state[k].detach().cpu().to(torch.float32))
        delta_norm = float(
            torch.sqrt(sum((t.float().pow(2).sum() for t in delta.values()))).item()
        )

        # 4. Quantize for transport
        if cfg.quantize:
            quantized = quantize_delta(delta)
            comp = estimate_compression_ratio(delta)
        else:
            from .diloco_serialize import serialize_state_dict
            quantized = serialize_state_dict(delta)
            comp = 1.0

        return TrainResult(
            quantized_delta=quantized,
            initial_loss=float(initial_loss),
            final_loss=float(last_loss),
            examples_seen=examples_seen,
            inner_steps=steps_done,
            wall_time=wall_time,
            delta_norm=delta_norm,
            compression_ratio=comp,
            base_weights_hash=global_hash,
            final_weights_hash=local_hash,
            device=str(self.device),
            metadata={
                "param_count": count_parameters(self.model),
                "optimizer": cfg.optimizer,
                "inner_lr": cfg.inner_lr,
            },
        )


# ----------------------------------------------------------------------
# Convenience: build a tensor data_iter from in-memory tensors.
# Keeps the local data on the worker; the iterator never leaves the box.
# ----------------------------------------------------------------------
def make_tensor_iter(x: torch.Tensor, y: torch.Tensor, seed: int = 0
                     ) -> Callable[[int], Tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator()
    g.manual_seed(seed)
    n = x.shape[0]

    def _iter(batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, n, (batch_size,), generator=g)
        return x[idx], y[idx]

    return _iter
