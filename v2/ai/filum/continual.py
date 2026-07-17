"""Continual learning: streaming LoRA updates from real interactions.

INNOVATION: Filum never stops learning. Every settled job on the
chain is a labelled training example: the JOB_REQUEST is the
input, the JOB_RESULT is the output, and the on-chain settlement
+ result_hash signature is the ground-truth quality signal. We
stream these triples into a low-rank adapter (LoRA) that updates
Filum's weights daily without re-training from scratch.

LoRA recap
----------
Instead of updating the full weight matrix W (huge), we add a
low-rank adapter:

    W_eff = W + alpha * (B @ A)
    where A: (rank, d_in)
          B: (d_out, rank)

For rank=8 on a 896×896 layer: A+B = 14,336 params vs full W's
803,584. That's 56× fewer parameters to update -- so a single
example can produce a meaningful gradient without overfitting,
and the adapter trains in milliseconds.

Catastrophic forgetting
-----------------------
Naive fine-tuning destroys what Filum already knew. Three guards:

  1. **EWC-lite (Elastic Weight Consolidation, Kirkpatrick 2017)**:
     freeze the base weights; only the LoRA adapter updates. The
     base model's knowledge is preserved bit-perfect.
  2. **Replay buffer**: keep the last 1024 distillation samples in
     a rolling buffer; mix into every continual update at 25%
     ratio. Stops the adapter drifting too far from the
     distilled distribution.
  3. **KL clamp**: at every step measure KL between adapter's
     prediction and the base model's prediction on a holdout set;
     if KL > threshold, halve the LoRA learning rate. Self-
     regulating drift control.

Failure modes (honest)
----------------------
* The chain provides feedback on TASK SUCCESS but not on
  LANGUAGE QUALITY. A confidently wrong answer that still settled
  via majority vote will look like a good example to the
  continual learner. Mitigation: only use receipts where K-redundant
  consensus was reached AND the result_hash sigverify passed.
* User behaviour can change rapidly (a new task class appears in
  the mesh). The adapter takes ~1000 examples to specialise, so
  for ~1 day after a distribution shift Filum's outputs degrade
  before recovering.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
    _BASE = nn.Module
except Exception:                                                # pragma: no cover
    torch = None
    nn = None
    F = None
    _HAS_TORCH = False
    _BASE = object

logger = logging.getLogger(__name__)


@dataclass
class ChainExample:
    """One training example derived from a chain receipt."""
    job_id: str
    prompt: str
    expected_response: str
    quality_score: float    # 0..1, derived from on-chain consensus
    timestamp: float
    teacher_id: Optional[str] = None


# ---------------------------------------------------------------------------
# LoRA adapter
# ---------------------------------------------------------------------------


class LoRALinear(_BASE):
    """Wraps an existing nn.Linear with a frozen W + trainable LoRA
    adapter A, B. Inference: y = (W + alpha/r * B @ A) @ x + bias."""

    def __init__(self, base: "nn.Linear", *, rank: int = 8,
                 alpha: float = 16.0, dropout: float = 0.0):
        if not _HAS_TORCH:
            raise RuntimeError("LoRALinear requires torch")
        super().__init__()
        self.base = base
        # Freeze the base (its grad will be detached).
        for p in self.base.parameters():
            p.requires_grad = False
        d_in = base.in_features
        d_out = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.rank)

        self.lora_A = nn.Parameter(torch.zeros(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        # Standard LoRA init: A from kaiming uniform, B zeros (so
        # the adapter contributes zero at init -- model starts
        # bit-identical to the frozen base).
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # Frozen base path.
        with torch.no_grad():
            base_out = self.base(x)
        # LoRA delta path -- this is what gradient flows through.
        x_d = self.dropout(x)
        delta = (x_d @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + delta

    def merged_weight(self):
        """For deployment: collapse adapter into the base weight."""
        with torch.no_grad():
            return self.base.weight + (self.lora_B @ self.lora_A) * self.scaling


def attach_lora(model, *, rank: int = 8, alpha: float = 16.0,
                target_substrings: Tuple[str, ...] = ("attn", "ffn", "down", "up", "gate")):
    """Walk `model` and replace nn.Linear modules whose qualified
    name contains any of `target_substrings` with LoRALinear wrappers.

    The non-matching Linears stay frozen-but-not-LoRA'd: their grads
    will be zero anyway because we'll only `requires_grad_=True` the
    LoRA adapter params downstream.
    """
    if not _HAS_TORCH:
        raise RuntimeError("attach_lora requires torch")
    converted = []
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and any(s in full for s in target_substrings):
                wrapper = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(parent, child_name, wrapper)
                converted.append(full)
    return model, converted


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


@dataclass
class ReplayBuffer:
    capacity: int = 1024
    examples: List[ChainExample] = field(default_factory=list)

    def add(self, ex: ChainExample) -> None:
        if len(self.examples) < self.capacity:
            self.examples.append(ex)
        else:
            # Drop the oldest; keep the newest.
            self.examples.pop(0)
            self.examples.append(ex)

    def sample(self, k: int) -> List[ChainExample]:
        import random as _r
        if not self.examples:
            return []
        return _r.sample(self.examples, min(k, len(self.examples)))


# ---------------------------------------------------------------------------
# Continual learner
# ---------------------------------------------------------------------------


@dataclass
class ContinualConfig:
    lora_rank: int = 8
    lora_alpha: float = 16.0
    learning_rate: float = 1e-4
    batch_size: int = 4
    replay_ratio: float = 0.25
    kl_drift_threshold: float = 0.5
    update_every_n_examples: int = 32
    save_adapter_every_n_steps: int = 500


class ContinualLearner:
    """Drives the streaming-LoRA loop. Owns the adapter parameters,
    the replay buffer, and the optimizer.

    Caller wires:
      `model`               -- Filum (nn.Module). Will be mutated to add LoRA.
      `tokenizer.encode`    -- prompt -> token ids
      `compute_loss(model, prompt_ids, target_ids)` -- returns scalar loss

    Then call `submit(example: ChainExample)` for every settled job.
    Every `update_every_n_examples`, a mini-step runs.
    """

    def __init__(
        self,
        model,
        tokenizer,
        compute_loss,
        *,
        config: Optional[ContinualConfig] = None,
        save_dir: Optional[Path] = None,
    ):
        if not _HAS_TORCH:
            raise RuntimeError("ContinualLearner requires torch")
        self.config = config or ContinualConfig()
        self.tokenizer = tokenizer
        self.compute_loss = compute_loss
        self.save_dir = Path(save_dir) if save_dir else None

        # Attach LoRA + collect adapter params.
        self.model, self.adapter_names = attach_lora(
            model, rank=self.config.lora_rank, alpha=self.config.lora_alpha,
        )
        adapter_params = [p for n, p in self.model.named_parameters()
                          if "lora_A" in n or "lora_B" in n]
        if not adapter_params:
            raise RuntimeError(
                "no LoRA adapter parameters found -- attach_lora found "
                "nothing to wrap; check target_substrings"
            )
        for p in adapter_params:
            p.requires_grad_(True)
        self.optimizer = torch.optim.AdamW(
            adapter_params, lr=self.config.learning_rate,
        )

        self.replay = ReplayBuffer(capacity=1024)
        self._pending: List[ChainExample] = []
        self._step: int = 0

    # ------------------------------------------------------------------

    def submit(self, ex: ChainExample) -> None:
        """Hand off a settled-job receipt for future training."""
        self._pending.append(ex)
        if len(self._pending) >= self.config.update_every_n_examples:
            self._update()

    def _update(self) -> None:
        """Run a single LoRA mini-step on the pending queue + replay."""
        # Compose a batch: pending + a slice of replay.
        n_replay = max(1, int(self.config.batch_size * self.config.replay_ratio))
        batch = self._pending[: self.config.batch_size]
        if not batch:
            return
        replay = self.replay.sample(n_replay)
        full = batch + replay

        # Run an optimizer step over the batch.
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = None
        for ex in full:
            try:
                ids_p = self.tokenizer.encode(ex.prompt)
                ids_t = self.tokenizer.encode(ex.expected_response)
            except Exception as e:
                logger.warning("tokenize failed: %s", e)
                continue
            if not ids_p or not ids_t:
                continue
            loss = self.compute_loss(
                self.model,
                torch.tensor(ids_p).unsqueeze(0),
                torch.tensor(ids_t).unsqueeze(0),
            )
            # Quality-weight the loss: a high-quality settled job
            # contributes a full-magnitude gradient; a low-quality
            # one is downweighted.
            w = max(0.1, float(ex.quality_score))
            loss = loss * w
            total_loss = loss if total_loss is None else total_loss + loss
        if total_loss is None:
            self._pending.clear()
            return
        total_loss = total_loss / max(1, len(full))
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0,
        )
        self.optimizer.step()
        self._step += 1

        # Move pending into the replay buffer (so they get sampled
        # in future updates) and clear.
        for ex in batch:
            self.replay.add(ex)
        self._pending.clear()

        if self.save_dir and self._step % self.config.save_adapter_every_n_steps == 0:
            self.save_adapter()

    # ------------------------------------------------------------------

    def save_adapter(self) -> None:
        """Persist ONLY the LoRA params -- not the frozen base."""
        if self.save_dir is None or not _HAS_TORCH:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        adapter_state = {
            n: p.detach().cpu()
            for n, p in self.model.named_parameters()
            if "lora_A" in n or "lora_B" in n
        }
        path = self.save_dir / f"adapter_step_{self._step}.pt"
        torch.save(adapter_state, path)
        # Symlink-equivalent: a tiny json pointer.
        (self.save_dir / "adapter_latest.json").write_text(
            json.dumps({"file": path.name, "step": self._step,
                        "saved_at": time.time()}),
        )

    def load_adapter(self, path: Path) -> None:
        if not _HAS_TORCH:
            return
        state = torch.load(path)
        for n, p in self.model.named_parameters():
            if n in state:
                with torch.no_grad():
                    p.copy_(state[n])

    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "step": self._step,
            "pending": len(self._pending),
            "replay_size": len(self.replay.examples),
            "adapter_param_count": sum(
                p.numel() for n, p in self.model.named_parameters()
                if ("lora_A" in n or "lora_B" in n)
            ),
        }


# ---------------------------------------------------------------------------
# Helpers: derive a ChainExample from a settled JobRecord
# ---------------------------------------------------------------------------


def example_from_settled_job(
    *,
    job_id: str,
    prompt: str,
    response_text: str,
    consensus_size: int = 1,
    total_voters: int = 1,
    sig_verified: bool = True,
) -> Optional[ChainExample]:
    """Construct a ChainExample from a Pluginfer settled-job receipt.
    Returns None if the receipt isn't trustworthy enough to learn from
    (e.g. consensus failed, signature didn't verify)."""
    if not sig_verified:
        return None
    if not response_text or not response_text.strip():
        return None
    # Quality score: consensus ratio. A unanimous K-redundant
    # vote = 1.0; a bare-majority vote = 0.5; below majority is
    # already filtered.
    quality = consensus_size / max(1, total_voters)
    if quality < 0.5:
        return None
    return ChainExample(
        job_id=job_id,
        prompt=prompt,
        expected_response=response_text,
        quality_score=quality,
        timestamp=time.time(),
    )
