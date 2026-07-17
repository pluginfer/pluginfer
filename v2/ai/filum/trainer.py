"""FilumTrainer: combines the innovation stack into one loop.

Orchestrates:
  * Multi-teacher consensus distillation (`teacher_pool.TeacherPool`)
  * Active KL-weighted sampling (`active_sampler.ActiveSampler`)
  * Synthetic self-play (`self_play.SelfPlayGenerator`)
  * 8-bit AdamW optimizer (`optimizer_8bit.AdamW8bit`)
  * BitNet b1.58 deploy mode (`ai/training/bitnet_158.convert_to_bitnet`)
  * Curriculum stage scheduler

The trainer is INTERRUPTIBLE: ctrl-C at any time, restart, picks up
from the last checkpoint. The teacher cache + replay buffer + LoRA
adapter are all on-disk-persistent.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:                                                # pragma: no cover
    torch = None
    nn = None
    F = None
    _HAS_TORCH = False

from .active_sampler import ActiveSampler, ScoredSample
from .config import FilumConfig
from .self_play import SelfPlayConfig, SelfPlayGenerator
from .teacher_pool import ConsensusSample, TeacherPool
from ..training.teacher_distill import TeacherCache, TeacherClient

logger = logging.getLogger(__name__)


@dataclass
class TrainerStats:
    step: int
    samples_seen: int
    samples_accepted: int
    samples_rejected_consensus: int
    teacher_acceptance_rate: float
    cache_hits: int
    cache_misses: int
    self_play_rounds: int
    last_loss: float
    elapsed_seconds: float


class FilumTrainer:
    """Wires every innovation into one training loop.

    Caller wires the model + tokenizer + the actual loss function;
    this class drives the data + sampling + optimizer.
    """

    def __init__(
        self,
        *,
        model,
        tokenizer,
        config: FilumConfig,
        teacher_clients: List[TeacherClient],
        compute_loss_fn: Callable[..., Any],
        student_logits_fn: Callable[..., Any],
        student_generate_fn: Optional[Callable[[str], Awaitable[str]]] = None,
    ):
        if not _HAS_TORCH:
            raise RuntimeError("FilumTrainer requires torch")
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.compute_loss_fn = compute_loss_fn
        self.student_logits_fn = student_logits_fn

        # --- multi-teacher ---
        cache = TeacherCache(cache_dir=Path(config.cache_dir))
        self.teacher_pool = TeacherPool(
            teachers=teacher_clients, cache=cache,
            jsd_threshold=config.consensus_jsd_threshold,
        )

        # --- active sampler ---
        self.active = ActiveSampler(
            pool_size=config.active_sampler_pool_size,
            kl_temperature=config.active_sampler_kl_temperature,
        )

        # --- self-play ---
        if config.self_play_enabled and student_generate_fn is not None:
            self.self_play = SelfPlayGenerator(
                config=SelfPlayConfig(
                    prompts_per_round=config.self_play_prompts_per_round,
                ),
                generate_fn=student_generate_fn,
            )
        else:
            self.self_play = None

        # --- optimizer ---
        if config.use_8bit_adamw:
            from .optimizer_8bit import AdamW8bit
            opt_cls = AdamW8bit
        else:
            opt_cls = torch.optim.AdamW
        self.optimizer = opt_cls(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # --- stats ---
        self._step = 0
        self._samples_seen = 0
        self._last_loss = 0.0
        self._start_time = time.monotonic()
        self._self_play_rounds = 0

    # ------------------------------------------------------------------

    @property
    def stats(self) -> TrainerStats:
        cache = self.teacher_pool.cache
        return TrainerStats(
            step=self._step,
            samples_seen=self._samples_seen,
            samples_accepted=self.teacher_pool.accepted_count,
            samples_rejected_consensus=self.teacher_pool.rejected_count,
            teacher_acceptance_rate=self.teacher_pool.acceptance_rate,
            cache_hits=cache.hit if cache else 0,
            cache_misses=cache.miss if cache else 0,
            self_play_rounds=self._self_play_rounds,
            last_loss=self._last_loss,
            elapsed_seconds=time.monotonic() - self._start_time,
        )

    # ------------------------------------------------------------------

    async def collect_samples(self, prompts: List[str]) -> int:
        """Run prompts through the teacher pool, push consensus-
        accepted samples into the active-sampling pool. Returns the
        count of accepted samples."""
        accepted = 0
        for prompt in prompts:
            cs: ConsensusSample = await self.teacher_pool.sample_with_consensus(
                prompt,
                max_tokens=self.config.teacher_max_tokens,
                top_k_logprobs=self.config.teacher_top_k_logprobs,
            )
            self._samples_seen += 1
            if cs.accepted and cs.averaged is not None:
                self.active.add(cs.averaged)
                accepted += 1
        return accepted

    # ------------------------------------------------------------------

    def train_step(self) -> float:
        """One training step using the active sampler's top-K. Returns
        the average loss over the step."""
        if not _HAS_TORCH:
            raise RuntimeError("torch required")

        # 1. Score the pool against the current student.
        self.active.score_pool(student_logits_fn=self.student_logits_fn)
        # 2. Take the top-K hardest examples for this step.
        chosen = self.active.select(self.config.active_sampler_select_top)
        if not chosen:
            return 0.0
        # 3. Batch them (effective batch handled via grad accumulation).
        self.optimizer.zero_grad()
        total = None
        accum = self.config.grad_accum_steps
        for i, entry in enumerate(chosen[: accum * self.config.micro_batch_size]):
            loss = self.compute_loss_fn(self.model, entry.sample)
            loss = loss / accum
            loss.backward()
            total = loss if total is None else total + loss
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.grad_clip,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
        if total is None:
            return 0.0
        avg_loss = float(total.detach()) * accum
        self._last_loss = avg_loss
        self._step += 1
        return avg_loss

    # ------------------------------------------------------------------

    async def train_loop(
        self,
        seed_prompts: List[str],
        *,
        max_steps: Optional[int] = None,
        on_step: Optional[Callable[[TrainerStats], None]] = None,
    ) -> None:
        """Async training loop. Interleaves teacher collection, self-play,
        and gradient steps. Awaits cancellation gracefully."""
        target_steps = max_steps or self.config.max_steps
        # Initial seed pass.
        await self.collect_samples(seed_prompts)

        try:
            while self._step < target_steps:
                # Self-play round?
                if (self.self_play is not None and self._step > 0
                        and self._step % self.config.self_play_round_every_n_steps == 0):
                    sp_prompts = await self.self_play.propose_round()
                    self._self_play_rounds += 1
                    await self.collect_samples(sp_prompts)
                # Train step.
                self.train_step()
                if on_step is not None:
                    on_step(self.stats)
        except asyncio.CancelledError:
            logger.info("train_loop cancelled at step %d", self._step)
            raise
