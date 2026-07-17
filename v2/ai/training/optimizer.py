"""AdamW + Cosine LR scheduler from scratch.

AdamW reference: Loshchilov & Hutter 2019 (arXiv:1711.05101).

Why from scratch when torch.optim.AdamW exists?
  Because the directive is "every gradient update is code you write and own."
  This implementation is literally the AdamW update; replacing it with
  torch's would be functionally identical for our purposes but breaks the
  audit trail.

The scheduler is the standard cosine-with-warmup used by Llama / GPT-3:
  - linear warmup from 0 -> max_lr over `warmup_steps`
  - cosine decay max_lr -> min_lr over the remaining max_steps
  - min_lr defaults to 0.1 * max_lr (don't decay all the way to zero)
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """AdamW (decoupled-weight-decay Adam).

    Update rule (per parameter):
        # 1. apply weight decay BEFORE the gradient step
        p <- p * (1 - lr * weight_decay)
        # 2. update first/second moments
        m <- beta1 * m + (1 - beta1) * grad
        v <- beta2 * v + (1 - beta2) * grad**2
        # 3. bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        # 4. parameter update
        p <- p - lr * m_hat / (sqrt(v_hat) + eps)

    Note that step 1 (decoupled weight decay) is what distinguishes AdamW
    from the L2-regularised Adam used in earlier transformer papers; the
    distinction matters a lot for transformer pretraining stability.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ) -> None:
        if lr < 0:
            raise ValueError(f"lr must be >= 0; got {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas must be in [0, 1); got {betas}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0; got {eps}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0; got {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # Decoupled weight decay (BEFORE the moment update)
                if wd != 0.0:
                    p.mul_(1 - lr * wd)

                # First moment: m <- beta1 * m + (1 - beta1) * grad
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Second moment: v <- beta2 * v + (1 - beta2) * grad ** 2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class CosineSchedulerWithWarmup:
    """Linear warmup + cosine decay scheduler.

    Usage:
        sched = CosineSchedulerWithWarmup(opt, max_lr=3e-4,
                                          warmup_steps=100, max_steps=10000)
        for step in range(max_steps):
            ... train ...
            sched.step(step)

    `current_lr` is exposed for logging.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0; got {warmup_steps}")
        if max_steps <= warmup_steps:
            raise ValueError(
                f"max_steps ({max_steps}) must be > warmup_steps ({warmup_steps})"
            )
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0, 1]; got {min_lr_ratio}")
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = max_lr * min_lr_ratio
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.current_lr: float = 0.0

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup. Start at lr=0 at step 0; reach max_lr at step
            # warmup_steps - 1.
            return self.max_lr * (step + 1) / max(1, self.warmup_steps)
        if step >= self.max_steps:
            return self.min_lr
        # Cosine decay
        progress = (step - self.warmup_steps) / max(
            1, self.max_steps - self.warmup_steps
        )
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * cos

    def step(self, step: int) -> float:
        lr = self.lr_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.current_lr = lr
        return lr
