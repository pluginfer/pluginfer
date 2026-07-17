"""Adaptive low-rank gradient projector.

Standard GaLore (Zhao et al., ICML 2024) projects each 2-D gradient
matrix ``G`` (shape ``[m, n]``) into a rank-``r`` subspace by ``G_low
= P^T G`` where ``P`` is the top-r left singular vectors of ``G``,
recomputed every K steps. Optimizer state lives in the projected
space (size ``[r, n]``), then is projected back at update time. Memory
saving: ``r/m`` of the optimizer footprint, with negligible loss.

GaLore as published is *rank-fixed*: you pick ``r`` once, e.g. r=128
for a 7B model, and live with it. That's a problem on consumer
hardware where the *available* memory budget changes minute-to-minute
(browser opens, video plays, OS compositor wakes up).

novel claim B2 (see the design notes): a method of training in
which the rank ``r`` of a low-rank gradient projection is selected
*per training step* as a monotonic function of an observed hardware
pressure scalar, with hysteresis to prevent thrash, and re-projection
of optimizer state when the rank changes by more than a threshold.

This file ships a CPU/GPU-agnostic implementation. ``torch`` is
imported lazily so the test suite can import the module without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
    _HAS_TORCH = True
except Exception:                                                  # pragma: no cover
    torch = None
    _HAS_TORCH = False


@dataclass(frozen=True)
class RankPolicy:
    """How rank scales with pressure P. Linear interpolation in log-2 of r."""
    r_min: int = 8       # heavy throttle: ~8x memory savings vs full
    r_max: int = 256     # idle: near-full quality
    p_lo: float = 0.30   # below this: use r_max
    p_hi: float = 0.85   # above this: use r_min
    hysteresis: float = 0.05  # ignore rank changes < this fractional delta


def choose_rank(p: float, policy: RankPolicy = RankPolicy()) -> int:
    """Map pressure P -> rank r. Monotonic non-increasing in P."""
    if p <= policy.p_lo:
        return policy.r_max
    if p >= policy.p_hi:
        return policy.r_min
    # Linear in log-2 so we move geometrically across rank doublings.
    import math
    frac = (p - policy.p_lo) / (policy.p_hi - policy.p_lo)
    log2_r = (1.0 - frac) * math.log2(policy.r_max) + frac * math.log2(policy.r_min)
    r = int(round(2 ** log2_r))
    return max(policy.r_min, min(policy.r_max, r))


class AdaptiveLowRankProjector:
    """Per-parameter low-rank projector for 2-D weight gradients.

    Usage::

        proj = AdaptiveLowRankProjector(policy=RankPolicy())
        ...
        for step in range(N):
            P = sampler.pressure()
            for name, param in model.named_parameters():
                if param.grad is None or param.grad.dim() != 2:
                    continue
                low_g = proj.project(name, param.grad, pressure=P)
                # ... apply optimizer in low-rank space ...
                full_g = proj.unproject(name, low_g)
                param.grad.copy_(full_g)

    Re-uses the SVD basis as long as the rank doesn't shift by more
    than the hysteresis band. The basis itself is also refreshed
    every ``refresh_steps`` to track the gradient's drift -- this is
    standard GaLore behaviour.

    Tensors with rank!=2 (biases, norms) bypass the projector.
    """

    def __init__(
        self,
        policy: RankPolicy = RankPolicy(),
        refresh_steps: int = 200,
    ):
        self.policy = policy
        self.refresh_steps = max(1, refresh_steps)
        # state per param-name: (P_basis, current_r, last_refresh_step)
        self._state: dict[str, tuple] = {}
        self._step = 0

    def step(self) -> None:
        """Advance internal step counter; call once per training step."""
        self._step += 1

    def reset(self) -> None:
        self._state.clear()
        self._step = 0

    def current_rank_for(self, name: str) -> Optional[int]:
        st = self._state.get(name)
        return None if st is None else int(st[1])

    # --- core projection ----------------------------------------------------

    def _need_refresh(self, name: str, target_r: int) -> bool:
        st = self._state.get(name)
        if st is None:
            return True
        _, cur_r, last_refresh = st
        if abs(cur_r - target_r) / max(cur_r, 1) > self.policy.hysteresis:
            return True
        if self._step - last_refresh >= self.refresh_steps:
            return True
        return False

    def _compute_basis(self, name: str, grad, target_r: int):
        """SVD of ``grad`` -> top-r left singular vectors. ``grad`` is 2-D."""
        if not _HAS_TORCH:
            raise RuntimeError("torch required for projection")
        m, n = grad.shape
        r = max(1, min(target_r, min(m, n)))
        # Use truncated SVD via lowrank for efficiency on big matrices.
        # For small matrices fall back to full SVD which is more stable.
        if min(m, n) <= 64:
            U, _, _ = torch.linalg.svd(grad.float(), full_matrices=False)
            P_basis = U[:, :r]
        else:
            try:
                U, _, _ = torch.svd_lowrank(grad.float(), q=r + 4)
                P_basis = U[:, :r]
            except Exception:
                U, _, _ = torch.linalg.svd(grad.float(), full_matrices=False)
                P_basis = U[:, :r]
        # Keep basis in fp32 for numerical stability of project/unproject.
        P_basis = P_basis.contiguous().detach()
        self._state[name] = (P_basis, r, self._step)
        return P_basis, r

    def project(self, name: str, grad, *, pressure: float):
        """Return rank-r projected gradient ``[r, n]`` for a 2-D input.

        Non-2-D inputs are returned unchanged (caller should treat as
        full-rank, no compression).
        """
        if not _HAS_TORCH:
            raise RuntimeError("torch required for projection")
        if grad.dim() != 2:
            return grad
        target_r = choose_rank(pressure, self.policy)
        if self._need_refresh(name, target_r):
            P_basis, r = self._compute_basis(name, grad, target_r)
        else:
            P_basis, r, _ = self._state[name]
        # low_g = P^T @ grad   shape [r, n]
        return P_basis.t() @ grad.float()

    def unproject(self, name: str, low_g):
        """Reconstruct full-shape gradient from a rank-r tensor.

        Returns ``P_basis @ low_g`` in the original dtype.
        """
        if not _HAS_TORCH:
            raise RuntimeError("torch required for projection")
        st = self._state.get(name)
        if st is None or low_g.dim() != 2:
            return low_g
        P_basis, _, _ = st
        return (P_basis @ low_g).to(low_g.dtype)

    def memory_savings_estimate(self, model_2d_param_count: int) -> float:
        """Rough estimate of optimizer-state memory ratio vs. full Adam.

        Optimizer keeps two moments per param. With projection we only
        keep moments for the projected r-dimensional view.
        """
        # Mean current r across tracked params, weighted equally.
        if not self._state:
            return 1.0
        mean_r = sum(int(s[1]) for s in self._state.values()) / len(self._state)
        # Heuristic: assume m ~ sqrt(model_2d_param_count) per matrix
        # so the saving fraction is r / m on each. This is a coarse
        # upper bound -- the real saving is per-matrix.
        import math
        m = max(1, int(math.sqrt(max(1, model_2d_param_count))))
        return min(1.0, mean_r / m)
