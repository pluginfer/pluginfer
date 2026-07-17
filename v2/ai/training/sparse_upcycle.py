"""Sparse Upcycling: convert dense MLPs to Mixture-of-Experts.

Why
---
A dense FFN inside the transformer block does:
    h = SwiGLU(x @ W1) @ W2          (cost: O(d * d_ff) per token)

In a 1.13B-param model with d_model=2048 and d_ff=8192, that's
~16M FLOPs per token per layer. With 24 layers, ~400M FLOPs/token.
At 1024-token sequences, ~400 GFLOPs/sequence/forward. Train on a
single 6 GB GeForce: ~1 step / 200 ms, or ~5 steps/sec.

A Mixture-of-Experts replaces each FFN with N "expert" FFNs and a
small router. Each token activates only top-k experts (typically
k=2 of N=8). Per-token cost drops to (k/N) * dense cost = 1/4×.
We get the parameter benefit (more total knowledge stored) at a
fraction of the compute.

The KEY trick (Komatsuzaki et al., "Sparse Upcycling", 2023) is:
DON'T train MoE from scratch. Train a dense model partially, then
clone each FFN into N copies as the initial expert set. The router
is small + initialized to uniform. This lets the cloned experts
DIVERGE during continued training rather than starting cold.

Practical recipe for the 1.13B target:
  1. Train dense for ~50% of total budget on debug-scale (10M -> 200M).
  2. Upcycle each FFN: 8 experts, top-2 routing.
  3. Train the now-MoE 1.13B for the remaining budget at ~3x the
     tokens-per-second.

What's in here
--------------
* MoEFFN: N-expert SwiGLU + a top-k router with load-balancing aux
  loss (so all experts get used, not just the first one).
* upcycle_dense_to_moe(model): walk the model, replace every dense
  FFN with an MoEFFN initialized from N clones of the dense weights.

Failure modes (honest)
----------------------
* Without the load-balancing loss, the router collapses (one expert
  gets all tokens, others starve). Aux loss coefficient = 0.01 is the
  standard.
* Top-1 routing converges faster initially but generalises worse;
  top-2 is the safer default.
* MoE introduces irregular memory access -- on consumer GPUs without
  an optimised expert dispatch kernel (Tutel / FasterMoE / Megablocks)
  the speedup is closer to 1.5x than 3x. We ship a dense-batched
  fallback that's correct but unoptimized.
"""

from __future__ import annotations

from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
    _BASE_MODULE = nn.Module
except Exception:                                                # pragma: no cover
    torch = None
    nn = None
    F = None
    _HAS_TORCH = False
    _BASE_MODULE = object


def _swiglu(x_gate, x_up):
    return F.silu(x_gate) * x_up


class _Expert(_BASE_MODULE):
    """A single SwiGLU expert FFN. Equivalent to the existing dense
    FFN in `ai/model/ffn.py` -- we re-implement here so the module
    is self-contained for upcycling."""
    def __init__(self, d_model: int, d_ff: int):
        if not _HAS_TORCH:
            raise RuntimeError("torch required")
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(_swiglu(self.gate(x), self.up(x)))


class MoEFFN(_BASE_MODULE):
    """N-expert SwiGLU FFN with top-k routing.

    Each token: router computes logits over N experts, picks top-k,
    weights by softmax over the chosen logits, sums the experts'
    outputs. Falls back to a slow dense-batched path if no fast
    dispatch kernel is wired -- correctness is preserved either way.
    """

    def __init__(self, d_model: int, d_ff: int, *,
                 num_experts: int = 8, top_k: int = 2,
                 aux_loss_coeff: float = 0.01):
        if not _HAS_TORCH:
            raise RuntimeError("torch required")
        super().__init__()
        if not (1 <= top_k <= num_experts):
            raise ValueError(f"top_k {top_k} not in [1, {num_experts}]")
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coeff = aux_loss_coeff

        # Router. Tiny; <0.1% of total params.
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Experts.
        self.experts = nn.ModuleList(
            [_Expert(d_model, d_ff) for _ in range(num_experts)]
        )

        # Last-batch aux loss. The Trainer reads this and adds it to
        # the main loss before backward.
        self.last_aux_loss: Optional["torch.Tensor"] = None

    def forward(self, x):
        # x: (B, T, d_model). Treat (B*T) as the token batch.
        bsz, seq_len, _ = x.shape
        x_flat = x.view(-1, self.d_model)
        n_tokens = x_flat.size(0)

        # Router logits + softmax over experts per token.
        router_logits = self.router(x_flat)                 # (n_tokens, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k experts per token.
        top_k_vals, top_k_idx = router_probs.topk(self.top_k, dim=-1)
        # Renormalize the top-k probs so they sum to 1 per token (this
        # is the standard MoE convention: a token's contributions
        # weight by the relative probability among the chosen experts).
        top_k_vals = top_k_vals / top_k_vals.sum(dim=-1, keepdim=True).clamp_min_(1e-8)

        # Dense-batched dispatch (correct but unoptimized). For each
        # expert, gather the tokens routed to it, run, scatter back.
        out = torch.zeros_like(x_flat)
        for expert_id in range(self.num_experts):
            # Mask of tokens routing to this expert (in any of their k slots).
            mask = (top_k_idx == expert_id)                 # (n_tokens, top_k)
            if not mask.any():
                continue
            # Per-token weight for this expert (sum across the slots
            # in case the same expert is in multiple top-k positions
            # -- usually only at most one).
            weight = (top_k_vals * mask.float()).sum(dim=-1)  # (n_tokens,)
            # Run only on tokens that actually use this expert (any nonzero weight).
            sel = weight > 0
            if not sel.any():
                continue
            xs = x_flat[sel]
            ys = self.experts[expert_id](xs)
            out[sel] += ys * weight[sel].unsqueeze(-1)

        # Load-balancing auxiliary loss (Switch Transformer / GShard).
        # Forces the router to spread tokens evenly across experts.
        # frac_tokens_per_expert: empirical token fraction routed to e.
        # router_prob_per_expert: average router prob assigned to e.
        # aux_loss = N * sum_e(frac * prob)
        # Minimised when both are 1/N (uniform).
        with torch.no_grad():
            ones = torch.ones_like(top_k_idx, dtype=x.dtype)
            tokens_per_expert = torch.zeros(
                self.num_experts, device=x.device, dtype=x.dtype,
            )
            tokens_per_expert.scatter_add_(
                0, top_k_idx.flatten(), ones.flatten() / self.top_k,
            )
            frac = tokens_per_expert / max(1, n_tokens)
        prob_per_expert = router_probs.mean(dim=0)
        self.last_aux_loss = (
            self.num_experts * (frac * prob_per_expert).sum()
            * self.aux_loss_coeff
        )

        return out.view(bsz, seq_len, self.d_model)


# ---------------------------------------------------------------------------
# Upcycling: dense FFN -> MoE
# ---------------------------------------------------------------------------


def upcycle_module(dense_ffn, *, num_experts: int = 8, top_k: int = 2,
                   noise_std: float = 0.01):
    """Replace `dense_ffn` (a SwiGLU FFN with .gate / .up / .down
    Linears) with an MoEFFN whose experts start as cloned copies of
    the dense weights + small Gaussian noise to break symmetry.

    Returns the new MoEFFN. Caller is responsible for setattr-ing it
    onto the parent module. See `upcycle_dense_to_moe` for the full
    walk."""
    if not _HAS_TORCH:
        raise RuntimeError("torch required")

    # Detect d_model / d_ff from the dense layer.
    if hasattr(dense_ffn, "gate") and hasattr(dense_ffn, "up") and hasattr(dense_ffn, "down"):
        gate, up, down = dense_ffn.gate, dense_ffn.up, dense_ffn.down
    else:
        # Other FFN layouts can be added here as encountered.
        raise ValueError(
            "expected SwiGLU FFN with .gate, .up, .down Linears; "
            f"got {type(dense_ffn).__name__}"
        )
    d_model = gate.in_features
    d_ff = gate.out_features

    moe = MoEFFN(d_model=d_model, d_ff=d_ff,
                 num_experts=num_experts, top_k=top_k)

    with torch.no_grad():
        for expert in moe.experts:
            # Copy the dense weights into each expert.
            expert.gate.weight.copy_(gate.weight)
            expert.up.weight.copy_(up.weight)
            expert.down.weight.copy_(down.weight)
            # Add a small Gaussian to break symmetry so they diverge
            # during continued training (otherwise they stay tied
            # forever and the aux loss can't help).
            if noise_std > 0:
                expert.gate.weight.add_(torch.randn_like(expert.gate.weight) * noise_std)
                expert.up.weight.add_(torch.randn_like(expert.up.weight) * noise_std)
                expert.down.weight.add_(torch.randn_like(expert.down.weight) * noise_std)
        # Initialize the router to uniform (small Gaussian; sigmoid
        # would collapse to 1/N anyway, but rng is needed so that
        # gradient differentiates the experts).
        nn.init.normal_(moe.router.weight, std=0.02)

    return moe


def upcycle_dense_to_moe(model, *, num_experts: int = 8, top_k: int = 2,
                         ffn_match_fn=None):
    """Walk `model`, replace every dense SwiGLU FFN with an MoEFFN
    initialized from cloned weights.

    `ffn_match_fn(module) -> bool` lets the caller customize which
    modules get replaced (default: any module exposing .gate, .up,
    .down). Returns (modified_in_place_model, stats_dict)."""
    if not _HAS_TORCH:
        raise RuntimeError("torch required")
    if ffn_match_fn is None:
        def ffn_match_fn(m):
            return (hasattr(m, "gate") and hasattr(m, "up") and hasattr(m, "down")
                    and isinstance(getattr(m, "gate", None), nn.Linear))

    converted = 0
    targets = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if ffn_match_fn(child) and not isinstance(child, MoEFFN):
                targets.append((parent, child_name, child))

    for parent, child_name, child in targets:
        new = upcycle_module(child, num_experts=num_experts, top_k=top_k)
        setattr(parent, child_name, new)
        converted += 1

    return model, {
        "converted_ffns": converted,
        "num_experts": num_experts,
        "top_k": top_k,
        "active_param_ratio": top_k / num_experts,
    }


def collect_aux_losses(model) -> "torch.Tensor":
    """Sum the load-balancing aux losses recorded by every MoEFFN in
    `model` after a forward pass. The trainer adds this to the main
    loss before backward."""
    if not _HAS_TORCH:
        raise RuntimeError("torch required")
    total = None
    for module in model.modules():
        if isinstance(module, MoEFFN) and module.last_aux_loss is not None:
            total = module.last_aux_loss if total is None else total + module.last_aux_loss
    if total is None:
        return torch.tensor(0.0)
    return total
