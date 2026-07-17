"""Gradient-flow tests (CP-AI-2 part 3).

Asserts that every trainable parameter in the debug-config model
receives a non-NaN gradient on a plausible cross-entropy loss.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from ai.model.config import ModelConfig  # noqa: E402
from ai.model.transformer import PluginferLM  # noqa: E402


@pytest.fixture(scope="module")
def cfg() -> ModelConfig:
    return ModelConfig.debug()


def test_every_parameter_gets_a_finite_gradient(cfg: ModelConfig) -> None:
    torch.manual_seed(1)
    model = PluginferLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    targets = torch.randint(0, cfg.vocab_size, (2, 32))
    logits = model(ids)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()

    missing: list[str] = []
    nan_params: list[str] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            missing.append(name)
            continue
        if not torch.isfinite(p.grad).all():
            nan_params.append(name)
    assert not missing, f"params with no gradient: {missing}"
    assert not nan_params, f"params with NaN/Inf gradient: {nan_params}"


def test_gradient_flows_into_embedding_via_lm_head(cfg: ModelConfig) -> None:
    """Weight tying means the embedding gets gradients from BOTH the
    lookup AND the output projection. Check the magnitude is non-trivial."""
    torch.manual_seed(2)
    model = PluginferLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(ids)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()
    embed_grad = model.embed.weight.grad
    assert embed_grad is not None
    assert torch.isfinite(embed_grad).all()
    assert embed_grad.abs().sum().item() > 0.0


def test_one_step_of_sgd_moves_loss_down(cfg: ModelConfig) -> None:
    """Sanity: 1 SGD step on the same batch should reduce the loss.

    Doesn't test convergence; just that gradients have the right sign
    relative to the loss function (i.e. forward+backward+step are wired
    correctly)."""
    torch.manual_seed(3)
    model = PluginferLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (4, 32))
    targets = torch.randint(0, cfg.vocab_size, (4, 32))

    def step_loss() -> float:
        logits = model(ids)
        return F.cross_entropy(
            logits.view(-1, cfg.vocab_size), targets.view(-1)
        ).item()

    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    pre = step_loss()
    optim.zero_grad()
    logits = model(ids)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()
    optim.step()
    post = step_loss()
    assert post < pre, f"loss did not decrease: pre={pre:.4f} post={post:.4f}"
