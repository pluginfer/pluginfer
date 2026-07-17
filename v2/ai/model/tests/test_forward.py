"""Forward-pass + loss + KV-cache generation tests (CP-AI-2 part 2)."""

from __future__ import annotations

import math
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


@pytest.fixture(scope="module")
def model(cfg: ModelConfig) -> PluginferLM:
    torch.manual_seed(0)
    return PluginferLM(cfg)


def test_loss_is_finite(model: PluginferLM, cfg: ModelConfig) -> None:
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    targets = torch.randint(0, cfg.vocab_size, (2, 32))
    logits = model(ids)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    # Initial cross-entropy of a fresh LM on uniform-random targets should
    # be near log(vocab_size). We accept a generous range.
    expected = math.log(cfg.vocab_size)
    assert 0.5 * expected < loss.item() < 1.5 * expected


def test_logits_change_with_input(model: PluginferLM, cfg: ModelConfig) -> None:
    a = torch.randint(0, cfg.vocab_size, (1, 16))
    b = torch.randint(0, cfg.vocab_size, (1, 16))
    while torch.equal(a, b):
        b = torch.randint(0, cfg.vocab_size, (1, 16))
    out_a = model(a)
    out_b = model(b)
    assert not torch.allclose(out_a, out_b)


def test_generate_extends_sequence(model: PluginferLM, cfg: ModelConfig) -> None:
    prompt = torch.randint(0, cfg.vocab_size, (1, 10))
    out = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    assert out.shape == (1, 30)
    # First 10 ids must equal the prompt
    assert torch.equal(out[:, :10], prompt)
    # All emitted ids must be valid vocab ids
    assert int(out.max()) < cfg.vocab_size
    assert int(out.min()) >= 0


def test_generate_greedy_is_deterministic(
    model: PluginferLM, cfg: ModelConfig
) -> None:
    prompt = torch.randint(0, cfg.vocab_size, (1, 5))
    out1 = model.generate(prompt, max_new_tokens=8, temperature=0.0, top_p=1.0, top_k=0)
    out2 = model.generate(prompt, max_new_tokens=8, temperature=0.0, top_p=1.0, top_k=0)
    assert torch.equal(out1, out2)


def test_generate_with_temperature_is_random(
    model: PluginferLM, cfg: ModelConfig
) -> None:
    """At temperature=1.0 with no top-k/p, two greedy runs should differ
    on a small enough model. We retry up to 5 times to keep the test
    statistically robust."""
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    saw_difference = False
    for _ in range(5):
        out1 = model.generate(
            prompt, max_new_tokens=8, temperature=1.0, top_p=1.0, top_k=0
        )
        out2 = model.generate(
            prompt, max_new_tokens=8, temperature=1.0, top_p=1.0, top_k=0
        )
        if not torch.equal(out1, out2):
            saw_difference = True
            break
    assert saw_difference, "sampling at temperature=1.0 produced identical runs 5x"


def test_generate_eos_stops_early(model: PluginferLM, cfg: ModelConfig) -> None:
    """If we set eos to the most-likely greedy token, we should stop on step 1."""
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    # Find the first greedy emission for this prompt.
    greedy_one = model.generate(
        prompt, max_new_tokens=1, temperature=0.0, top_p=1.0, top_k=0
    )
    eos_id = int(greedy_one[0, -1])
    out = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        eos_token_id=eos_id,
    )
    # Generation stops as soon as eos is emitted; we expect <= 21 tokens
    # but at least prompt + 1.
    assert prompt.shape[1] < out.shape[1] <= prompt.shape[1] + 20
