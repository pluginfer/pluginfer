"""CP-AI-4 gating tests: 100-step CPU training run + checkpoint resume."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from ai.data.dataset import PluginferDataset, TaskMix  # noqa: E402
from ai.model.config import ModelConfig  # noqa: E402
from ai.model.transformer import PluginferLM  # noqa: E402
from ai.tokenizer.bpe import BASE_VOCAB_SIZE  # noqa: E402
from ai.tokenizer.tokenizer import PluginferTokenizer  # noqa: E402
from ai.tokenizer.vocab_builder import CorpusBuilder  # noqa: E402
from ai.training.checkpointing import load_checkpoint  # noqa: E402
from ai.training.optimizer import AdamW, CosineSchedulerWithWarmup  # noqa: E402
from ai.training.trainer import Trainer, TrainingConfig  # noqa: E402
from ai.training.distributed import (  # noqa: E402
    DDPNotImplementedError,
    init_process_group,
    wrap_ddp,
)
from ai.training.mesh_trainer import MeshTrainer  # noqa: E402


# ---------------------------------------------------------------------------
# AdamW + CosineSchedulerWithWarmup unit tests
# ---------------------------------------------------------------------------

def test_adamw_minimises_quadratic_in_few_steps() -> None:
    """f(x) = x.dot(x); minimum at zero. AdamW should drive x toward 0."""
    torch.manual_seed(0)
    x = torch.nn.Parameter(torch.randn(8))
    opt = AdamW([x], lr=0.1, weight_decay=0.0)
    initial = x.detach().pow(2).sum().item()
    for _ in range(200):
        opt.zero_grad()
        loss = x.pow(2).sum()
        loss.backward()
        opt.step()
    final = x.detach().pow(2).sum().item()
    assert final < initial * 0.01


def test_adamw_decoupled_weight_decay_shrinks_param() -> None:
    """With weight_decay > 0 and zero gradient, AdamW should shrink the param."""
    x = torch.nn.Parameter(torch.ones(4))
    opt = AdamW([x], lr=0.1, weight_decay=0.5)
    # Zero gradient -> only weight decay applies
    for _ in range(5):
        opt.zero_grad()
        loss = (x * 0).sum()  # no real loss; produce zero grad
        loss.backward()
        opt.step()
    # After 5 steps of (1 - 0.05) multiplicative shrinkage = 0.95^5 ~= 0.774
    assert (x.detach() < 1.0).all()
    assert (x.detach() > 0.7).all()


def test_cosine_scheduler_warmup_then_decay() -> None:
    p = torch.nn.Parameter(torch.zeros(1))
    opt = AdamW([p], lr=1e-3)
    sched = CosineSchedulerWithWarmup(
        opt, max_lr=1e-3, warmup_steps=10, max_steps=100, min_lr_ratio=0.1
    )
    lr_at = [sched.lr_at(s) for s in range(101)]
    assert lr_at[0] < lr_at[5] < lr_at[10]  # warmup increasing
    assert abs(lr_at[10] - 1e-3) < 1e-9 or lr_at[10] <= 1e-3  # peak around 10
    assert lr_at[100] <= 1e-4 + 1e-9  # reached min
    # Cosine is monotone decreasing during decay phase
    decay = lr_at[10:101]
    assert all(decay[i] >= decay[i + 1] - 1e-12 for i in range(len(decay) - 1))


# ---------------------------------------------------------------------------
# 100-step CPU training run (CP-AI-4 gate)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer() -> PluginferTokenizer:
    builder = CorpusBuilder(seed=2026)
    corpus = builder.build_all(n_jobs=400, n_provider=200, n_auction=200)
    return PluginferTokenizer.train_new(corpus, vocab_size=BASE_VOCAB_SIZE + 800)


@pytest.fixture(scope="module")
def train_loader(tokenizer: PluginferTokenizer) -> DataLoader:
    ds = PluginferDataset(
        tokenizer,
        seed=11,
        mix=TaskMix(job_router=400, provider_quality=200),
        context_length=64,
    )
    return DataLoader(ds, batch_size=4, shuffle=True)


@pytest.fixture(scope="module")
def val_loader(tokenizer: PluginferTokenizer) -> DataLoader:
    ds = PluginferDataset(
        tokenizer,
        seed=22,
        mix=TaskMix(job_router=80, provider_quality=40),
        context_length=64,
    )
    return DataLoader(ds, batch_size=4, shuffle=False)


@pytest.fixture(scope="module")
def model_cfg(tokenizer: PluginferTokenizer) -> ModelConfig:
    # Sized to comfortably exceed the tokenizer vocab while staying tiny.
    vocab = max(tokenizer.vocab_size + 4, 1024)
    return ModelConfig(
        vocab_size=vocab,
        context_length=64,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        d_ff=256,
    )


def test_trainer_100_steps_on_cpu_loss_decreases(
    train_loader: DataLoader, val_loader: DataLoader, model_cfg: ModelConfig
) -> None:
    torch.manual_seed(7)
    model = PluginferLM(model_cfg)
    cfg = TrainingConfig(
        max_steps=100,
        eval_every=50,
        log_every=10,
        warmup_steps=10,
        max_lr=1e-3,
        weight_decay=0.0,
        grad_clip_norm=1.0,
        device="cpu",
        amp_dtype="none",
        seed=7,
    )
    trainer = Trainer(model, cfg)
    metrics = trainer.train(train_loader, val_loader)

    assert metrics.n_steps == 100, f"expected 100 steps, got {metrics.n_steps}"
    assert math.isfinite(metrics.final_loss)
    assert math.isfinite(metrics.initial_loss)
    assert (
        metrics.final_loss < metrics.initial_loss
    ), f"loss did not decrease: initial={metrics.initial_loss:.3f} final={metrics.final_loss:.3f}"
    assert metrics.final_ppl < metrics.initial_ppl
    assert metrics.max_grad_norm < 100.0  # bounded by grad clip + reasonable scale
    # No NaN anywhere in the history
    for h in metrics.history:
        if "loss" in h:
            assert not math.isnan(h["loss"])
        if "eval" in h:
            assert not math.isnan(h["eval"]["loss"])


def test_checkpoint_save_and_resume_match_eval(
    train_loader: DataLoader, val_loader: DataLoader, model_cfg: ModelConfig, tmp_path: Path
) -> None:
    torch.manual_seed(13)
    model = PluginferLM(model_cfg)
    cfg = TrainingConfig(
        max_steps=20,
        eval_every=10,
        log_every=5,
        warmup_steps=4,
        max_lr=1e-3,
        weight_decay=0.0,
        grad_clip_norm=1.0,
        device="cpu",
        amp_dtype="none",
        seed=13,
    )
    trainer = Trainer(model, cfg)
    trainer.train(train_loader, val_loader)
    ckpt_path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(ckpt_path)

    # Eval loss before / after reload should match
    pre = trainer.evaluate(val_loader)["loss"]

    # Build fresh trainer from checkpoint and re-eval the same val batch
    fresh = Trainer.from_checkpoint(ckpt_path, cfg)
    post = fresh.evaluate(val_loader)["loss"]

    assert abs(pre - post) < 1e-4, f"pre={pre} post={post}"

    # The checkpoint dict round-trips its layout
    body = load_checkpoint(ckpt_path)
    assert body["global_step"] == 20
    assert "model_state_dict" in body
    assert "optimizer_state_dict" in body


# ---------------------------------------------------------------------------
# Distributed / Mesh stubs are honest NotImplementedError
# ---------------------------------------------------------------------------

def test_ddp_wrapper_is_honest_stub() -> None:
    with pytest.raises(DDPNotImplementedError):
        wrap_ddp(None, world_size=2, rank=0)
    with pytest.raises(DDPNotImplementedError):
        init_process_group(backend="gloo")


def test_mesh_trainer_is_honest_stub() -> None:
    with pytest.raises(NotImplementedError):
        MeshTrainer()
