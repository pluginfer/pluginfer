"""Filum real training entry point.

Bridges `python -m ai.filum train --max-steps N` to a working
distillation loop using whichever teacher API keys are present in the
environment. Mirrors `demo_train.py`'s shape but uses real teachers
(Gemini / Anthropic / OpenAI) and a configurable model size.

Day-1 of the launch playbook expects this to actually train. The path
to 127M-param production training is the same code with vocab=16384
BPE + larger d_model/n_layers. For the warm-up we use byte-level
vocab=256 so we don't need a pretrained tokenizer for the first run.
The teacher's response text is byte-encoded into the student's vocab.

Usage:

    $env:GOOGLE_API_KEY = "..."
    python -m ai.filum train --max-steps 5000

The demo path (`--demo`) stays in `demo_train.py`; this is the real
path. Both share the same loss + optimizer + grad-clip wiring so
behaviour at scale is verified by the demo.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


_DEFAULT_PROMPTS: List[str] = [
    "Continue the sentence: The quick brown fox",
    "What is two plus two?",
    "Write one short sentence about Pluginfer.",
    "Explain compute auctions in one sentence.",
    "What does ECDSA stand for?",
    "Name three properties of a hash function.",
    "What is the chain rule of probability?",
    "Define a Merkle tree in one sentence.",
    "Why is replay protection important?",
    "What is a sealed-bid auction?",
    "Give a one-line definition of latency.",
    "What is gradient descent?",
    "Explain a peer-to-peer mesh in one sentence.",
    "Define a zero-knowledge proof briefly.",
    "What is a token bucket rate limiter?",
    "Why is fp16 useful for training?",
]


def _log(msg: str) -> None:
    print(f"[filum train] {msg}", flush=True)


def _build_teachers() -> list:
    """Construct teacher clients from env vars. Returns an empty list
    if no key is configured -- caller should refuse to proceed."""
    from ..training.teacher_distill import GeminiTeacher, AnthropicTeacher

    teachers = []
    if os.environ.get("GOOGLE_API_KEY"):
        try:
            teachers.append(GeminiTeacher())
            _log("teacher: Gemini (free tier, 1500 req/day)")
        except Exception as e:
            _log(f"  Gemini unavailable: {e}")
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            teachers.append(AnthropicTeacher())
            _log("teacher: Anthropic Claude Haiku")
        except Exception as e:
            _log(f"  Anthropic unavailable: {e}")
    return teachers


def _byte_encode(text: str, max_len: int) -> list:
    return list(text.encode("utf-8", errors="replace"))[:max_len]


@dataclass
class RealTrainArgs:
    max_steps: int = 5000
    device: str = "auto"          # auto | cpu | cuda
    d_model: int = 256            # warm-up scale; use 896 for 127M target
    n_layers: int = 4             # warm-up scale; use 14 for 127M target
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 64
    d_ff: int = 768
    context_length: int = 256
    vocab_size: int = 256         # byte-level for warm-up
    micro_batch_size: int = 4
    lr: float = 5e-4
    grad_clip: float = 1.0
    log_every: int = 10
    ckpt_every: int = 500
    seed_prompts_per_round: int = 16
    out_dir: str = "ai/filum/_work"
    resume: Optional[str] = None


async def run_real_train(args: RealTrainArgs) -> int:
    """Run a real distillation training loop. Returns 0 on success."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        _log("ERROR: torch not installed.")
        _log("  Run v2/ai/filum/setup_filum.ps1 to install.")
        return 2

    teachers = _build_teachers()
    if not teachers:
        _log("ERROR: no teacher API keys found.")
        _log("  Set at least one of GOOGLE_API_KEY, ANTHROPIC_API_KEY.")
        _log("  Or run the demo:  python -m ai.filum train --demo")
        return 2

    from .architecture import FilumArchConfig, FilumModel
    from .optimizer_8bit import AdamW8bit
    from .training_governor import TrainingGovernor

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    _log(f"device: {device}")
    if device == "cuda":
        _log(f"  CUDA: {torch.cuda.get_device_name(0)}")

    # Adaptive governor: detects-and-recovers, never throttles the happy path.
    # See training_governor.TrainingGovernor for the full state machine.
    governor = TrainingGovernor.start(device=device, log=_log)

    cfg = FilumArchConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        head_dim=args.head_dim,
        d_ff=args.d_ff,
        ssm_every_n_layers=999,        # disable SSM at warm-up scale
        sliding_window=min(args.context_length, 128),
        use_differential=False,        # vanilla GQA at warm-up scale
    )
    model = FilumModel(cfg).to(device)
    n_params = model.n_params()
    _log(f"model params: {n_params:,}")

    # Use stock fp32 AdamW for tiny warm-up runs (<10M params): the
    # int8 optimizer's memory savings don't matter at this scale and
    # fp32 is numerically more forgiving when the model is freshly
    # initialised. AdamW8bit only earns its keep at 50M+ params.
    if n_params < 10_000_000:
        _log("optimizer: torch.optim.AdamW (fp32, warm-up scale)")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )
    else:
        _log("optimizer: AdamW8bit (int8 state)")
        optimizer = AdamW8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=0.01,
        )

    # Linear LR warmup over the first 100 steps.
    warmup_steps = min(100, max(10, args.max_steps // 50))
    target_lr = args.lr

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if args.resume and Path(args.resume).exists():
        sd = torch.load(args.resume, map_location=device)
        model.load_state_dict(sd["state_dict"])
        start_step = int(sd.get("step", 0))
        _log(f"resumed from {args.resume} at step {start_step}")

    losses: List[float] = []
    samples_seen = 0
    t0 = time.monotonic()

    _log(f"begin training: {args.max_steps} steps")
    step = start_step
    prompt_idx = 0
    while step < args.max_steps:
        # Gather one batch of teacher responses round-robin across teachers.
        batch_inputs: list = []
        for _ in range(args.seed_prompts_per_round):
            prompt = _DEFAULT_PROMPTS[prompt_idx % len(_DEFAULT_PROMPTS)]
            prompt_idx += 1
            teacher = teachers[samples_seen % len(teachers)]
            try:
                ts = await teacher.generate(
                    prompt, max_tokens=64, top_k_logprobs=5,
                )
            except Exception as e:
                _log(f"  teacher error ({teacher.teacher_id}): {e}")
                continue
            samples_seen += 1
            ids = _byte_encode(ts.response_text, args.context_length)
            if len(ids) >= 4:
                batch_inputs.append(ids)
        if not batch_inputs:
            _log("  no usable teacher samples this round; sleeping 5s")
            await asyncio.sleep(5)
            continue

        # Train on what we collected. Each ids list is one micro-batch
        # entry; we treat the response itself as next-token target.
        for ids in batch_inputs:
            if step >= args.max_steps:
                break
            governor.tick(step)
            # Build input on CPU first, clamp, then move — keeps any
            # out-of-vocab ids from ever reaching the embedding kernel.
            x_cpu = torch.tensor(ids, dtype=torch.long).clamp_(0, cfg.vocab_size - 1)
            x = x_cpu.unsqueeze(0).to(device, non_blocking=True)
            logits = model(x[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                x[:, 1:].reshape(-1),
            )
            loss_val = float(loss.detach())
            # Skip only on non-finite. Big-but-finite losses (cold-start)
            # are handled by grad-clip + the optimizer's NaN guard.
            if not math.isfinite(loss_val):
                _log(f"  step {step+1}: loss={loss_val} non-finite — skipping")
                optimizer.zero_grad(set_to_none=True)
                step += 1
                continue
            # Linear LR warmup.
            if step < warmup_steps:
                warm = (step + 1) / warmup_steps
                for g in optimizer.param_groups:
                    g["lr"] = target_lr * warm
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip,
            )
            if not torch.isfinite(grad_norm):
                _log(f"  step {step+1}: grad-norm non-finite — skipping update")
                optimizer.zero_grad(set_to_none=True)
                step += 1
                continue
            try:
                optimizer.step()
            except RuntimeError as e:
                # CUDA OOM / illegal-address / NaN-cast — the governor
                # decides whether to retry on CPU, shrink, or skip.
                action = governor.handle_runtime_error(e, step=step)
                if action == "skip":
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    continue
                if action == "fallback_cpu":
                    device = "cpu"
                    model.to(device)
                    _log("  governor: falling back to CPU for the rest of this run")
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    continue
                raise
            losses.append(loss_val)
            step += 1
            governor.record_ok(loss_val)

            if step % args.log_every == 0 or step == 1:
                elapsed = time.monotonic() - t0
                _log(
                    f"  step {step:5d}/{args.max_steps}  "
                    f"loss={losses[-1]:.4f}  "
                    f"samples={samples_seen}  "
                    f"elapsed={elapsed:.1f}s"
                )
            if step % args.ckpt_every == 0:
                ckpt = ckpt_dir / f"filum_step{step}.pt"
                torch.save({
                    "state_dict": model.state_dict(),
                    "config": asdict(args),
                    "step": step,
                    "losses": losses,
                }, ckpt)
                _log(f"  checkpoint saved: {ckpt}")

    elapsed = time.monotonic() - t0
    _log("")
    _log("training complete.")
    _log(f"  steps        : {step}")
    _log(f"  samples seen : {samples_seen}")
    _log(f"  first loss   : {losses[0]:.4f}" if losses else "  first loss   : -")
    _log(f"  last loss    : {losses[-1]:.4f}" if losses else "  last loss    : -")
    _log(f"  elapsed      : {elapsed:.1f}s")

    final_ckpt = ckpt_dir / f"filum_step{step}_final.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": asdict(args),
        "step": step,
        "losses": losses,
        "final": True,
    }, final_ckpt)
    _log(f"  final ckpt   : {final_ckpt}")

    summary = out_dir / f"train_summary_step{step}.json"
    summary.write_text(json.dumps({
        "step": step,
        "samples_seen": samples_seen,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "elapsed_seconds": elapsed,
        "device": device,
        "config": asdict(args),
        "teacher_ids": [t.teacher_id for t in teachers],
    }, indent=2), encoding="utf-8")
    _log(f"  summary      : {summary}")
    return 0


def main_from_args(args) -> int:
    real_args = RealTrainArgs(
        max_steps=args.max_steps,
        device=getattr(args, "device", "auto"),
        d_model=getattr(args, "d_model", 256),
        n_layers=getattr(args, "n_layers", 4),
        log_every=getattr(args, "log_every", 10),
        ckpt_every=getattr(args, "ckpt_every", 500),
        resume=getattr(args, "resume", None),
    )
    return asyncio.run(run_real_train(real_args))
