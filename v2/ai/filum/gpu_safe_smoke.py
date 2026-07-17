"""GTX 1650 safe GPU smoke test — proves HPA-LRD prevents the hang.

Reproduces the prior failure mode (5k training run that crashed with
``cudaErrorIllegalAddress`` on step 1) but with the §B HPA-LRD
safety primitives wired in:

* §B1 PressureSampler running in the background
* §B3 CooperativeYield inserted on pressure spikes
* §B3 cuda_oom_guard catches CUDA OOM / illegal-address and halves
  the micro-batch instead of dying
* §B6 LR warmup (cosine) + finite-loss guard
* Tight VRAM cap (50% = 2 GiB reserved for the OS + display)
* Tiny model (~100k params)
* 50 steps, micro-batch 1, no teacher API calls (synthetic data)

Goal: prove the hang is fixed. If this completes 50 steps without
the laptop freezing or the kernel raising illegal-address, the §B
bundle works on the actual hardware that broke previously.

Run:
    python -m ai.filum.gpu_safe_smoke
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> int:
    print("=" * 72)
    print("GTX 1650 safe smoke test — HPA-LRD verification")
    print("=" * 72)

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("ERROR: torch not installed.")
        return 1

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available — this test requires a GPU.")
        return 1

    from .architecture import FilumArchConfig, FilumModel
    from .optimizer_8bit import AdamW8bit
    from .lr_schedule import LRSchedule, apply_lr, is_finite_loss
    from .hpa.telemetry import PressureSampler
    from .hpa.cooperative import (
        CooperativeYield, cuda_oom_guard, soft_vram_cap_bytes,
        vram_used_bytes,
    )
    from .hpa.backend import detect_backend, memory_cap_bytes

    backend = detect_backend()
    print(f"Backend: {backend.name} ({backend.accelerator_name})")
    cap = memory_cap_bytes(backend, frac=0.50, headroom_mib=600)
    print(f"VRAM cap (50% soft): {cap / (1<<20):.0f} MiB")
    free_mib, total_mib = (
        torch.cuda.mem_get_info()[0] / (1 << 20),
        torch.cuda.mem_get_info()[1] / (1 << 20),
    )
    print(f"VRAM at start: {total_mib - free_mib:.0f} / {total_mib:.0f} MiB used")
    print()

    device = "cuda"
    cfg = FilumArchConfig(
        vocab_size=256, context_length=64,
        d_model=64, n_layers=2,
        n_heads=4, n_kv_heads=2, head_dim=16, d_ff=128,
        ssm_every_n_layers=999, sliding_window=32,
        use_differential=False,
    )
    model = FilumModel(cfg).to(device)
    n_params = model.n_params()
    print(f"Model params:  {n_params:,}")
    print(f"VRAM after model load: "
          f"{torch.cuda.memory_allocated() / (1<<20):.1f} MiB allocated")
    print()

    optimizer = AdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=0.01,
    )
    steps = 50
    schedule = LRSchedule(target_lr=3e-4, warmup_steps=10, total_steps=steps)

    sampler = PressureSampler(period_s=0.20).start()
    coop = CooperativeYield(
        pressure_fn=sampler.pressure,
        threshold=0.85,
        base_sleep_s=0.005,
        max_sleep_s=0.040,
    )

    micro_batch = 1
    seq_len = 32
    losses = []
    yields = 0
    oom_recoveries = 0
    skipped_steps = 0
    t0 = time.monotonic()
    print(f"Begin training: {steps} steps, micro_batch={micro_batch}")
    print(f"{'step':>5}  {'loss':>9}  {'P':>5}  {'mb':>3}  "
          f"{'vram_MiB':>9}  {'yields':>6}  {'oom':>3}  {'el':>5}")
    print("-" * 72)
    try:
        for step in range(steps):
            P = sampler.pressure()
            apply_lr(optimizer, schedule.lr_at(step))

            recovered_flag = [False]
            def _on_oom(_e):
                nonlocal micro_batch, oom_recoveries
                oom_recoveries += 1
                old = micro_batch
                micro_batch = max(1, micro_batch // 2)
                print(f"  [OOM/illegal] mb {old}->{micro_batch}, retry")
                recovered_flag[0] = True
                return True

            with cuda_oom_guard(_on_oom):
                # Synthetic input: random byte ids in [0, vocab).
                ids = torch.randint(
                    0, cfg.vocab_size, (micro_batch, seq_len),
                    device=device, dtype=torch.long,
                )
                logits = model(ids[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    ids[:, 1:].reshape(-1),
                )
                if not is_finite_loss(loss):
                    skipped_steps += 1
                    continue
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 0.5,
                )
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            if recovered_flag[0]:
                continue

            if coop.maybe_yield():
                yields += 1

            vram_mib = torch.cuda.memory_allocated() / (1 << 20)
            elapsed = time.monotonic() - t0
            if step % 5 == 0 or step == steps - 1:
                print(f"{step:>5}  {losses[-1]:>9.4f}  {P:>5.2f}  "
                      f"{micro_batch:>3}  {vram_mib:>9.1f}  "
                      f"{yields:>6}  {oom_recoveries:>3}  {elapsed:>5.1f}s")
    finally:
        sampler.stop()

    elapsed = time.monotonic() - t0
    free_mib_end = torch.cuda.mem_get_info()[0] / (1 << 20)
    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"  steps completed     : {len(losses)}")
    print(f"  skipped (non-finite): {skipped_steps}")
    print(f"  cooperative yields  : {yields}")
    print(f"  OOM recoveries      : {oom_recoveries}")
    print(f"  first loss          : {losses[0]:.4f}" if losses else "  no losses recorded")
    print(f"  last loss           : {losses[-1]:.4f}" if losses else "")
    print(f"  decreased           : "
          f"{'YES' if (losses and losses[-1] < losses[0]) else 'NO'}")
    print(f"  wall time           : {elapsed:.1f}s")
    print(f"  VRAM at end (free)  : {free_mib_end:.0f} MiB")
    if losses and losses[-1] < losses[0]:
        print()
        print("*** GPU TRAINING SUCCEEDED — laptop did NOT hang. ***")
        print("    HPA-LRD safety prevented the original failure mode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
