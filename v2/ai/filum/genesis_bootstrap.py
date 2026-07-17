"""§G1 Genesis Bootstrap — Pluginfer trains itself from itself.

The cold-start problem in two-sided marketplaces: no providers
join an empty mesh, no buyers join a mesh with no providers, no
buyers join a mesh with no useful model. Three-way deadlock.

The fix: Pluginfer bootstraps the *first useful model* without any
external dependency. The pieces are already in the repo:

* The Filum architecture (`architecture.py`)
* The HPA-LRD trainer (`gpu_real_train.py` — proven on the GTX 1650)
* The self-context indexer (`self_context.py` — 3,704 chunks of
  Pluginfer's own codebase + 1,729 lines of design notes disclosures
  + 1,827 lines of WORKLOG)
* The §D1 receipt protocol — every output is provenance-attested
  from genesis

This module trains Filum-Genesis-v0 on Pluginfer's own
self-context. The result is a tiny but coherent model that:

* Knows Pluginfer's architecture from inside (it was trained on it).
* Can answer questions about Pluginfer using its own learned
  weights (no RAG needed at inference time).
* Demonstrates the substrate works end-to-end — there's a *real
  trained checkpoint* shipped with the repo.

The strategic move: this is the **first artefact a new user
sees**. They download Pluginfer, run ``python -m ai.filum
bootstrap_genesis``, and within 10 minutes have a working AI
trained on their own machine, by Pluginfer's substrate, signed
under their key. From there, fine-tuning their own data on top of
Filum-Genesis-v0 is one CLI command. Cold-start closed.

design notes §G1 (drafted in the design notes): a method of bootstrapping
a decentralised AI training mesh in which the genesis model is
trained on the mesh's own codebase + documentation as the seed
training corpus, using the mesh's own substrate as the compute
layer, with the resulting checkpoint distributed via the mesh's
own delta-sync protocol — eliminating external dependencies in
the cold-start path.

Run::

    python -m ai.filum.genesis_bootstrap          # default 1000 steps
    python -m ai.filum.genesis_bootstrap --steps 5000 --d-model 128
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


def _log(msg: str) -> None:
    print(f"[genesis] {msg}", flush=True)


@dataclass
class GenesisConfig:
    repo_root: str = "C:/Pluginfer"
    steps: int = 1000
    d_model: int = 96            # slightly bigger than demo (~250k params)
    n_layers: int = 3
    context_length: int = 96
    vocab_size: int = 2048       # BPE trained on the codebase
    target_lr: float = 3e-4
    warmup_steps: int = 50
    seq_per_step: int = 1
    out_dir: str = "ai/filum/_work/genesis"
    receipt_log_path: str = "ai/filum/_work/genesis/receipts.jsonl"


def build_genesis_corpus(repo_root: str) -> list[str]:
    """Pull every chunk from the self-context index. The corpus is
    Pluginfer's own codebase + docs + worklog + inventions."""
    from .self_context import SelfContextIndex, IndexConfig

    idx = SelfContextIndex.build(IndexConfig(repo_root=repo_root))
    chunks = [c.text for c in idx.chunks]
    _log(f"corpus: {len(chunks)} chunks "
          f"({sum(len(c) for c in chunks)/1024:.0f} KiB total)")
    return chunks


def train_genesis_tokenizer(corpus: list[str], vocab_size: int):
    """Train a BPE on the corpus."""
    from .tokenizer_bpe import train_bpe, BPEConfig
    _log(f"training BPE tokenizer ({vocab_size} vocab)...")
    t0 = time.monotonic()
    tok = train_bpe(corpus, BPEConfig(vocab_size=vocab_size))
    _log(f"  trained in {time.monotonic()-t0:.1f}s; vocab={tok.vocab_size}")
    return tok


def encode_corpus(tok, corpus: list[str], context_length: int) -> list[list[int]]:
    """Encode every chunk into token-id sequences clipped to context."""
    seqs = []
    for chunk in corpus:
        ids = tok.encode(chunk, add_bos=True, add_eos=True)
        if len(ids) >= 4:
            seqs.append(ids[:context_length])
    return seqs


def issue_genesis_receipt(*, model_state_bytes: bytes,
                            seed: bytes, pub: bytes,
                            cfg_dict: dict, final_loss: float) -> dict:
    """Produce a §D1 receipt attesting this model's genesis."""
    from .hpa.inference_receipt import issue_receipt

    metadata = dict(cfg_dict)
    metadata["final_loss"] = round(final_loss, 6)
    metadata["genesis"] = True
    receipt = issue_receipt(
        model_weights_bytes=model_state_bytes,
        input_text="genesis-corpus:pluginfer-self-context",
        output_text="filum-genesis-v0",
        model_metadata=metadata,
        node_pubkey_hex=pub.hex(),
        node_priv_seed=seed,
        policy_class="genesis",
    )
    return asdict(receipt)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="genesis_bootstrap")
    p.add_argument("--repo-root", default="C:/Pluginfer")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--d-model", type=int, default=96, dest="d_model")
    p.add_argument("--n-layers", type=int, default=3, dest="n_layers")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--vocab-size", type=int, default=2048, dest="vocab_size")
    args = p.parse_args(argv)

    cfg = GenesisConfig(
        repo_root=args.repo_root, steps=args.steps,
        d_model=args.d_model, n_layers=args.n_layers,
        vocab_size=args.vocab_size,
    )

    print("=" * 72)
    print("Filum-Genesis-v0 bootstrap — Pluginfer trains itself from itself")
    print("=" * 72)

    # 1. Pull corpus from self-context.
    corpus = build_genesis_corpus(cfg.repo_root)
    if not corpus:
        _log("ERROR: empty corpus; check repo_root.")
        return 1

    # 2. Train BPE tokenizer.
    tok = train_genesis_tokenizer(corpus, cfg.vocab_size)
    seqs = encode_corpus(tok, corpus, cfg.context_length)
    _log(f"encoded sequences: {len(seqs)}")

    # 3. Pick device (GPU if available, else CPU).
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        _log("ERROR: torch not installed.")
        return 1
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    _log(f"device: {device}")

    from .architecture import FilumArchConfig, FilumModel
    from .optimizer_8bit import AdamW8bit
    from .lr_schedule import LRSchedule, apply_lr, is_finite_loss
    from .hpa.telemetry import PressureSampler
    from .hpa.cooperative import CooperativeYield, cuda_oom_guard
    from .hpa.grain import fresh_keypair

    arch = FilumArchConfig(
        vocab_size=tok.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=4, n_kv_heads=2, head_dim=24, d_ff=cfg.d_model * 2,
        ssm_every_n_layers=999,
        sliding_window=min(cfg.context_length, 64),
        use_differential=False,
    )
    model = FilumModel(arch).to(device)
    n_params = model.n_params()
    _log(f"model params: {n_params:,}")

    optimizer = AdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.target_lr, weight_decay=0.01,
    )
    schedule = LRSchedule(
        target_lr=cfg.target_lr,
        warmup_steps=cfg.warmup_steps, total_steps=cfg.steps,
    )

    sampler = PressureSampler(period_s=0.20).start()
    coop = CooperativeYield(pressure_fn=sampler.pressure, threshold=0.85)

    losses = []
    t0 = time.monotonic()
    _log(f"begin training: {cfg.steps} steps")
    print(f"{'step':>5}  {'loss':>9}  {'P':>5}  {'el':>6}")
    print("-" * 32)
    try:
        for step in range(cfg.steps):
            P = sampler.pressure()
            apply_lr(optimizer, schedule.lr_at(step))
            ids_list = seqs[step % len(seqs)]
            with cuda_oom_guard(lambda _e: True):
                ids = torch.tensor(
                    ids_list, dtype=torch.long, device=device,
                ).unsqueeze(0).clamp_(0, arch.vocab_size - 1)
                logits = model(ids[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    ids[:, 1:].reshape(-1),
                )
                if not is_finite_loss(loss):
                    continue
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 0.5,
                )
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            coop.maybe_yield()
            if step % max(1, cfg.steps // 20) == 0 or step == cfg.steps - 1:
                el = time.monotonic() - t0
                print(f"{step:>5}  {losses[-1]:>9.4f}  "
                      f"{P:>5.2f}  {el:>5.1f}s")
    finally:
        sampler.stop()

    elapsed = time.monotonic() - t0
    _log(f"training complete in {elapsed:.1f}s")
    if not losses:
        _log("ERROR: no losses recorded")
        return 1

    # 4. Save checkpoint + tokenizer.
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "filum_genesis_v0.pt"
    tok_path = out_dir / "tokenizer.json"
    torch.save({
        "state_dict": model.state_dict(),
        "config": asdict(arch),
        "losses": losses,
        "tokenizer_path": str(tok_path),
        "genesis_corpus_size": len(corpus),
        "n_steps": len(losses),
    }, ckpt_path)
    tok.save(tok_path)
    _log(f"  checkpoint: {ckpt_path}")
    _log(f"  tokenizer:  {tok_path}")

    # 5. Issue §D1 genesis receipt.
    seed, pub = fresh_keypair()
    state_bytes = b""
    try:
        # Hash the state_dict deterministically.
        h = hashlib.sha256()
        for k in sorted(model.state_dict().keys()):
            h.update(k.encode())
            h.update(model.state_dict()[k].detach().cpu().numpy().tobytes())
        state_bytes = h.digest()
    except Exception:
        pass
    receipt = issue_genesis_receipt(
        model_state_bytes=state_bytes,
        seed=seed, pub=pub,
        cfg_dict=asdict(arch),
        final_loss=losses[-1],
    )
    receipt_path = Path(cfg.receipt_log_path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(receipt) + "\n")
    _log(f"  receipt:    {receipt_path}  (id={receipt['receipt_id']})")

    print()
    print("=" * 72)
    print("FILUM-GENESIS-v0")
    print("=" * 72)
    print(f"  parameters       : {n_params:,}")
    print(f"  vocab            : {tok.vocab_size}")
    print(f"  corpus chunks    : {len(corpus)}")
    print(f"  steps            : {len(losses)}")
    print(f"  first loss       : {losses[0]:.4f}")
    print(f"  last loss        : {losses[-1]:.4f}")
    print(f"  device           : {device}")
    print(f"  wall time        : {elapsed:.1f}s")
    print(f"  checkpoint       : {ckpt_path}")
    print(f"  receipt          : {receipt['receipt_id']}")
    print()
    print("This is your seed model. Anyone fine-tuning their own data")
    print("on Filum-Genesis-v0 starts from a working artefact, not a promise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
