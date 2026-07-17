"""Filum command-line interface.

Usage:

    python -m ai.filum init                    # build a fresh model + tokenizer
    python -m ai.filum config                   # print arch + VRAM math
    python -m ai.filum chat                     # interactive REPL
    python -m ai.filum bench                    # quick architecture sanity test
    python -m ai.filum collect --steps 5000    # generate distill samples
    python -m ai.filum train --max-steps 50000 # full training run

The CLI is the user-facing entry point. Every command's privacy
behaviour is documented inline; LOCAL_ONLY-by-default for `chat`
unless the user passes `--allow-teacher`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from .config import FilumConfig
from .privacy_modes import PrivacyMode, PrivacyPolicy

logger = logging.getLogger("ai.filum.cli")


def _print_config(cfg: FilumConfig) -> None:
    p = cfg.estimate_param_count()
    train = cfg.estimate_vram_mb(training=True)
    deploy = cfg.estimate_vram_mb(training=False, bitnet=True)
    print(f"Filum architecture")
    print(f"  vocab_size       : {cfg.vocab_size}")
    print(f"  d_model          : {cfg.d_model}")
    print(f"  n_layers         : {cfg.n_layers}")
    print(f"  n_heads / kv     : {cfg.n_heads} / {cfg.n_kv_heads}")
    print(f"  d_ff             : {cfg.d_ff}")
    print(f"  context_length   : {cfg.context_length}")
    print(f"")
    print(f"Parameters")
    print(f"  embedding        : {p['embedding_M']:.2f} M")
    print(f"  per_layer        : {p['per_layer_M']:.2f} M")
    print(f"  layers_total     : {p['layers_total_M']:.2f} M")
    print(f"  TOTAL            : {p['total_M']:.2f} M")
    print(f"")
    print(f"Training VRAM (GTX 1650 ceiling 4096 MB)")
    print(f"  weights fp16     : {train['weights_MB']} MB")
    print(f"  optimizer state  : {train['optimizer_state_gb' if 'optimizer_state_gb' in train else 'adamw_state_MB']} {'GB' if 'optimizer_state_gb' in train else 'MB'}")
    print(f"  gradients fp16   : {train['grad_buffer_MB']} MB")
    print(f"  activations      : {train['activations_MB']} MB")
    print(f"  TOTAL            : {train['total_MB']} MB")
    print(f"")
    print(f"Deploy VRAM (BitNet b1.58)")
    print(f"  weights ternary  : {deploy['weights_MB']} MB")
    print(f"  KV cache         : {deploy['kv_cache_MB']} MB")
    print(f"  TOTAL            : {deploy['total_MB']} MB")


def _cmd_config(args) -> int:
    cfg = FilumConfig()
    _print_config(cfg)
    return 0


def _cmd_init(args) -> int:
    """Allocate the work dir + write a config file. Doesn't allocate
    any tensors yet (avoids needing torch on the dev machine)."""
    cfg = FilumConfig()
    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    print(f"Filum workspace initialised at {work.resolve()}")
    _print_config(cfg)
    print(f"")
    print(f"Next steps:")
    print(f"  python -m ai.filum bench       # 30-second arch sanity check")
    print(f"  python -m ai.filum collect ... # gather distillation data")
    print(f"  python -m ai.filum train  ...  # multi-day training")
    return 0


def _cmd_bench(args) -> int:
    """Run a quick architecture sanity check on a tiny config so we
    don't allocate hundreds of MB of tensors during a smoke test."""
    try:
        import torch
    except ImportError:
        print("ERROR: torch not installed. pip install torch first.")
        return 2
    from .architecture import FilumArchConfig, FilumModel
    cfg = FilumArchConfig(
        vocab_size=256, context_length=64,
        d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16,
        d_ff=128, ssm_every_n_layers=2, sliding_window=32,
    )
    model = FilumModel(cfg)
    n = model.n_params()
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    y = model(x)
    print(f"Filum tiny-bench: forward pass OK")
    print(f"  params       : {n:,}")
    print(f"  input shape  : {tuple(x.shape)}")
    print(f"  output shape : {tuple(y.shape)}")
    print(f"  sanity       : {'PASS' if y.shape == (1, 16, cfg.vocab_size) else 'FAIL'}")
    return 0


def _cmd_chat(args) -> int:
    """Interactive REPL. Privacy mode defaults to LOCAL_ONLY (no
    network); pass --allow-teacher to enable speculative escalation."""
    if args.allow_teacher:
        mode = PrivacyMode.HYBRID
    elif args.allow_mesh:
        mode = PrivacyMode.MESH_FULL
    else:
        mode = PrivacyMode.LOCAL_ONLY
    policy = PrivacyPolicy.from_mode(mode)
    print(f"Filum chat ({policy.explain()})")
    print(f"Type 'exit' to quit.")
    try:
        import torch
        from .architecture import FilumArchConfig, FilumModel
    except ImportError:
        print("ERROR: torch not installed.")
        return 2
    # For the bare CLI, use a tiny config so it actually runs without
    # a trained checkpoint -- the goal here is to prove the END-TO-END
    # plumbing works. A real run loads a checkpoint via --ckpt.
    cfg = FilumArchConfig(
        vocab_size=256, context_length=128,
        d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16,
        d_ff=128, ssm_every_n_layers=2, sliding_window=64,
    )
    model = FilumModel(cfg)
    if args.ckpt:
        try:
            sd = torch.load(args.ckpt, map_location="cpu")
            model.load_state_dict(sd)
            print(f"loaded checkpoint: {args.ckpt}")
        except Exception as e:
            print(f"failed to load {args.ckpt}: {e} -- continuing untrained")
    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip().lower() in ("exit", "quit", ":q"):
            break
        if not line.strip():
            continue
        # Byte-level encode for the demo; production uses the BPE.
        ids = list(line.encode("utf-8"))[: cfg.context_length - 1]
        ids_t = torch.tensor(ids).unsqueeze(0)
        with torch.no_grad():
            logits = model(ids_t)
            next_id = int(logits[0, -1].argmax())
        # Untrained model: just echo + show the policy.
        print(f"filum> [untrained: shipping {len(ids)}-byte echo] {line[:80]}")
        print(f"        next-token argmax: {next_id} (no signal)")
    return 0


def _cmd_collect(args) -> int:
    """Generate distillation samples by asking free-tier teachers
    over a curriculum. Honours the privacy policy (LOCAL_ONLY -> 0
    samples since teachers are gated)."""
    print(f"distillation collect: {args.steps} samples target")
    print(f"NOTE: teachers must be configured via env vars:")
    print(f"  ANTHROPIC_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY")
    print(f"This CLI command is a stub for the full pipeline -- see")
    print(f"`ai/filum/teacher_pool.TeacherPool` and `trainer.FilumTrainer`")
    print(f"for the actual collection loop.")
    return 0


def _cmd_train(args) -> int:
    """If --demo, run the 100-step CPU smoke test.
    Otherwise, run the real training loop (requires teacher API keys).
    With --adaptive, use the HPA-LRD pressure-adaptive trainer."""
    if args.demo:
        from . import demo_train
        return demo_train.main()
    if not (os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")):
        print("ERROR: real training requires at least one teacher API key.")
        print("Set ANTHROPIC_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY.")
        print("To run the demo (no API keys needed):")
        print("  python -m ai.filum train --demo")
        return 2
    if getattr(args, "adaptive", False):
        from . import hpa_trainer
        return hpa_trainer.main_from_args(args)
    from . import real_train
    return real_train.main_from_args(args)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="ai.filum", description="Filum CLI")
    sp = parser.add_subparsers(dest="cmd", required=True)

    sp.add_parser("init", help="initialise the Filum workspace")
    sp.add_parser("config", help="print architecture + VRAM math")
    sp.add_parser("bench", help="quick architecture sanity test")

    chat = sp.add_parser("chat", help="interactive chat (LOCAL_ONLY by default)")
    chat.add_argument("--ckpt", help="path to a trained checkpoint")
    chat.add_argument("--allow-teacher", action="store_true",
                      help="enable HYBRID mode (teacher escalation)")
    chat.add_argument("--allow-mesh", action="store_true",
                      help="enable MESH_FULL mode (peer inference)")

    collect = sp.add_parser("collect", help="gather distillation samples")
    collect.add_argument("--steps", type=int, default=5_000)

    train = sp.add_parser("train", help="run the training loop")
    train.add_argument("--max-steps", type=int, default=50_000)
    train.add_argument("--demo", action="store_true",
                       help="100-step CPU smoke test (no API keys)")
    train.add_argument("--device", default="auto",
                       choices=["auto", "cpu", "cuda"])
    train.add_argument("--d-model", type=int, default=256,
                       dest="d_model",
                       help="hidden size (warm-up: 256; 127M target: 896)")
    train.add_argument("--n-layers", type=int, default=4,
                       dest="n_layers",
                       help="number of layers (warm-up: 4; 127M target: 14)")
    train.add_argument("--log-every", type=int, default=10,
                       dest="log_every")
    train.add_argument("--ckpt-every", type=int, default=500,
                       dest="ckpt_every")
    train.add_argument("--resume", default=None,
                       help="path to a checkpoint to resume from")
    train.add_argument("--adaptive", action="store_true",
                       help="use HPA-LRD pressure-adaptive trainer "
                            "(prevents laptop hangs on consumer GPUs)")
    train.add_argument("--vram-cap-frac", type=float, default=0.70,
                       dest="vram_cap_frac",
                       help="adaptive: soft VRAM cap as fraction of total "
                            "(default 0.70)")
    train.add_argument("--rank-min", type=int, default=8, dest="rank_min",
                       help="adaptive: minimum GaLore rank under high pressure")
    train.add_argument("--rank-max", type=int, default=256, dest="rank_max",
                       help="adaptive: maximum GaLore rank when idle")

    fed = sp.add_parser("federation",
                        help="status of the §J1 multi-model federation "
                             "(local Filum + Ollama + remote APIs)")
    fed.add_argument("subcmd", nargs="?", default="status",
                     choices=["status", "ask"],
                     help="status: list available backends; "
                          "ask: route a prompt through the federation")
    fed.add_argument("--prompt", default=None,
                     help="prompt for `federation ask`")
    fed.add_argument("--privacy", default="HYBRID",
                     choices=["LOCAL_ONLY", "HYBRID", "MESH_FULL"],
                     help="privacy mode for `federation ask`")
    fed.add_argument("--max-tokens", type=int, default=128,
                     dest="max_tokens")

    peer = sp.add_parser("peer",
                         help="discover/add/list mesh peers (auto-form mesh, "
                              "manual peering for cross-network nodes)")
    peer.add_argument("subcmd", choices=["discover", "list", "add", "remove",
                                          "myinfo"],
                      help="discover: scan LAN+DNS+history; "
                           "list: print peers.json; "
                           "add: add ADDR:PORT (optional NODE_ID); "
                           "remove: drop a peer; "
                           "myinfo: print this node's id+ip:port to share")
    peer.add_argument("target", nargs="?", default=None,
                      help="for `add`/`remove`: ADDR:PORT (e.g. 1.2.3.4:5300)")
    peer.add_argument("--node-id", default=None, dest="node_id",
                      help="optional node public-key hex for the peer")

    job = sp.add_parser("job",
                        help="submit / list / inspect tasks routed through "
                             "the mesh auction (inference, training, anything)")
    job.add_argument("subcmd", choices=["submit", "list", "status"],
                     help="submit: route a job through the auction; "
                          "list: enumerate locally-known providers; "
                          "status: print job by id from local log")
    job.add_argument("--kind", default="inference",
                     choices=["inference", "training", "embed", "fine_tune",
                              "custom"],
                     help="job type")
    job.add_argument("--prompt", default=None,
                     help="for inference/embed: the input text")
    job.add_argument("--payload-json", default=None, dest="payload_json",
                     help="raw JSON payload (overrides --prompt)")
    job.add_argument("--cost-ceiling-usd", type=float, default=0.10,
                     dest="cost_ceiling_usd")
    job.add_argument("--latency-ceiling-ms", type=int, default=30_000,
                     dest="latency_ceiling_ms")
    job.add_argument("--privacy", default="public",
                     choices=["public", "private", "sensitive"])
    job.add_argument("--quality-floor", type=float, default=0.7,
                     dest="quality_floor")
    job.add_argument("--job-id", default=None, dest="job_id",
                     help="for `status`: the job id to inspect")

    args = parser.parse_args(argv)
    if args.cmd == "init":
        return _cmd_init(args)
    if args.cmd == "config":
        return _cmd_config(args)
    if args.cmd == "bench":
        return _cmd_bench(args)
    if args.cmd == "chat":
        return _cmd_chat(args)
    if args.cmd == "collect":
        return _cmd_collect(args)
    if args.cmd == "train":
        return _cmd_train(args)
    if args.cmd == "federation":
        return _cmd_federation(args)
    if args.cmd == "peer":
        return _cmd_peer(args)
    if args.cmd == "job":
        return _cmd_job(args)
    parser.print_help()
    return 1


def _cmd_peer(args) -> int:
    """Mesh peer management — discover (LAN+DNS), add/remove, my-info."""
    from .auto_setup import auto_setup, default_state_dir, load_runtime_config
    from .mesh_discovery import (
        DEFAULT_PORT, MeshDiscovery, add_peer_manual, detect_public_ip,
        load_peers, save_peers, quick_status,
    )
    state = default_state_dir()
    cfg = auto_setup(state_dir=state)

    if args.subcmd == "myinfo":
        ip = detect_public_ip() or "<your-public-ip>"
        nid = cfg.identity.pubkey_hex
        print(f"Your node ID  : {nid}")
        print(f"Your public IP: {ip}")
        print(f"Your port     : {DEFAULT_PORT}")
        print()
        print("Share this string with a friend (any messenger):")
        print(f"   {nid[:32]}@{ip}:{DEFAULT_PORT}")
        print()
        print("They run:")
        print(f"   python -m ai.filum peer add {ip}:{DEFAULT_PORT} --node-id {nid}")
        return 0

    if args.subcmd == "discover":
        print(quick_status(
            my_node_id=cfg.identity.pubkey_hex,
            my_port=DEFAULT_PORT,
            state_dir=cfg.state_dir,
            seeds=cfg.seed_addresses,
        ))
        return 0

    if args.subcmd == "list":
        peers = load_peers(cfg.state_dir)
        if not peers:
            print("(no peers yet — `peer discover` to scan, "
                  "or `peer add ADDR:PORT` to add manually)")
            return 0
        for p in peers:
            nid = (p.get("node_id") or "")[:16]
            print(f"  [{p.get('source','?'):<7}] "
                  f"{p.get('ip')}:{p.get('port')}  "
                  f"id={nid}{'...' if nid else '(unknown)'}")
        return 0

    if args.subcmd == "add":
        if not args.target:
            print("error: `peer add` requires ADDR:PORT")
            return 1
        if ":" not in args.target:
            print("error: target must be ADDR:PORT (e.g. 1.2.3.4:5300)")
            return 1
        addr, port_s = args.target.rsplit(":", 1)
        try:
            port = int(port_s)
        except ValueError:
            print(f"error: port must be an integer, got {port_s!r}")
            return 1
        rec = add_peer_manual(cfg.state_dir, addr, port,
                                node_id=args.node_id)
        print(f"added peer: {rec}")
        return 0

    if args.subcmd == "remove":
        if not args.target:
            print("error: `peer remove` requires ADDR:PORT")
            return 1
        if ":" not in args.target:
            print("error: target must be ADDR:PORT")
            return 1
        addr, port_s = args.target.rsplit(":", 1)
        try:
            port = int(port_s)
        except ValueError:
            print(f"error: port must be int, got {port_s!r}")
            return 1
        peers = load_peers(cfg.state_dir)
        before = len(peers)
        peers = [p for p in peers
                  if not (p.get("ip") == addr
                          and int(p.get("port", DEFAULT_PORT)) == port)]
        save_peers(cfg.state_dir, peers)
        print(f"removed {before - len(peers)} peer(s)")
        return 0

    return 1


def _cmd_job(args) -> int:
    """Task submission to the mesh — federation locally, auction globally."""
    import json as _json
    import time as _time
    import uuid

    if args.subcmd == "list":
        return _job_list_providers()
    if args.subcmd == "status":
        return _job_status(args.job_id)
    if args.subcmd != "submit":
        print(f"unknown job subcmd: {args.subcmd}")
        return 1

    # Build payload.
    if args.payload_json:
        try:
            payload = _json.loads(args.payload_json)
        except Exception as e:
            print(f"--payload-json failed to parse: {e}")
            return 1
    elif args.prompt is not None:
        payload = {"prompt": args.prompt, "max_tokens": 256}
    else:
        print("error: --prompt or --payload-json required for `submit`")
        return 1

    job_id = "job-" + uuid.uuid4().hex[:12]
    print(f"Submitting {args.kind} job {job_id}...")
    print(f"  cost ceiling : ${args.cost_ceiling_usd:.4f}")
    print(f"  latency cap  : {args.latency_ceiling_ms} ms")
    print(f"  privacy class: {args.privacy}")

    # Single end-to-end path: same JobsService the FastAPI router uses.
    # The auction routes the job to whichever provider best fits the
    # constraints (local federation for inference; peer MeshGPUProvider
    # for training when wired). No CLI fork of the execution path.
    return _job_submit_through_jobs_service(job_id, payload, args)


def _job_list_providers() -> int:
    """Print every locally-registered provider + each one's eligibility."""
    print("Locally-known providers:")
    try:
        from .hpa.model_federation import ModelFederation
        fed = ModelFederation()
        for entry in fed.list_available():
            mlist = entry.get("models", []) or ["(none listed)"]
            print(f"  [federation/{entry['backend']:<14}] "
                  f"{'local' if entry['is_local'] else 'remote'}  "
                  f"models={', '.join(mlist)}")
    except Exception as e:
        print(f"  federation probe failed: {e}")
    try:
        from .mesh_discovery import load_peers, peers_json_path
        from .auto_setup import default_state_dir
        peers = load_peers(default_state_dir())
        for p in peers:
            print(f"  [mesh/peer        ] {p.get('ip')}:{p.get('port')}  "
                  f"src={p.get('source','?')}")
    except Exception as e:
        print(f"  peer list failed: {e}")
    return 0


def _job_status(job_id) -> int:
    if not job_id:
        print("error: --job-id required for `job status`")
        return 1
    from .auto_setup import default_state_dir
    from pathlib import Path
    import json as _json
    log = Path(default_state_dir()) / "jobs.jsonl"
    if not log.exists():
        print("(no jobs.jsonl yet)")
        return 1
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            rec = _json.loads(line)
        except Exception:
            continue
        if rec.get("job_id") == job_id:
            print(_json.dumps(rec, indent=2))
            return 0
    print(f"job_id {job_id} not found")
    return 1


def _append_job_log(record: dict) -> None:
    import json as _json
    from pathlib import Path
    from .auto_setup import default_state_dir
    log = Path(default_state_dir()) / "jobs.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(_json.dumps(record) + "\n")


def _build_jobs_service():
    """Construct a JobsService wired with every locally-available Provider.

    Today this is:
      * LocalFederationProvider — Filum-Genesis + Ollama. Always present.
      * MeshGPUProvider per peers.json entry. (Bidding only; without
        an injected task_router/local_executor they raise on execute,
        which the JobsService catches and surfaces as `failed`. Useful
        for testing the auction path; production replaces with remote-
        provider proxies that round-trip to actual peer nodes.)
      * Cloud providers (OpenAI / Anthropic / Gemini / etc.) only if
        a key is configured in the OS keychain. Fail-closed by default.

    Returned JobsService can be driven directly (no HTTP) or installed
    on `app.state.jobs` for the FastAPI router. Either way the same
    submit -> auction -> execute -> settle path runs.
    """
    from api.jobs_service import JobsService
    from core.providers import (
        Auction, AnthropicProvider, MeshGPUProvider, OpenAIProvider,
    )
    from .auto_setup import default_state_dir
    from .hpa.federation_provider import LocalFederationProvider
    from .mesh_discovery import load_peers

    auction = Auction()
    auction.register(LocalFederationProvider())
    # Cloud providers — fail closed when no key is in keychain.
    for cls in (OpenAIProvider, AnthropicProvider):
        try:
            auction.register(cls())
        except Exception:
            continue
    # Peer-derived MeshGPUProviders are bid-only stubs; without a remote-
    # broker proxy we don't auto-register them in the CLI default. The
    # auction layer still runs cleanly for the local-federation case.
    return JobsService(auction=auction)


def _job_submit_through_jobs_service(job_id: str, payload: dict, args) -> int:
    """Drive the same JobsService the REST API uses — sealed-bid auction
    -> winner.execute() -> result hash + sig -> settle. End-to-end with
    no CLI-side fork of the execution pipeline."""
    import asyncio
    import time as _time

    async def _run() -> int:
        svc = _build_jobs_service()
        # JobsService allocates its own job_id; we keep the user-facing
        # one for the local jobs.jsonl trail.
        rec = await svc.submit(
            kind=args.kind,
            payload=payload,
            cost_ceiling_usd=args.cost_ceiling_usd,
            latency_ceiling_ms=args.latency_ceiling_ms,
            privacy_class=args.privacy,
            quality_floor=args.quality_floor,
            requester_identity=f"cli:{job_id}",
        )

        if rec.state == "failed":
            print(f"auction: {rec.detail}")
            if rec.auction_result is not None:
                for r in rec.auction_result.rejected:
                    print(f"  rejected: {r}")
            _append_job_log({
                "job_id": job_id, "kind": args.kind, "ok": False,
                "reason": rec.detail or "submission_failed",
                "submitted_at": rec.submitted_at_unix,
            })
            return 1

        # Wait for the execution task to reach a terminal state.
        terminal = {"completed", "failed", "cancelled", "timeout"}
        deadline = _time.monotonic() + max(2.0, args.latency_ceiling_ms / 1000.0 + 5.0)
        while rec.state not in terminal and _time.monotonic() < deadline:
            await asyncio.sleep(0.05)

        # Surface what happened.
        winner = rec.matched_provider_pubkey or "(none)"
        price = rec.price_locked_usd if rec.price_locked_usd is not None else 0.0
        print()
        print(f"  winner       : {winner}")
        print(f"  price (USD)  : {price:.4f}")
        print(f"  state        : {rec.state}")
        if rec.detail:
            print(f"  detail       : {rec.detail}")
        if rec.execution_ms is not None:
            print(f"  exec time    : {rec.execution_ms:.0f} ms")
        if rec.result_hash_hex:
            print(f"  result hash  : {rec.result_hash_hex}")
        if rec.provider_signature_b64:
            print(f"  prov sig     : {rec.provider_signature_b64[:32]}...")

        # If a textual result is present, print it. The JobsService
        # forwards `result_text` from the provider response when set.
        text_out = None
        # auction_result.winner is the Bid; the recorded provider is set
        # after execute. Check the executor's cached output from JobsService.
        # JobsService.to_result currently exposes result_b64 not result_text;
        # we re-derive from result_b64 if present.
        if rec.result_b64:
            try:
                import base64 as _b64
                text_out = _b64.b64decode(rec.result_b64).decode(
                    "utf-8", errors="replace",
                )
            except Exception:
                pass
        if text_out:
            print()
            print(text_out)

        _append_job_log({
            "job_id": job_id,
            "service_job_id": rec.job_id,
            "kind": args.kind,
            "ok": rec.state == "completed",
            "state": rec.state,
            "detail": rec.detail,
            "winner": winner,
            "price_usd": price,
            "result_hash": rec.result_hash_hex,
            "execution_ms": rec.execution_ms,
            "submitted_at": rec.submitted_at_unix,
        })
        return 0 if rec.state == "completed" else 1

    return asyncio.run(_run())


def _cmd_federation(args) -> int:
    """Status / ask via the §J1 multi-model federation."""
    from .hpa.model_federation import (
        FederationConfig, GenerationRequest, ModelFederation, quick_status,
    )
    if args.subcmd == "status" or (args.subcmd == "ask" and not args.prompt):
        if args.subcmd == "ask" and not args.prompt:
            print("(--prompt required for `federation ask`; printing status)\n")
        print(quick_status())
        return 0
    fed = ModelFederation(config=FederationConfig(issue_receipts=True))
    avail = fed.list_available()
    if not avail:
        print("Federation has no backends available.")
        print("  Install Ollama (https://ollama.com) for local LLM access,")
        print("  or run `python -m ai.filum.genesis_bootstrap` to build")
        print("  a Filum-Genesis checkpoint.")
        return 1
    req = GenerationRequest(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        privacy_mode=args.privacy,
        require_receipt=True,
    )
    try:
        resp = fed.generate(req)
    except RuntimeError as e:
        print(f"federation error: {e}")
        return 1
    print(f"[{resp.backend_name}/{resp.model_id}] "
          f"({resp.elapsed_s:.2f}s)")
    if resp.receipt_id:
        print(f"  receipt: {resp.receipt_id}")
    print()
    print(resp.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
