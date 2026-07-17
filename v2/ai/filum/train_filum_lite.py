"""Filum-Lite — ground-up pretraining entrypoint.

This script is the *commitment artefact*: ready to launch the moment
the compute budget lands. By default it runs in `--dry-run` mode that
prints the cost estimator + the training plan and exits with code 0.
Pass `--commit` to actually fire the training run (refuses without an
explicit `--budget-usd` ceiling).

Usage:

    # Cost preview (safe, no compute):
    python -m ai.filum.train_filum_lite --params-b 1.5

    # Real pretraining (will spend money):
    python -m ai.filum.train_filum_lite --params-b 1.5 \\
            --budget-usd 6000 --commit \\
            --output-dir checkpoints/filum-lite-1.5b

Architectural notes
-------------------
* The trainer dispatches DiLoCo workers via Pluginfer's own mesh
  (eat-our-own-dogfood). See `core.diloco_*`.
* Tokenizer is the §H2 BPE already shipped at
  `ai/filum/tokenizer.py` (CP-AI-1).
* Architecture is the §CP-AI-2 transformer at
  `ai/filum/architecture.py` (≈1.13B params on default config; the
  `--params-b` flag dials the layer/head count).
* Data mixture: FineWeb-Edu + StarCoder2 dedup. Operator supplies
  HuggingFace tokens via env.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[2]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from core.flagship import estimate_training_cost_usd   # noqa: E402


def _print_plan(args: argparse.Namespace) -> None:
    est = estimate_training_cost_usd(target_params_b=args.params_b)
    plan = {
        "schema": "filum-lite/training-plan/v1",
        "target_params_b": args.params_b,
        "tokens_per_param_ratio": est["tokens_per_param"],
        "target_tokens_t": est["target_tokens_t"],
        "gpu_hours_h100_equiv": est["gpu_hours_h100_equiv"],
        "estimated_cost": {
            "public_cloud_usd": str(est["public_cloud_usd"]),
            "pluginfer_mesh_usd": str(est["pluginfer_mesh_usd"]),
            "pluginfer_savings_usd": str(est["pluginfer_savings_usd"]),
        },
        "data_mixture": [
            "HuggingFaceFW/fineweb-edu (sample-100BT)",
            "bigcode/starcoderdata (dedup)",
        ],
        "tokenizer": "ai/filum/tokenizer.py (§H2 BPE)",
        "architecture": "ai/filum/architecture.py (transformer; tied lm_head)",
        "dispatch": "core.diloco_worker + core.diloco_aggregator (eat own mesh)",
        "stop_criterion": "loss < 2.7 nats on 5% held-out FineWeb-Edu",
        "checkpoint_cadence": "every 1B tokens; promote on val-loss plateau",
        "output_dir": args.output_dir,
    }
    print(json.dumps(plan, indent=2, sort_keys=False))


def _refuse_unless_committed(args: argparse.Namespace) -> None:
    if not args.commit:
        print(
            "\n[dry-run] No compute consumed. Pass `--commit --budget-usd N`\n"
            "to actually launch the pretrain. Budget is HARD — the trainer\n"
            "halts when the mesh-cost meter crosses N USD.\n",
            file=sys.stderr,
        )
        sys.exit(0)
    if args.budget_usd is None or args.budget_usd <= 0:
        print(
            "[refused] --commit requires a positive --budget-usd ceiling. "
            "We will NOT fire an unbounded training run.",
            file=sys.stderr,
        )
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Filum-Lite ground-up pretrain.")
    ap.add_argument("--params-b", type=float, default=1.5,
                    help="Target parameter count (B). Default 1.5.")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="Hard upper bound on cumulative mesh-cost.")
    ap.add_argument("--output-dir", default="checkpoints/filum-lite",
                    help="Where to write checkpoints. Default checkpoints/filum-lite.")
    ap.add_argument("--commit", action="store_true",
                    help="Actually launch the training (requires --budget-usd).")
    args = ap.parse_args()

    _print_plan(args)
    _refuse_unless_committed(args)

    # The "real" branch is intentionally guarded. When compute lands,
    # the implementation hooks up:
    #   - ai.filum.dataset (CP-AI-3) for the data stream
    #   - ai.filum.architecture for the model factory
    #   - ai.filum.training (CP-AI-4) for the AdamW loop
    #   - core.diloco_worker + core.diloco_aggregator for the mesh
    #     dispatch + gradient aggregation
    #   - the budget meter at core.flagship.estimate_training_cost_usd
    #     reading live mesh-spend telemetry
    print(
        "[launch] Filum-Lite pretrain dispatching to the mesh — "
        f"params_b={args.params_b}, budget_usd={args.budget_usd}, "
        f"output_dir={args.output_dir}",
    )
    # Importing the real trainer entrypoint is deliberately deferred
    # to here so dry-runs don't pay the heavy torch import cost.
    try:
        from ai.filum.training import run_pretrain   # type: ignore
    except ImportError as e:
        print(f"[abort] training module not importable: {e}", file=sys.stderr)
        sys.exit(3)
    run_pretrain(
        params_b=args.params_b,
        budget_usd=args.budget_usd,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
