"""
Real DiLoCo smoke test
======================
Proves:
  1. The federated trainer runs real PyTorch SGD on real data and
     converges (loss strictly decreases).
  2. Multiple workers with disjoint local data shards aggregate via
     the AsyncDiLoCoAggregator and the global model improves on a
     held-out evaluation set across rounds.
  3. The async staleness path works: workers can submit deltas based
     on stale base rounds and they're weighted accordingly.
  4. The gradient-provenance audit (`verify_delta`) detects a
     fabricated noise delta from a malicious worker.

Runs in <30 s on CPU. No external data download required.
"""

from __future__ import annotations

import base64
import math
import sys
import time
from pathlib import Path

# --- repo import bootstrap -------------------------------------------------
_HERE = Path(__file__).resolve()
for parent in [_HERE.parents[1], _HERE.parents[2]]:
    if (parent / "core").is_dir():
        sys.path.insert(0, str(parent))
        break

import torch  # noqa: E402

from core.diloco_models import build_model, loss_fn_for, count_parameters  # noqa: E402
from core.diloco_serialize import serialize_state_dict, deserialize_state_dict  # noqa: E402
from core.diloco_worker import DiLoCoWorker, InnerLoopConfig, make_tensor_iter  # noqa: E402
from core.diloco_aggregator import (  # noqa: E402
    AsyncDiLoCoAggregator,
    AggregatorConfig,
    WorkerSubmission,
    verify_delta,
)


# --------------------------------------------------------------------------
# Test fixtures
# --------------------------------------------------------------------------
MODEL_SPEC = {
    "arch": "mlp",
    "config": {"in_dim": 16, "hidden_dim": 64, "out_dim": 1, "depth": 2},
    "init_seed": 1234,
}


def _make_global_dataset(n: int = 4096, d: int = 16, noise: float = 0.05):
    """Single 'true' regression task all workers learn together."""
    torch.manual_seed(424242)
    W = torch.randn(d, 1)
    b = torch.randn(1)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(n, d, generator=g)
    y = x @ W + b + noise * torch.randn(n, 1, generator=g)
    return x, y, W, b


def _shard(x, y, num_workers: int, worker_idx: int):
    """Disjoint shard for one worker — its 'private' local data."""
    n = x.shape[0]
    per = n // num_workers
    s = worker_idx * per
    e = s + per
    return x[s:e].clone(), y[s:e].clone()


def _global_eval_loss(model_spec, agg_payload, x_eval, y_eval) -> float:
    model = build_model(model_spec)
    state = deserialize_state_dict(agg_payload)
    model.load_state_dict(state, strict=True)
    model.eval()
    loss_fn = loss_fn_for(model_spec)
    with torch.no_grad():
        return float(loss_fn(model(x_eval), y_eval).item())


# --------------------------------------------------------------------------
# Test 1: single-worker convergence
# --------------------------------------------------------------------------
def test_single_worker_converges():
    print("\n[1] SINGLE-WORKER CONVERGENCE")
    print("-" * 60)
    x, y, _, _ = _make_global_dataset(n=2048)
    worker = DiLoCoWorker(MODEL_SPEC, device_pref="cpu")
    cfg = InnerLoopConfig(inner_steps=200, inner_lr=5e-3, batch_size=64,
                          optimizer="adamw")
    data_iter = make_tensor_iter(x, y, seed=0)

    # Round 0: no global weights yet (uses init).
    res = worker.run_round(data_iter, cfg, global_weights_payload=None)
    print(f"  initial_loss = {res.initial_loss:.4f}")
    print(f"  final_loss   = {res.final_loss:.4f}")
    print(f"  delta_norm   = {res.delta_norm:.4f}")
    print(f"  comp ratio   = {res.compression_ratio:.2f}x")
    print(f"  inner steps  = {res.inner_steps}, examples = {res.examples_seen}")
    print(f"  device       = {res.device}")
    print(f"  base hash    = {res.base_weights_hash[:12]}...")
    print(f"  final hash   = {res.final_weights_hash[:12]}...")

    assert res.final_loss < res.initial_loss * 0.5, (
        f"Loss didn't decrease enough: {res.initial_loss:.4f} -> {res.final_loss:.4f}"
    )
    assert res.delta_norm > 1e-4, "Delta is suspiciously zero — training did not happen"
    assert res.compression_ratio > 2.0, f"Compression too low: {res.compression_ratio}"
    print("  PASS")
    return res


# --------------------------------------------------------------------------
# Test 2: multi-worker async aggregation
# --------------------------------------------------------------------------
def test_async_multiworker_aggregation():
    print("\n[2] MULTI-WORKER ASYNC AGGREGATION (4 workers, 6 outer rounds)")
    print("-" * 60)
    x, y, _, _ = _make_global_dataset(n=4096)
    x_eval, y_eval = x[-512:], y[-512:]

    NUM_WORKERS = 4
    OUTER_ROUNDS = 6
    INNER_STEPS = 50

    agg = AsyncDiLoCoAggregator(
        MODEL_SPEC,
        config=AggregatorConfig(
            # Staleness-weighted async aggregation needs a more conservative
            # outer step than synchronous DiLoCo (paper uses 0.7 / 0.9). With
            # 4 async workers the effective batch is smaller and momentum
            # accumulates faster, so we damp accordingly.
            outer_lr=0.3,
            outer_momentum=0.5,
            staleness_tau=4.0,
            min_submissions_per_round=NUM_WORKERS,
            max_submissions_per_round=NUM_WORKERS,
            round_deadline_sec=120.0,
            audit_probability=0.0,  # explicit audit done in test 4
            reject_stale_after=8,
        ),
    )

    workers = [DiLoCoWorker(MODEL_SPEC, device_pref="cpu") for _ in range(NUM_WORKERS)]
    cfg = InnerLoopConfig(inner_steps=INNER_STEPS, inner_lr=5e-3,
                          batch_size=32, optimizer="adamw")

    eval_history = []
    initial_round, initial_payload, _ = agg.current_global_payload()
    eval_history.append((initial_round, _global_eval_loss(
        MODEL_SPEC, initial_payload, x_eval, y_eval)))
    print(f"  round {initial_round}: eval_loss = {eval_history[-1][1]:.4f}")

    for outer in range(OUTER_ROUNDS):
        round_idx, global_payload, _ = agg.current_global_payload()
        for w_idx, worker in enumerate(workers):
            x_shard, y_shard = _shard(x[:-512], y[:-512], NUM_WORKERS, w_idx)
            data_iter = make_tensor_iter(x_shard, y_shard, seed=w_idx + outer * 10)
            res = worker.run_round(data_iter, cfg, global_weights_payload=global_payload)
            sub = WorkerSubmission(
                worker_id=f"w{w_idx}",
                base_round=round_idx,
                quantized_delta=res.quantized_delta,
                received_at=time.time(),
                base_weights_hash=res.base_weights_hash,
                final_weights_hash=res.final_weights_hash,
                examples_seen=res.examples_seen,
            )
            ack = agg.submit_delta(sub)
            assert ack["accepted"], ack
        new_round, new_payload, _ = agg.current_global_payload()
        eval_loss = _global_eval_loss(MODEL_SPEC, new_payload, x_eval, y_eval)
        eval_history.append((new_round, eval_loss))
        print(f"  round {new_round}: eval_loss = {eval_loss:.4f}")

    initial = eval_history[0][1]
    final = eval_history[-1][1]
    print(f"  improvement: {initial:.4f} -> {final:.4f} ({(initial - final) / initial:.1%})")
    assert final < initial * 0.6, (
        f"Async DiLoCo failed to improve eval loss enough: {initial:.4f} -> {final:.4f}"
    )
    print("  PASS")
    return eval_history


# --------------------------------------------------------------------------
# Test 3: stale submission handling
# --------------------------------------------------------------------------
def test_stale_submission_weighting():
    print("\n[3] STALENESS WEIGHTING (stale worker contributes less)")
    print("-" * 60)
    agg = AsyncDiLoCoAggregator(
        MODEL_SPEC,
        config=AggregatorConfig(
            min_submissions_per_round=1,
            max_submissions_per_round=1,
            round_deadline_sec=120.0,
            staleness_tau=2.0,
            reject_stale_after=10,
        ),
    )

    # Pull the global once, then advance the round 3 times before
    # submitting based on the original snapshot.
    base_round, payload, _ = agg.current_global_payload()

    x, y, _, _ = _make_global_dataset(n=512)
    worker = DiLoCoWorker(MODEL_SPEC, device_pref="cpu")
    cfg = InnerLoopConfig(inner_steps=20, inner_lr=5e-3, batch_size=32,
                          optimizer="adamw")
    data_iter = make_tensor_iter(x, y, seed=99)
    res = worker.run_round(data_iter, cfg, global_weights_payload=payload)

    # Advance the global round by submitting fake fresh deltas with weight 1.
    for _ in range(3):
        cur_round, cur_payload, _ = agg.current_global_payload()
        fresh_res = worker.run_round(data_iter, cfg, global_weights_payload=cur_payload)
        ack = agg.submit_delta(WorkerSubmission(
            worker_id="fresh", base_round=cur_round,
            quantized_delta=fresh_res.quantized_delta,
            received_at=time.time(),
            base_weights_hash=fresh_res.base_weights_hash,
            final_weights_hash=fresh_res.final_weights_hash,
            examples_seen=fresh_res.examples_seen,
        ))
        assert ack["accepted"]

    # Now submit our STALE delta against base_round.
    stale_ack = agg.submit_delta(WorkerSubmission(
        worker_id="stale", base_round=base_round,
        quantized_delta=res.quantized_delta,
        received_at=time.time(),
        base_weights_hash=res.base_weights_hash,
        final_weights_hash=res.final_weights_hash,
        examples_seen=res.examples_seen,
    ))
    print(f"  stale submission ack = {stale_ack}")
    assert stale_ack["accepted"], stale_ack
    expected_weight = math.exp(-stale_ack["staleness"] / 2.0)
    assert abs(stale_ack["weight"] - expected_weight) < 1e-6, (
        f"Staleness weight mismatch: {stale_ack['weight']} vs {expected_weight}"
    )
    print(f"  staleness={stale_ack['staleness']} weight={stale_ack['weight']:.4f}")
    print("  PASS")


# --------------------------------------------------------------------------
# Test 4: gradient provenance audit catches a forged delta
# --------------------------------------------------------------------------
def test_audit_detects_forged_delta():
    print("\n[4] GRADIENT PROVENANCE AUDIT (catches malicious worker)")
    print("-" * 60)

    # Honest worker: trains for one step deterministically with audit_seed.
    audit_seed = 4242
    torch.manual_seed(audit_seed)
    model = build_model(MODEL_SPEC)
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    base_payload = serialize_state_dict(base_state)

    x, y, _, _ = _make_global_dataset(n=128)
    audit_batch = (x[:32], y[:32])

    # Honest delta: actually run K=1 SGD step.
    worker = DiLoCoWorker(MODEL_SPEC, device_pref="cpu")
    worker.load_global_weights(base_payload)
    cfg = InnerLoopConfig(inner_steps=1, inner_lr=1e-2, batch_size=32,
                          optimizer="sgd", quantize=True)
    torch.manual_seed(audit_seed)  # deterministic single step
    data_iter = lambda bs: (audit_batch[0][:bs], audit_batch[1][:bs])  # noqa: E731
    honest_res = worker.run_round(data_iter, cfg, global_weights_payload=base_payload)

    ok, why, cos_ok = verify_delta(
        MODEL_SPEC, base_payload,
        honest_res.quantized_delta, audit_seed=audit_seed,
        audit_step_data=audit_batch, lr=cfg.inner_lr,
    )
    print(f"  honest worker  : passed={ok} reason='{why}' cos={cos_ok:.3f}")
    assert ok, f"Honest delta should pass audit: {why}"

    # Malicious delta: pure noise of similar magnitude.
    from core.diloco_quantize import quantize_delta
    malicious = {k: torch.randn_like(v) * 0.05 for k, v in base_state.items()}
    forged_payload = quantize_delta(malicious)

    bad_ok, bad_why, cos_bad = verify_delta(
        MODEL_SPEC, base_payload,
        forged_payload, audit_seed=audit_seed,
        audit_step_data=audit_batch, lr=cfg.inner_lr,
    )
    print(f"  forged worker  : passed={bad_ok} reason='{bad_why}' cos={cos_bad:.3f}")
    assert not bad_ok, f"Forged delta should fail audit (cos={cos_bad:.3f})"
    print("  PASS")


# --------------------------------------------------------------------------
# Test 5: federated_trainer plugin end-to-end
# --------------------------------------------------------------------------
def test_plugin_end_to_end():
    print("\n[5] PLUGIN END-TO-END (federated_trainer)")
    print("-" * 60)

    from plugins.federated_trainer import FederatedTrainer

    plugin = FederatedTrainer()
    spec = MODEL_SPEC
    seed_model = build_model(spec)
    init_payload = serialize_state_dict({
        k: v.detach().clone() for k, v in seed_model.state_dict().items()
    })

    out = plugin.run({
        "model_spec": spec,
        "global_weights": base64.b64encode(init_payload).decode("ascii"),
        "inner_steps": 100,
        "inner_lr": 5e-3,
        "batch_size": 32,
        "optimizer": "adamw",
        "data": {"kind": "synthetic_regression", "seed": 11, "n": 1024, "d": 16},
        "device": "cpu",
    })

    assert "error" not in out, out
    assert out["status"] == "success"
    m = out["metrics"]
    print(f"  initial_loss   = {m['initial_loss']:.4f}")
    print(f"  final_loss     = {m['final_loss']:.4f}")
    print(f"  inner_steps    = {m['inner_steps']}")
    print(f"  examples_seen  = {m['examples_seen']}")
    print(f"  delta_norm     = {m['delta_norm']:.4f}")
    print(f"  compression    = {out['compression_ratio']:.2f}x")
    print(f"  param_count    = {out['param_count']}")
    assert m["final_loss"] < m["initial_loss"] * 0.7
    print("  PASS")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("PLUGINFER REAL-DiLoCo SMOKE TEST")
    print("=" * 60)
    t_start = time.time()
    test_single_worker_converges()
    test_async_multiworker_aggregation()
    test_stale_submission_weighting()
    test_audit_detects_forged_delta()
    test_plugin_end_to_end()
    print("\n" + "=" * 60)
    print(f"ALL TESTS PASSED in {time.time() - t_start:.1f}s")
    print("=" * 60)
