"""Latency / throughput benchmarks for InferenceEngine.

Returns:
  - latency_ms_first_token : prefill + first-decode time
  - tokens_per_sec         : steady-state decode throughput
  - peak_memory_bytes      : approximate model+cache RSS estimate

Usage:
    from ai.inference.benchmarks import benchmark
    metrics = benchmark(engine, prompt="Run SDXL inference",
                        max_new_tokens=50)
"""

from __future__ import annotations

import time

from .engine import GenerationParams, InferenceEngine


def benchmark(
    engine: InferenceEngine,
    prompt: str = "Run SDXL inference on a 1024px image",
    max_new_tokens: int = 50,
    temperature: float = 0.0,
) -> dict:
    params = GenerationParams(
        max_new_tokens=max_new_tokens, temperature=temperature
    )

    # Warm up so the first run's autograd-graph + RoPE-cache build don't
    # pollute the timing.
    _ = engine.generate(prompt, params)

    # Prefill + first token
    t0 = time.time()
    ids = engine.tokenizer.encode(prompt, add_bos=True, add_eos=False)
    out_ids = engine.generate_ids(ids, GenerationParams(max_new_tokens=1, temperature=0.0))
    first_token_ms = (time.time() - t0) * 1000.0

    # Steady-state: time the full max_new_tokens run
    t0 = time.time()
    out_ids = engine.generate_ids(ids, params)
    full_ms = (time.time() - t0) * 1000.0
    emitted = len(out_ids) - len(ids)
    tps = emitted / (full_ms / 1000.0) if full_ms > 0 else 0.0

    return {
        "prompt_tokens": len(ids),
        "emitted_tokens": emitted,
        "latency_ms_first_token": first_token_ms,
        "total_ms": full_ms,
        "tokens_per_sec": tps,
    }
