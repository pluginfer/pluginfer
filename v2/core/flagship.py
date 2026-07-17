"""G4 — alpha-tier flagship model registration via open-weights loader.

Why this exists
---------------
The Pluginfer supply side needs a marquee workload that providers can
serve from day one. Training Filum-Lite (sub-3B, MIT-licensed,
ground-up) is a 2-6 week / $2-8k compute action — gated on funding +
calendar, not on code. Meanwhile every Chromium tab and every laptop
in the mesh wants to be useful TODAY.

The architectural answer: register a permissively-licensed open model
(Qwen2.5-1.5B-Instruct, Gemma-2-2B-Instruct, Llama-3.2-3B-Instruct —
each MIT/Apache/Llama-Community-licensed for redistribution) as the
network's **alpha-tier flagship**. PNIS receipts stamp the upstream
model_id transparently so there is no claim that Pluginfer trained
the weights — only that Pluginfer routed the inference and signed the
receipt. When Filum-Lite (our own pretrain) ships, the registration
just swaps the spec.

This module is **transport-agnostic**. It does not import torch /
transformers; the caller supplies a `run(prompt)` callable that
returns bytes. That callable can wrap llama-cpp-python, ollama, vllm,
huggingface transformers — whatever the operator has installed. The
adapter layer lives in `ai/filum/flagship_adapters/` (out of scope
for the core registration).

Public API
----------
* `FlagshipModelSpec` — frozen dataclass capturing model identity +
  capability targets + licence.
* `register_alpha_flagship(jobs_service, spec, runner_fn)` —
  attaches an in-process MeshGPUProvider to the auction wired to
  the runner_fn.
* `ALPHA_FLAGSHIPS` — pre-populated catalogue of three default
  open-weights specs.
* `estimate_training_cost_usd(target_params)` — pure-function cost
  estimator for when the operator wants to launch a real
  Filum-Lite pretrain.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FlagshipModelSpec:
    """Describes the alpha-tier flagship the operator wants to serve."""
    model_id: str                       # HuggingFace-style id
    licence: str                        # "MIT" | "Apache-2.0" | "Llama-Community" | ...
    parameter_count_b: float            # e.g. 1.5
    context_window_tokens: int
    target_quality_score: float = 0.7   # informs Bid.expected_quality
    target_price_per_1k_tok_usd: float = 0.0002
    target_eta_ms: int = 2000
    hardware_class: str = "consumer-gpu-mid"
    privacy_grade: str = "private"      # mesh > cloud APIs by default

    def to_receipt_model_field(self) -> dict:
        """Stamp on every signed PNIS receipt as `model.id` so the
        audit trail records *which* upstream weights served the
        completion. Transparency = trust."""
        return {
            "id": self.model_id,
            "hash": "",   # set per-call from the result bytes
            "kind": "llm",
            "licence": self.licence,
            "parameter_count_b": self.parameter_count_b,
        }


# Pre-populated catalogue of permissively-licensed open weights the
# operator can register without paying for a pretrain run. Order is
# meaningful: the first entry whose runner is available is the default.
ALPHA_FLAGSHIPS: List[FlagshipModelSpec] = [
    FlagshipModelSpec(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        licence="Apache-2.0",
        parameter_count_b=1.5,
        context_window_tokens=32_768,
        target_quality_score=0.72,
        target_price_per_1k_tok_usd=0.0002,
        target_eta_ms=1200,
        hardware_class="consumer-gpu-mid",
    ),
    FlagshipModelSpec(
        model_id="google/gemma-2-2b-it",
        licence="Gemma-License",
        parameter_count_b=2.0,
        context_window_tokens=8_192,
        target_quality_score=0.74,
        target_price_per_1k_tok_usd=0.00025,
        target_eta_ms=1400,
        hardware_class="consumer-gpu-mid",
    ),
    FlagshipModelSpec(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        licence="Llama-Community",
        parameter_count_b=3.0,
        context_window_tokens=128_000,
        target_quality_score=0.78,
        target_price_per_1k_tok_usd=0.00035,
        target_eta_ms=1800,
        hardware_class="consumer-gpu-high",
    ),
]


def spec_for_runtime(model_id: str, runtime_name: str) -> FlagshipModelSpec:
    """The spec MUST describe what the runner actually serves — receipts
    stamp `model.id` from it, and a receipt claiming Qwen while echo (or
    gemma) answered is a provenance lie. Catalogue match first; honest
    echo pseudo-model when no runtime resolved; otherwise a spec built
    around the real model id with parameter count parsed from the name."""
    for s in ALPHA_FLAGSHIPS:
        if s.model_id.lower() == model_id.lower():
            return s
    if runtime_name in ("echo", "alpha-echo"):
        return FlagshipModelSpec(
            model_id="pluginfer/alpha-echo",
            licence="n/a",
            parameter_count_b=0.0,
            context_window_tokens=4_096,
            target_quality_score=0.1,
            target_price_per_1k_tok_usd=0.0,
            target_eta_ms=50,
            hardware_class="cpu",
        )
    import re
    m = re.search(r"(\d+(?:\.\d+)?)b", model_id.lower())
    return FlagshipModelSpec(
        model_id=model_id,
        licence="upstream",
        parameter_count_b=float(m.group(1)) if m else 0.0,
        context_window_tokens=8_192,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

@dataclass
class FlagshipProvider:
    """Provider-shaped wrapper around a runner_fn.

    Conforms to the same surface as `core.providers.Provider` (bid +
    execute), but bypasses the abstract base to keep this module
    free of the heavy provider-stack import."""

    spec: FlagshipModelSpec
    runner_fn: Callable[[str, Dict[str, Any]], bytes]
    wallet: Any   # core.tokenomics.Wallet — we keep it loose-typed to
                  # avoid the import at module load.
    provider_id: str = field(default="")
    privacy_grade: str = "private"
    hardware_class: str = "consumer-gpu-mid"
    enabled: bool = True

    def __post_init__(self):
        if not self.provider_id:
            self.provider_id = f"flagship-{self.spec.model_id.replace('/', '-')}"
        self.privacy_grade = self.spec.privacy_grade
        self.hardware_class = self.spec.hardware_class

    def bid(self, job: Any) -> Any:
        """Return a Bid if we can serve this job kind, else None."""
        # Imports are local to keep this module's import-time light.
        from core.providers import Bid, PRIVACY_PUBLIC, PRIVACY_PRIVATE
        if not self.enabled:
            return None
        # Only bid on llm.* / embed kinds — the flagship is a chat
        # model, not a generic compute worker.
        kind = getattr(job, "kind", "") or ""
        if not (kind.startswith("llm.") or kind == "embed"):
            return None
        approx_tokens = 0
        try:
            payload = getattr(job, "payload", {}) or {}
            approx_tokens = float(payload.get("max_tokens", 200))
        except Exception:
            approx_tokens = 200.0
        price = self.spec.target_price_per_1k_tok_usd * (approx_tokens / 1000.0)
        return Bid(
            provider_id=self.provider_id,
            price_usd=price,
            eta_ms=self.spec.target_eta_ms,
            expected_quality=self.spec.target_quality_score,
            privacy_grade=self.privacy_grade,
            evidence={
                "model_id": self.spec.model_id,
                "licence": self.spec.licence,
                "tier": "alpha-flagship",
            },
        )

    def execute(self, job: Any, bid: Any, *, on_delta: Optional[Callable] = None) -> dict:
        prompt = (getattr(job, "payload", {}) or {}).get("prompt", "")
        t0 = time.monotonic()
        try:
            result_bytes = self.runner_fn(prompt, getattr(job, "payload", {}) or {})
        except Exception as e:
            return {
                "status": "failed",
                "job_id": getattr(job, "job_id", ""),
                "reason": f"flagship_runner_error: {type(e).__name__}: {e}",
            }
        if not isinstance(result_bytes, (bytes, bytearray)):
            result_bytes = str(result_bytes).encode("utf-8")
        result_hash = hashlib.sha256(result_bytes).hexdigest()
        try:
            sig = self.wallet.sign(result_hash)
        except Exception:
            sig = ""
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return {
            "status": "executed",
            "job_id": getattr(job, "job_id", ""),
            "result_bytes": base64.b64encode(bytes(result_bytes)).decode("ascii"),
            "result_hash": result_hash,
            "provider_sig": sig,
            "provider_pubkey_pem": getattr(self.wallet, "public_key_pem", ""),
            "execution_ms": elapsed_ms,
            "model_id": self.spec.model_id,
        }


def register_alpha_flagship(
    *,
    jobs_service: Any,
    spec: FlagshipModelSpec,
    runner_fn: Callable[[str, Dict[str, Any]], bytes],
    wallet: Optional[Any] = None,
) -> FlagshipProvider:
    """Wire a flagship into the auction. Returns the registered
    provider so the operator can disable/enable it at runtime."""
    if wallet is None:
        from core.tokenomics import Wallet
        wallet = Wallet()
    provider = FlagshipProvider(spec=spec, runner_fn=runner_fn, wallet=wallet)
    jobs_service.auction.register(provider)
    logger.info(
        "alpha-flagship registered: %s (licence=%s, %.1fB params)",
        spec.model_id, spec.licence, spec.parameter_count_b,
    )
    return provider


# ---------------------------------------------------------------------------
# Cost estimator for when Filum-Lite gets a real pretrain budget
# ---------------------------------------------------------------------------

# Empirical anchors from public training reports (2024-2025 cohort):
#  * DeepSeek-V3:   $5.6M / 671B params / 14.8T tokens
#  * Qwen2-1.5B:    ~$60k / 1.5B params / 7T tokens
#  * Llama-3.2-1B:  ~$45k / 1B params / 9T tokens
#
# Pluginfer's mesh has a 30-60% cost advantage on raw GPU $/hr per the
# §A11 / §A12 architecture; we conservatively model 40% off the
# public-cloud quote.
TOKENS_PER_PARAM_RATIO = 5_000          # Chinchilla floor
PUBLIC_CLOUD_USD_PER_GPU_HOUR_H100 = 2.50
PLUGINFER_DISCOUNT = 0.40               # 40% cheaper on the mesh
GPU_HOURS_PER_B_PARAM_PER_T_TOKENS = 9_000   # H100-equivalent


def estimate_training_cost_usd(
    target_params_b: float,
    *,
    tokens_per_param: int = TOKENS_PER_PARAM_RATIO,
    discount: float = PLUGINFER_DISCOUNT,
) -> Dict[str, Any]:
    """Pure cost estimator for a Filum-Lite-style ground-up pretrain.
    Returns a dict with both the public-cloud reference and the
    Pluginfer-mesh-discounted figure."""
    target_tokens_t = (target_params_b * tokens_per_param) / 1000.0   # in T
    gpu_hours = target_params_b * target_tokens_t * GPU_HOURS_PER_B_PARAM_PER_T_TOKENS
    cloud_usd = Decimal(gpu_hours * PUBLIC_CLOUD_USD_PER_GPU_HOUR_H100)
    mesh_usd = (cloud_usd * Decimal(1 - discount)).quantize(Decimal("0.01"))
    return {
        "target_params_b": target_params_b,
        "tokens_per_param": tokens_per_param,
        "target_tokens_t": target_tokens_t,
        "gpu_hours_h100_equiv": gpu_hours,
        "public_cloud_usd": cloud_usd.quantize(Decimal("0.01")),
        "pluginfer_mesh_usd": mesh_usd,
        "pluginfer_savings_usd": (cloud_usd - mesh_usd).quantize(Decimal("0.01")),
    }


__all__ = [
    "ALPHA_FLAGSHIPS",
    "FlagshipModelSpec",
    "FlagshipProvider",
    "estimate_training_cost_usd",
    "register_alpha_flagship",
]
