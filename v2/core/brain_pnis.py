"""PNIS integration surface for the Pluginfer core.

The new high-capability brain (PluginferLM, CP-AI-1 through CP-AI-5) is
exposed through a single class `PluginferBrainPNIS` whose five methods
correspond to the five intelligence modules:

  parse_job(description)            -> structured job spec (Module 5)
  route_job(job_spec)               -> recommended GPU + runtime (Module 1)
  score_provider(provider_id, hist) -> quality score + anomaly flag (Module 2)
  price(market_state)               -> floor/ceiling + surge (Module 3)
  detect_anomaly(behaviour_vec)     -> anomaly score + flag (Module 4)

This class is what task_router.py / providers.py / mesh_controller call
when they want a brain decision. It is NOT the same as the existing
`core.pluginfer_brain.PluginferBrain` (which is a small MLP for
low-level routing decisions); both can coexist.

Until task heads are trained against the backbone, methods that depend
on heads (route_job / score_provider / price) emit
`{"untrained_head": True}` markers rather than fabricating answers.
parse_job and the anomaly detector are runnable as soon as the
backbone and AE are constructed (the AE is a simple feed-forward
network and is fine to use untrained for the *contract test*; the
score values will be near-uniform garbage until it's fitted on normal
data; that's the test surface).

Every call is logged via FlywheelCollector so future fine-tuning has
a real data source.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from ai.data.synthetic_generator import BEHAVIOUR_FEATURE_DIM
from ai.flywheel.collector import FlywheelCollector
from ai.inference.engine import GenerationParams, InferenceEngine
from ai.model.heads import (
    AnomalyDetectorAutoencoder,
    JobRouterHead,
    PriceEngineHead,
    ProviderQualityScorerHead,
)


class PluginferBrainPNIS:
    """Integration class for the trained PNIS backbone + task heads."""

    def __init__(
        self,
        engine: InferenceEngine,
        *,
        flywheel_dir: str | Path = "ai/data/flywheel",
        job_router_head: Optional[JobRouterHead] = None,
        provider_head: Optional[ProviderQualityScorerHead] = None,
        price_head: Optional[PriceEngineHead] = None,
        anomaly_head: Optional[AnomalyDetectorAutoencoder] = None,
        model_checkpoint_hash: str = "",
    ) -> None:
        self.engine = engine
        self.job_router_head = job_router_head
        self.provider_head = provider_head
        self.price_head = price_head
        self.anomaly_head = anomaly_head
        self.collector = FlywheelCollector(
            log_path=Path(flywheel_dir) / "events.jsonl",
            model_checkpoint_hash=model_checkpoint_hash,
        )

    # ------------------------------------------------------------------
    # Module 5 - NL job parsing (backbone-only, no head needed)
    # ------------------------------------------------------------------

    def parse_job(self, description: str) -> dict:
        t0 = time.time()
        prompt = f"<JOB>{description}<SEP>"
        text = self.engine.generate(
            prompt, GenerationParams(max_new_tokens=64, temperature=0.7)
        )
        out = {
            "input": description,
            "structured_text": text,
            "parsed": _try_parse_structured(text),
        }
        self.collector.log(
            module="parse_job",
            input_value=description,
            output_value=out,
            latency_ms=(time.time() - t0) * 1000.0,
        )
        return out

    # ------------------------------------------------------------------
    # Module 1 - Job Router
    # ------------------------------------------------------------------

    def route_job(self, job_spec: dict | str) -> dict:
        t0 = time.time()
        if self.job_router_head is None:
            description = job_spec if isinstance(job_spec, str) else json.dumps(job_spec)
            text = self.engine.generate(
                f"<JOB>{description}<SEP>",
                GenerationParams(max_new_tokens=48, temperature=0.4),
            )
            out = {
                "untrained_head": True,
                "structured_text": text,
                "parsed": _try_parse_structured(text),
            }
        else:
            # When head is attached: backbone -> last hidden -> head.
            # Implementation deferred to head-fine-tune phase.
            raise NotImplementedError(
                "JobRouterHead pooled-hidden path requires head fine-tuning; "
                "see CP-AI-FINAL+"
            )
        self.collector.log(
            module="route_job",
            input_value=job_spec,
            output_value=out,
            latency_ms=(time.time() - t0) * 1000.0,
        )
        return out

    # ------------------------------------------------------------------
    # Module 2 - Provider Quality
    # ------------------------------------------------------------------

    def score_provider(self, provider_id: str, history: list[dict]) -> dict:
        t0 = time.time()
        if self.provider_head is None:
            out = {
                "untrained_head": True,
                "provider_id": provider_id,
                "n_history_events": len(history),
            }
        else:
            raise NotImplementedError(
                "ProviderQualityScorerHead pooled-hidden path requires head "
                "fine-tuning; see CP-AI-FINAL+"
            )
        self.collector.log(
            module="score_provider",
            input_value={"provider_id": provider_id, "n_events": len(history)},
            output_value=out,
            latency_ms=(time.time() - t0) * 1000.0,
        )
        return out

    # ------------------------------------------------------------------
    # Module 3 - Price Engine
    # ------------------------------------------------------------------

    def price(self, market_state: dict) -> dict:
        t0 = time.time()
        if self.price_head is None:
            out = {"untrained_head": True, "market_state_keys": list(market_state.keys())}
        else:
            raise NotImplementedError(
                "PriceEngineHead pooled-hidden path requires head fine-tuning; "
                "see CP-AI-FINAL+"
            )
        self.collector.log(
            module="price",
            input_value=market_state,
            output_value=out,
            latency_ms=(time.time() - t0) * 1000.0,
        )
        return out

    # ------------------------------------------------------------------
    # Module 4 - Anomaly Detector (AE; runs as soon as constructed)
    # ------------------------------------------------------------------

    def detect_anomaly(self, behaviour_features: list[float]) -> dict:
        import torch

        t0 = time.time()
        if len(behaviour_features) != BEHAVIOUR_FEATURE_DIM:
            raise ValueError(
                f"behaviour_features must have length {BEHAVIOUR_FEATURE_DIM}; "
                f"got {len(behaviour_features)}"
            )
        if self.anomaly_head is None:
            out = {
                "untrained_head": True,
                "n_features": len(behaviour_features),
            }
        else:
            x = torch.tensor(behaviour_features, dtype=torch.float32).unsqueeze(0)
            result = self.anomaly_head.anomaly_score(x)
            out = {
                "score": float(result["score"].item()),
                "is_anomalous": bool(result["is_anomalous"].item()),
            }
        self.collector.log(
            module="detect_anomaly",
            input_value={"n_features": len(behaviour_features)},
            output_value=out,
            latency_ms=(time.time() - t0) * 1000.0,
        )
        return out

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        st = self.engine.status()
        st["flywheel_events"] = self.collector.count()
        st["heads_attached"] = {
            "job_router": self.job_router_head is not None,
            "provider": self.provider_head is not None,
            "price": self.price_head is not None,
            "anomaly": self.anomaly_head is not None,
        }
        return st


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STRUCT_RE = re.compile(
    r"<(GPU|VRAM|RUNTIME|PRICE|QUALITY|ANOMALY)>([^<]*)"
)


def _try_parse_structured(text: str) -> dict:
    """Best-effort parse of <GPU>...<VRAM>...<RUNTIME>... markers.

    Returns whatever it can extract; missing keys are simply absent.
    Until the model is trained these markers won't be present at all,
    so this returns {} - the parse_job caller can detect that and
    fall back to its own logic.
    """
    out: dict = {}
    for m in _STRUCT_RE.finditer(text):
        key, val = m.group(1).lower(), m.group(2).strip()
        if not val:
            continue
        if key in {"vram", "runtime"}:
            try:
                out[key] = int(re.sub(r"[^\d-]", "", val) or "0") or None
                if out[key] is None:
                    del out[key]
            except ValueError:
                pass
        elif key == "price":
            try:
                out[key] = float(re.sub(r"[^\d.\-]", "", val) or "0")
            except ValueError:
                pass
        else:
            out[key] = val
    return out
