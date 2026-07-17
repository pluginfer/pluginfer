"""Synthetic training-data generators for the 5 PNIS modules.

All data is procedurally generated from finite slot pools. No scraped
or copyrighted text. Same random seed = same dataset, so train/eval
splits are reproducible.

Data realism comes from grounding in the same catalogs the tokenizer
was trained on (`ai/tokenizer/vocab_builder.py`):
  - GPU_CATALOG: real GPU specs (vram, fp16 tflops, tier)
  - MODEL_CATALOG: real model param counts and min_vram requirements

Ground truth is computed deterministically:
  - `recommended_gpu` is the smallest tier that satisfies model.min_vram
    plus a safety margin determined by task type
  - `runtime_ms` is derived from a (model, gpu, task) regression
    grounded in fp16 tflops
  - Provider attack patterns have signature behavioural fingerprints
    (high message rate + zero job-completion = sybil; low duration
    delta + verified=False = lazy provider; etc.)
  - Price floor/ceiling derive from a simulated demand curve indexed
    by hour-of-day x workload-class
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Iterable

from ai.tokenizer.vocab_builder import (
    GPU_CATALOG,
    GPU_CLASSES,
    MODEL_CATALOG,
    MODEL_NAMES,
    JOB_TEMPLATES,
    DATASET_SIZES,
    DATA_TYPES,
    PRIORITIES,
    PHRASE_LEAD_INS,
    ALL_TASKS,
    GPU_TIERS,
)

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

ATTACK_TYPES: tuple[str, ...] = (
    "honest",
    "sybil",
    "bid_manipulation",
    "result_forgery",
    "timing",
    "eclipse",
    "lazy",
)

# Feature vector for the anomaly autoencoder; matches AnomalyDetectorAutoencoder.
BEHAVIOUR_FEATURE_DIM: int = 64

# Mapping from VRAM requirement to the smallest GPU class that satisfies it.
# Used as ground-truth for the JobRouter classifier.
def _smallest_gpu_for_vram(vram_gb: float) -> str:
    candidates = sorted(
        (
            (gpu, int(spec["vram_gb"]))  # type: ignore[arg-type]
            for gpu, spec in GPU_CATALOG.items()
        ),
        key=lambda x: x[1],
    )
    for gpu, gv in candidates:
        if gv >= vram_gb:
            return gpu
    # Should never happen since H100-NVL has 94 GB
    return candidates[-1][0]


# ---------------------------------------------------------------------------
# Example dataclasses
# ---------------------------------------------------------------------------

@dataclass
class JobRouterExample:
    """One training example for Module 1 (Job Router).

    `input` is free-text the user submitted. `label` is the structured
    spec the router must predict.
    """

    input: str
    label: dict
    # Render-friendly copy of label for LM training (text the model
    # should learn to emit after the <SEP> token).
    label_text: str = ""

    def to_dict(self) -> dict:
        return {"input": self.input, "label": self.label, "label_text": self.label_text}


@dataclass
class ProviderSequenceExample:
    """Module 2: a sequence of past job outcomes for one provider, plus
    the quality label the model must predict for the next 24h."""

    input: list[dict]  # list of {job_type, duration_delta, verified, rep_delta}
    label: dict  # {quality_score, reliability_24h, anomaly_flag, anomaly_reason}

    def to_dict(self) -> dict:
        return {"input": self.input, "label": self.label}


@dataclass
class PriceScenarioExample:
    """Module 3: a market-state snapshot + the optimal price-range label."""

    input: dict  # market state vector
    label: dict  # {floor, ceiling, demand_1hr, supply_1hr, surge_factor}

    def to_dict(self) -> dict:
        return {"input": self.input, "label": self.label}


@dataclass
class AnomalyExample:
    """Module 4: a 64-d behaviour vector + binary anomaly label."""

    input: list[float]  # behaviour features
    label: dict  # {is_anomalous: bool, attack_type: str}

    def to_dict(self) -> dict:
        return {"input": self.input, "label": self.label}


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Module 1: Job Router
    # ------------------------------------------------------------------

    def generate_job_router_training_data(self, n: int) -> list[dict]:
        out: list[dict] = []
        for _ in range(n):
            model_name = self.rng.choice(MODEL_NAMES)
            model_spec = MODEL_CATALOG[model_name]
            min_vram = float(model_spec["min_vram"])  # type: ignore[arg-type]

            template = self.rng.choice(JOB_TEMPLATES)
            slots = {
                "lead": self.rng.choice(PHRASE_LEAD_INS),
                "task": self.rng.choice(ALL_TASKS),
                "model": model_name,
                "model2": self.rng.choice(MODEL_NAMES),
                "dsize": self.rng.choice(DATASET_SIZES),
                "dtype": self.rng.choice(DATA_TYPES),
                "priority": self.rng.choice(PRIORITIES),
                "gpu": self.rng.choice(GPU_CLASSES),
                "gpu2": self.rng.choice(GPU_CLASSES),
                "tier": self.rng.choice(GPU_TIERS),
                "batch": self.rng.choice([1, 2, 4, 8, 16, 32, 64, 128]),
                "n_images": self.rng.choice([1, 4, 16, 64, 100, 500, 1000]),
                "res": self.rng.choice(["256", "512", "768", "1024", "2048"]),
                "budget": self.rng.choice([1, 5, 10, 25, 50, 100, 500, 1000]),
            }
            input_text = template.format(**slots)

            # Ground-truth derivation
            # 1) Required VRAM = model min_vram * task multiplier
            task_text = slots["task"]
            if task_text in (
                "fine-tune",
                "LoRA-finetune",
                "QLoRA-finetune",
                "instruction-tune",
                "continue pretraining of",
                "do RLHF on",
                "DPO-tune",
            ):
                vram_mult = 2.5  # training needs activations + optimiser state
            else:
                vram_mult = 1.1
            required_vram = min_vram * vram_mult
            recommended_gpu = _smallest_gpu_for_vram(required_vram)

            # 2) Runtime estimate: tflops-grounded
            gpu_spec = GPU_CATALOG[recommended_gpu]
            tflops = float(gpu_spec["tflops_fp16"])  # type: ignore[arg-type]
            params_b = float(model_spec["params_b"])  # type: ignore[arg-type]
            # Inference: ~2 * params * 1 token = 2*params FLOP per token,
            # assume 100 tokens per request as rough average. Training is
            # 6x flop per token, scale by dataset size.
            if vram_mult > 1.5:  # training
                tokens = 100_000  # rough proxy
                flops = 6 * params_b * 1e9 * tokens
            else:  # inference
                tokens = 100
                flops = 2 * params_b * 1e9 * tokens
            runtime_s = flops / (tflops * 1e12)
            # Add log-normal jitter so the regression has variance to
            # learn from; sigma=0.4 in log space (~1 stop spread).
            runtime_s *= math.exp(self.rng.gauss(0.0, 0.4))
            runtime_ms = max(1.0, runtime_s * 1000)

            label = {
                "recommended_gpu": recommended_gpu,
                "vram_gb": float(int(gpu_spec["vram_gb"])),  # type: ignore[arg-type]
                "runtime_ms": runtime_ms,
                "model_family": model_name.split("-")[0],
                "model_size_params": params_b * 1e9,
                "task_type": "training" if vram_mult > 1.5 else "inference",
                "confidence": min(1.0, 0.6 + self.rng.random() * 0.4),
            }
            label_text = (
                f"<GPU>{recommended_gpu}<VRAM>{int(label['vram_gb'])}"
                f"<RUNTIME>{int(runtime_ms)}<PRICE>"
                f"{0.0001 * runtime_ms / 1000:.4f}"
            )
            out.append(JobRouterExample(input=input_text, label=label, label_text=label_text).to_dict())
        return out

    # ------------------------------------------------------------------
    # Module 2: Provider Quality
    # ------------------------------------------------------------------

    def generate_provider_sequences(
        self,
        n: int,
        min_seq_len: int = 8,
        max_seq_len: int = 100,
    ) -> list[dict]:
        out: list[dict] = []
        for _ in range(n):
            attack = self.rng.choice(ATTACK_TYPES)
            seq_len = self.rng.randint(min_seq_len, max_seq_len)
            sequence: list[dict] = []
            for _i in range(seq_len):
                # Honest providers: verified=True majority, duration ~ expected.
                # Lazy: verified=False often, duration_delta < 0 (submit too fast).
                # Sybil: many short sessions, low verification rate, abnormal cadence.
                # Result_forgery: verified=True but rep_delta drifts negative on audit.
                # Timing: duration_delta has anomalously low variance.
                # Eclipse: duration alternates wildly; rep_delta sawtooth.
                if attack == "honest":
                    verified = self.rng.random() < 0.95
                    duration_delta = self.rng.gauss(0.0, 0.1)
                    rep_delta = 0.01 if verified else -0.02
                elif attack == "lazy":
                    verified = self.rng.random() < 0.4
                    duration_delta = self.rng.gauss(-0.6, 0.1)
                    rep_delta = -0.05 if not verified else 0.0
                elif attack == "sybil":
                    verified = self.rng.random() < 0.7
                    duration_delta = self.rng.gauss(0.0, 0.05)
                    rep_delta = self.rng.gauss(0.0, 0.005)
                elif attack == "result_forgery":
                    verified = True  # clean verifications, but late audits flag
                    duration_delta = self.rng.gauss(0.0, 0.08)
                    rep_delta = self.rng.gauss(-0.03, 0.01)
                elif attack == "timing":
                    verified = self.rng.random() < 0.95
                    duration_delta = self.rng.gauss(0.0, 0.005)  # eerily flat
                    rep_delta = 0.005
                elif attack == "eclipse":
                    verified = self.rng.random() < 0.6
                    duration_delta = (
                        self.rng.gauss(0.4, 0.1)
                        if (_i % 2 == 0)
                        else self.rng.gauss(-0.4, 0.1)
                    )
                    rep_delta = 0.01 if (_i % 2 == 0) else -0.02
                else:  # bid_manipulation
                    verified = self.rng.random() < 0.5
                    duration_delta = self.rng.gauss(0.2, 0.3)
                    rep_delta = self.rng.gauss(-0.01, 0.02)
                sequence.append(
                    {
                        "job_type": self.rng.choice(["llm", "image", "audio", "code", "vision"]),
                        "duration_delta": duration_delta,
                        "verified": verified,
                        "rep_delta": rep_delta,
                    }
                )
            # Aggregate -> label
            verified_rate = sum(1 for s in sequence if s["verified"]) / seq_len
            mean_delta = sum(s["duration_delta"] for s in sequence) / seq_len
            mean_rep = sum(s["rep_delta"] for s in sequence) / seq_len
            quality_score = max(
                0.0,
                min(1.0, 0.5 + 0.5 * verified_rate + mean_rep - abs(mean_delta) * 0.2),
            )
            anomaly = attack != "honest"
            label = {
                "quality_score": quality_score,
                "reliability_24h": max(0.0, min(1.0, verified_rate * 0.9)),
                "anomaly_flag": anomaly,
                "anomaly_reason": attack if anomaly else "",
            }
            out.append(ProviderSequenceExample(input=sequence, label=label).to_dict())
        return out

    # ------------------------------------------------------------------
    # Module 3: Price Engine
    # ------------------------------------------------------------------

    def generate_price_scenarios(self, n: int) -> list[dict]:
        out: list[dict] = []
        for _ in range(n):
            hour = self.rng.randint(0, 23)
            day_of_week = self.rng.randint(0, 6)
            queued = self.rng.randint(0, 5000)
            providers = self.rng.randint(10, 5000)
            util = {tier: self.rng.random() for tier in GPU_TIERS}
            recent_clears = [self.rng.uniform(0.001, 5.0) for _ in range(10)]

            # Demand curve: peak at office hours, trough at night
            # demand_factor in [0.3, 1.5]
            demand_factor = 1.2 - 0.5 * math.cos(2 * math.pi * hour / 24)
            if day_of_week >= 5:
                demand_factor *= 0.7  # weekend dip
            queue_pressure = math.tanh(queued / 1000.0)
            supply_pressure = 1.0 / max(0.1, providers / 1000.0)

            base_price = sum(recent_clears) / len(recent_clears)
            floor = base_price * (0.7 + 0.2 * (1 - queue_pressure))
            ceiling = base_price * (1.3 + 0.5 * queue_pressure * demand_factor)
            surge = max(1.0, demand_factor * (1.0 + queue_pressure))
            demand_1hr = queued * demand_factor * 1.2
            supply_1hr = providers * 0.6  # rough capacity

            input_state = {
                "queued_jobs": queued,
                "active_providers": providers,
                "gpu_utilization_by_class": util,
                "hour_of_day": hour,
                "day_of_week": day_of_week,
                "recent_auction_clearing_prices": recent_clears,
            }
            label = {
                "recommended_floor_price": max(0.0001, floor),
                "recommended_ceiling_price": max(0.001, ceiling),
                "demand_forecast_1hr": max(0.0, demand_1hr),
                "supply_forecast_1hr": max(0.0, supply_1hr),
                "surge_factor": surge,
            }
            out.append(PriceScenarioExample(input=input_state, label=label).to_dict())
        return out

    # ------------------------------------------------------------------
    # Module 4: Anomaly Detector
    # ------------------------------------------------------------------

    def generate_anomaly_examples(self, n: int) -> list[dict]:
        """Balanced 50/50 normal/attack. Returns 64-d feature vectors."""
        out: list[dict] = []
        for _ in range(n):
            is_anomalous = self.rng.random() < 0.5
            if is_anomalous:
                attack_type = self.rng.choice(ATTACK_TYPES[1:])  # exclude 'honest'
                features = self._anomalous_features(attack_type)
            else:
                attack_type = "honest"
                features = self._normal_features()
            assert len(features) == BEHAVIOUR_FEATURE_DIM
            out.append(
                AnomalyExample(
                    input=features,
                    label={"is_anomalous": is_anomalous, "attack_type": attack_type},
                ).to_dict()
            )
        return out

    def _normal_features(self) -> list[float]:
        # 64-dim vector: msg_rate, bid_rate, accept_rate, timing stats, ...
        # We sample from a narrow Gaussian centred on normal-population means.
        return [self.rng.gauss(0.5, 0.1) for _ in range(BEHAVIOUR_FEATURE_DIM)]

    def _anomalous_features(self, attack_type: str) -> list[float]:
        feats = self._normal_features()
        # Each attack type leaves a distinct fingerprint by tweaking
        # specific feature dims. The autoencoder, trained on normal-only,
        # can't reconstruct these tweaks well -> high MSE.
        if attack_type == "sybil":
            # High message rate, near-zero job completion, low VRAM diversity
            feats[0:8] = [self.rng.gauss(2.5, 0.3) for _ in range(8)]
        elif attack_type == "bid_manipulation":
            # Heavy spike in bid_rate then drop; high cancel rate
            feats[8:16] = [self.rng.gauss(3.0, 0.5) for _ in range(8)]
        elif attack_type == "result_forgery":
            # Clean exec metrics but audit-mismatch dim spikes
            feats[16:24] = [self.rng.gauss(2.0, 0.4) for _ in range(8)]
        elif attack_type == "timing":
            # Eerily uniform timing -> low variance feature spikes
            feats[24:32] = [self.rng.gauss(0.0, 0.01) for _ in range(8)]
        elif attack_type == "eclipse":
            # Network-isolation signature: peer-set churn dims spike
            feats[32:40] = [self.rng.gauss(2.5, 0.3) for _ in range(8)]
        elif attack_type == "lazy":
            # Submit-too-fast -> negative duration delta dims
            feats[40:48] = [self.rng.gauss(-2.0, 0.3) for _ in range(8)]
        return feats

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def generate_all(
        self,
        n_router: int = 1000,
        n_provider: int = 500,
        n_price: int = 500,
        n_anomaly: int = 1000,
    ) -> dict[str, list[dict]]:
        return {
            "job_router": self.generate_job_router_training_data(n_router),
            "provider_quality": self.generate_provider_sequences(n_provider),
            "price_engine": self.generate_price_scenarios(n_price),
            "anomaly": self.generate_anomaly_examples(n_anomaly),
        }
