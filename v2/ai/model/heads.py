"""Task-specific heads for the five Pluginfer intelligence modules.

Each head consumes a hidden representation produced by `PluginferLM`
(typically the final-position hidden state, which is the standard
sequence-summary trick: train the model to put the answer at the last
token) and emits the structured output for its task.

The heads are intentionally small. The LM backbone does the heavy
lifting of language understanding; heads are 1-2 linear layers + an
activation. This keeps multi-task training cheap (each head is < 1M
params) and lets a single backbone serve all five modules with separate
checkpoints per task.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig

# Number of GPU classes the JobRouter classifies into. Must stay in sync
# with `ai/data/synthetic_generator.py::GPU_CLASSES`.
DEFAULT_N_GPU_CLASSES: int = 13


class JobRouterHead(nn.Module):
    """Module 1: classify GPU class + regress (vram, runtime, confidence).

    Inputs:  pooled hidden state of shape (B, d_model)
    Outputs: dict {gpu_logits, vram_gb, runtime_ms_log, confidence}
    """

    def __init__(
        self, config: ModelConfig, n_gpu_classes: int = DEFAULT_N_GPU_CLASSES
    ) -> None:
        super().__init__()
        d = config.d_model
        self.gpu_classifier = nn.Sequential(
            nn.Linear(d, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_gpu_classes),
        )
        self.vram_regressor = nn.Linear(d, 1)
        # We predict log-runtime (ms) because runtime distributions are
        # heavy-tailed log-normal: log space gives stable MSE loss.
        self.runtime_regressor = nn.Linear(d, 1)
        self.confidence_head = nn.Sequential(
            nn.Linear(d, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, h: Tensor) -> dict[str, Tensor]:
        return {
            "gpu_logits": self.gpu_classifier(h),
            "vram_gb": F.softplus(self.vram_regressor(h)).squeeze(-1),
            "runtime_ms_log": self.runtime_regressor(h).squeeze(-1),
            "confidence": self.confidence_head(h).squeeze(-1),
        }


class ProviderQualityScorerHead(nn.Module):
    """Module 2: score a provider given a sequence of past job outcomes.

    Inputs:  pooled hidden state of shape (B, d_model)
    Outputs: dict {quality, reliability_24h, anomaly_logit}
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model
        self.quality = nn.Sequential(
            nn.Linear(d, 256), nn.SiLU(), nn.Linear(256, 1), nn.Sigmoid()
        )
        self.reliability = nn.Sequential(
            nn.Linear(d, 256), nn.SiLU(), nn.Linear(256, 1), nn.Sigmoid()
        )
        # Logit so callers can pair with BCEWithLogitsLoss without re-sigmoiding.
        self.anomaly_logit = nn.Linear(d, 1)

    def forward(self, h: Tensor) -> dict[str, Tensor]:
        return {
            "quality": self.quality(h).squeeze(-1),
            "reliability_24h": self.reliability(h).squeeze(-1),
            "anomaly_logit": self.anomaly_logit(h).squeeze(-1),
        }


class PriceEngineHead(nn.Module):
    """Module 3: predict price floor + ceiling + 1hr supply/demand forecast.

    All outputs use softplus so they're guaranteed non-negative; downstream
    code adds a small epsilon and divides for ratio calculations.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model
        self.trunk = nn.Sequential(nn.Linear(d, 256), nn.SiLU(), nn.Dropout(0.1))
        self.price_head = nn.Linear(256, 2)  # (floor, ceiling)
        self.flow_head = nn.Linear(256, 3)   # (demand, supply, surge)

    def forward(self, h: Tensor) -> dict[str, Tensor]:
        x = self.trunk(h)
        floor_ceil = F.softplus(self.price_head(x))
        flows = F.softplus(self.flow_head(x))
        return {
            "floor": floor_ceil[..., 0],
            "ceiling": floor_ceil[..., 1],
            "demand_1hr": flows[..., 0],
            "supply_1hr": flows[..., 1],
            "surge_factor": flows[..., 2],
        }


class AnomalyDetectorAutoencoder(nn.Module):
    """Module 4: anomaly detector via reconstruction error.

    NOT attached to the LM backbone - operates directly on a flat
    behaviour-feature vector (message rates, bid patterns, timing
    statistics, etc). Trained on normal-only data; reconstruction error
    spikes for anomalies because the bottleneck can't memorise the
    out-of-distribution input.

    Default input_dim=64 matches the feature vector built in
    `ai/data/synthetic_generator.py::generate_anomaly_examples`.
    """

    def __init__(self, input_dim: int = 64, bottleneck_dim: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.SiLU(),
            nn.Linear(32, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.SiLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        # Per-row reconstruction error (mean over feature dim).
        err = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)
        return x_hat, err

    @torch.no_grad()
    def anomaly_score(self, x: Tensor, threshold: float = 3.0) -> dict:
        """Return a dict with the anomaly score and a boolean flag.

        `threshold` is in units of training-set MSE std. Pass after fitting
        baseline_mean / baseline_std on normal data; here we report the
        raw error so the caller can apply its own threshold.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        _, err = self.forward(x)
        return {
            "score": err,
            "is_anomalous": err > threshold,
        }
