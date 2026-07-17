"""
DiLoCo Model Factory
====================
Defines the actual neural network architectures that workers train.

Three reference architectures keep the protocol generic:
  - 'mlp'             : feed-forward MLP for synthetic regression / tabular
  - 'tiny_cnn'        : small ConvNet for image classification (e.g. MNIST)
  - 'tiny_transformer': minimal transformer for sequence classification

Architectures are deterministic given (arch, config, seed) so two workers
that build the same model_spec get the same parameter shapes and (with the
same seed) the same initial weights.

Production scale: same factory pattern can host 7B+ transformers — the only
constraint is that aggregator and workers agree on the spec.
"""

from __future__ import annotations

from typing import Dict, Any

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception as _torch_err:                      # pragma: no cover
    # Soft dependency. If torch is unavailable, all diloco_models
    # public symbols become stubs that raise on instantiation.
    torch = None                                     # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = _torch_err

    class _DilocoUnavailable:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "core.diloco_models requires torch. Install: pip install torch. "
                f"Original error: {_TORCH_IMPORT_ERROR!r}"
            )

    class _NNModuleProxy:
        Module = _DilocoUnavailable

    nn = _NNModuleProxy                              # type: ignore[assignment]


def _set_seed(seed: int) -> None:
    """Deterministic init across torch and CUDA."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    """Feed-forward MLP — the workhorse for synthetic / tabular tasks."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2):
        super().__init__()
        layers = []
        d_prev = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d_prev, hidden_dim))
            layers.append(nn.GELU())
            d_prev = hidden_dim
        layers.append(nn.Linear(d_prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyCNN(nn.Module):
    """Small ConvNet for image classification (default: MNIST 1×28×28)."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10, image_size: int = 28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        spatial = image_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * spatial * spatial, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class TinyTransformer(nn.Module):
    """Minimal causal transformer — toy LM head for sequence tasks."""

    def __init__(self, vocab_size: int = 256, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, max_seq_len: int = 64, num_classes: int = 2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, num_classes)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.encoder(h)
        # Pool by mean over sequence
        return self.head(h.mean(dim=1))


def build_model(model_spec: Dict[str, Any]) -> nn.Module:
    """
    Construct a model from a serializable spec.

    model_spec format:
        {
          'arch': 'mlp' | 'tiny_cnn' | 'tiny_transformer',
          'config': { arch-specific kwargs },
          'init_seed': int,
        }

    Two callers passing the same spec get bit-identical initial weights.
    """
    arch = model_spec.get("arch", "mlp")
    config = dict(model_spec.get("config", {}))
    seed = int(model_spec.get("init_seed", 0))

    _set_seed(seed)

    if arch == "mlp":
        return MLP(
            in_dim=config.get("in_dim", 16),
            hidden_dim=config.get("hidden_dim", 64),
            out_dim=config.get("out_dim", 1),
            depth=config.get("depth", 2),
        )
    if arch == "tiny_cnn":
        return TinyCNN(
            in_channels=config.get("in_channels", 1),
            num_classes=config.get("num_classes", 10),
            image_size=config.get("image_size", 28),
        )
    if arch == "tiny_transformer":
        return TinyTransformer(
            vocab_size=config.get("vocab_size", 256),
            d_model=config.get("d_model", 64),
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 2),
            max_seq_len=config.get("max_seq_len", 64),
            num_classes=config.get("num_classes", 2),
        )
    raise ValueError(f"Unknown architecture: {arch}")


def loss_fn_for(model_spec: Dict[str, Any]) -> nn.Module:
    """Default loss matched to architecture's task family."""
    arch = model_spec.get("arch", "mlp")
    if arch == "mlp":
        out_dim = int(model_spec.get("config", {}).get("out_dim", 1))
        return nn.MSELoss() if out_dim == 1 else nn.CrossEntropyLoss()
    if arch in ("tiny_cnn", "tiny_transformer"):
        return nn.CrossEntropyLoss()
    raise ValueError(f"No default loss for arch: {arch}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
