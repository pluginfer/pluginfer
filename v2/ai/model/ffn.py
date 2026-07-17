"""SwiGLU feed-forward network.

Reference: Shazeer 2020 (arXiv:2002.05202).

Why SwiGLU and not ReLU FFN:
  - SiLU (Swish) is smooth -> better gradient flow than ReLU's discontinuity.
  - Gating: SiLU(W1 x) * (W3 x) lets the network learn which features to
    amplify per position. Empirically +0.5-1.0 perplexity improvement at
    constant parameter count.
  - Used in PaLM, Llama 2/3, Mistral, Gemma.

Architecture: 3 linear projections (gate, up, down), no biases.
  out = down( silu(gate(x)) * up(x) )

Note that the conventional d_ff for SwiGLU is (8/3) * d_model so the
3-matrix block has the same parameter count as a standard 4*d_model
ReLU FFN with 2 matrices.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig


class SwiGLUFFN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # Gate projection: x -> activation values
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        # Up projection: x -> features
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        # Down projection: gated features -> hidden
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
