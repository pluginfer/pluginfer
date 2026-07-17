"""Filum -- Pluginfer's in-built ground-up AI.

A 32M-param transformer designed for the GTX 1650 / 4 GB VRAM /
16 GB RAM laptop class. Trained NOT by spending compute, but by
distilling structured signal from free-tier LLM teachers (Gemini
Flash, Claude Haiku, etc.) over a curated curriculum.

Sized so a complete training run fits in ~600 MB of disk
(model + tokenizer + teacher cache) and converges to genuinely
useful task performance in a few days of intermittent runs on the
target hardware.

Usage:

    # 1. Build a fresh Filum + tokenizer
    python -m ai.filum init

    # 2. Generate distillation data from free-tier teachers
    python -m ai.filum collect --steps 5000 --max-budget-usd 5

    # 3. Train (resumable; safe to ctrl-C at any time)
    python -m ai.filum train --epochs 3

    # 4. Chat
    python -m ai.filum chat
"""

from .config import FilumConfig

__all__ = ["FilumConfig"]
__version__ = "0.1.0"
