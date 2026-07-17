"""High-level tokenizer wrapper that bundles BPE + special-token handling.

`BPETrainer` does the BPE work; `PluginferTokenizer` adds:
  - `encode(text, add_bos=True, add_eos=False)` for sequence-prep
  - `encode_pair(text_a, text_b)` with <SEP> insertion
  - `decode(ids, skip_special=True)` that prints `<JOB>` etc. on demand
  - `pad_to_length` helper
  - safe `__len__` for vocab size and `id_for(special_name)`
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .bpe import BPETrainer
from .special_tokens import (
    N_SPECIAL,
    SPECIAL_TOKENS,
    SPECIAL_TOKEN_NAMES,
    SpecialTokens,
    is_special,
)


class PluginferTokenizer:
    """Thin wrapper around `BPETrainer` for downstream model code."""

    def __init__(self, bpe: BPETrainer) -> None:
        self.bpe = bpe
        self.specials = SpecialTokens()

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = self.bpe.encode(text)
        if add_bos:
            ids = [self.specials.BOS] + ids
        if add_eos:
            ids = ids + [self.specials.EOS]
        return ids

    def encode_pair(
        self,
        text_a: str,
        text_b: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.specials.BOS)
        ids.extend(self.bpe.encode(text_a))
        ids.append(self.specials.SEP)
        ids.extend(self.bpe.encode(text_b))
        if add_eos:
            ids.append(self.specials.EOS)
        return ids

    def encode_batch(
        self,
        texts: Iterable[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[list[int]]:
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode ids to a string.

        With `skip_special=True` (default), special tokens are dropped.
        With `skip_special=False`, we emit each special token as its
        literal name (`<BOS>` etc.) interleaved with decoded byte runs.
        Useful for debugging packed sequences.
        """
        if skip_special:
            return self.bpe.decode([i for i in ids if not is_special(i)])

        # Mixed mode: split runs of content ids around special ids.
        out: list[str] = []
        run: list[int] = []
        for i in ids:
            if is_special(i):
                if run:
                    out.append(self.bpe.decode(run))
                    run = []
                out.append(SPECIAL_TOKEN_NAMES[i])
            else:
                run.append(i)
        if run:
            out.append(self.bpe.decode(run))
        return "".join(out)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def pad_to_length(
        self, ids: list[int], target_len: int, truncate: bool = True
    ) -> list[int]:
        if len(ids) >= target_len:
            return ids[:target_len] if truncate else ids
        return ids + [self.specials.PAD] * (target_len - len(ids))

    def id_for(self, special_name: str) -> int:
        if special_name not in SPECIAL_TOKENS:
            raise KeyError(f"unknown special token: {special_name!r}")
        return SPECIAL_TOKENS[special_name]

    def __len__(self) -> int:
        return self.bpe.actual_vocab_size

    @property
    def vocab_size(self) -> int:
        return self.bpe.actual_vocab_size

    @property
    def configured_vocab_size(self) -> int:
        """The vocab_size requested at training time. Always >= actual."""
        return self.bpe.vocab_size

    @property
    def n_special(self) -> int:
        return N_SPECIAL

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        self.bpe.save(path)

    @classmethod
    def load(cls, path: str | Path) -> "PluginferTokenizer":
        return cls(BPETrainer.load(path))

    @classmethod
    def train_new(
        cls,
        texts: Iterable[str],
        vocab_size: int = 32000,
        verbose: bool = False,
    ) -> "PluginferTokenizer":
        bpe = BPETrainer(vocab_size=vocab_size)
        bpe.train(texts, verbose=verbose)
        return cls(bpe)
