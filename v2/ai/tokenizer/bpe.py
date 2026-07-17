"""Byte-level Byte-Pair Encoding from scratch.

Reference: Sennrich, Haddow & Birch 2016 -- "Neural Machine Translation
of Rare Words with Subword Units" (arXiv:1508.07909). Implemented
fresh, not copied: pair counts via `collections.Counter`, merges
applied via single-pass scan of each sequence, vocab persisted as a
plain JSON document with byte keys hex-encoded for portability.

Performance notes:
  - Pair counting is the hot inner loop. We use `Counter.update(zip(...))`
    on each sequence which is O(L) per sequence and is fast enough for
    corpora up to ~tens of millions of bytes on a single core.
  - We DO NOT split on whitespace before BPE (the GPT-2 trick). It
    costs us ~10-20% compression at convergence but keeps the algorithm
    obvious and gives us cross-word merges for the recurring domain
    phrases we care about ("vram", "tokens/sec", "rtx-4090").
  - Pure Python; no numpy/torch dependency. The trainer must be
    runnable on a chain-only node (no torch installed).

Vocabulary layout (centralised here so other modules can rely on it):

  IDs 0 .. N_SPECIAL - 1                : special tokens (PAD/BOS/...)
  IDs N_SPECIAL .. N_SPECIAL + 255      : raw bytes 0..255 (offset)
  IDs N_SPECIAL + 256 .. vocab_size - 1 : learned BPE merges

`BYTE_OFFSET = N_SPECIAL = 13` is exposed as the public constant.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

from .special_tokens import N_SPECIAL, SPECIAL_TOKEN_NAMES

BYTE_OFFSET: int = N_SPECIAL  # byte b -> id (b + BYTE_OFFSET)
N_BYTE_TOKENS: int = 256
BASE_VOCAB_SIZE: int = N_SPECIAL + N_BYTE_TOKENS  # 269

# Saved-file format version. Bump on any breaking change to the JSON layout.
TOKENIZER_FORMAT_VERSION: int = 1


class BPETrainer:
    """Train, save, and load a byte-level BPE vocabulary.

    Typical use:

        trainer = BPETrainer(vocab_size=32000)
        trainer.train(list_of_strings)
        trainer.save("tokenizer.json")
        # ... later:
        trainer2 = BPETrainer.load("tokenizer.json")
        ids = trainer2.encode("Run SDXL inference")
        text = trainer2.decode(ids)  # exact roundtrip

    The trainer also serves as the runtime tokenizer; `PluginferTokenizer`
    is the higher-level wrapper that adds special-token insertion.
    """

    def __init__(self, vocab_size: int = 32000) -> None:
        if vocab_size < BASE_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size must be >= {BASE_VOCAB_SIZE} "
                f"(specials + bytes); got {vocab_size}"
            )
        self.vocab_size: int = vocab_size
        # `merges` is an ordered list of (a, b, new_id). Order is load-bearing
        # at encode time: earlier merges are applied first.
        self.merges: list[tuple[int, int, int]] = []
        # `vocab` maps id -> the byte sequence that the id decodes to.
        # Specials map to placeholder b"" - they are never produced by BPE
        # and the higher-level tokenizer handles their string form.
        self.vocab: dict[int, bytes] = {}
        for i in range(N_SPECIAL):
            self.vocab[i] = b""  # special tokens have no byte content
        for b in range(N_BYTE_TOKENS):
            self.vocab[BYTE_OFFSET + b] = bytes([b])
        # Fast lookup for encode: pair (a, b) -> new_id.
        self._merge_lookup: dict[tuple[int, int], int] = {}
        # Optional metric captured at training time.
        self.training_time_seconds: float = 0.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @staticmethod
    def _text_to_ids(text: str) -> list[int]:
        """UTF-8 text -> list of byte ids with the +BYTE_OFFSET shift."""
        return [b + BYTE_OFFSET for b in text.encode("utf-8")]

    @staticmethod
    def _get_pair_freqs(sequences: list[list[int]]) -> Counter:
        """Count adjacent pairs across all sequences. O(sum of lengths)."""
        counts: Counter = Counter()
        for seq in sequences:
            if len(seq) < 2:
                continue
            counts.update(zip(seq, seq[1:]))
        return counts

    @staticmethod
    def _merge_pair_in_sequence(
        seq: list[int], a: int, b: int, new_id: int
    ) -> list[int]:
        """Replace every adjacent (a, b) with new_id in one left-to-right pass.

        Linear time; allocates one new list per sequence per merge step.
        Greedy non-overlapping replacement: ...aaab... with merge (a,b)
        -> ...aa<new>... (the leftmost match wins; that's the BPE rule).
        """
        out: list[int] = []
        i = 0
        n = len(seq)
        while i < n:
            if i < n - 1 and seq[i] == a and seq[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    def train(
        self,
        texts: Iterable[str],
        verbose: bool = False,
        progress_every: int = 500,
    ) -> None:
        """Run BPE training to completion.

        `texts` is iterated once; pass a list if you need to reuse the
        corpus elsewhere. Trains until `len(self.vocab) == self.vocab_size`
        OR no more improvable pairs exist.
        """
        t0 = time.time()
        sequences: list[list[int]] = [self._text_to_ids(t) for t in texts if t]
        if not sequences:
            raise ValueError("training corpus is empty")

        next_id = BASE_VOCAB_SIZE  # first merge id
        target_merges = self.vocab_size - BASE_VOCAB_SIZE

        for step in range(target_merges):
            pair_counts = self._get_pair_freqs(sequences)
            if not pair_counts:
                break  # no more pairs; corpus fully fused
            (a, b), freq = pair_counts.most_common(1)[0]
            if freq < 2:
                # No pair occurs more than once; further merges only memorise
                # noise. Stop early; vocab will be smaller than requested.
                break
            new_id = next_id
            next_id += 1
            self.merges.append((a, b, new_id))
            self._merge_lookup[(a, b)] = new_id
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            sequences = [
                self._merge_pair_in_sequence(seq, a, b, new_id) for seq in sequences
            ]
            if verbose and (step + 1) % progress_every == 0:
                avg_len = sum(len(s) for s in sequences) / len(sequences)
                print(
                    f"  merge {step + 1}/{target_merges} "
                    f"freq={freq} new_id={new_id} avg_len={avg_len:.1f}"
                )

        self.training_time_seconds = time.time() - t0

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids by applying learned merges in order.

        Complexity: O(n * len(merges)) worst case. For the typical case
        where most merges don't apply to a given input, the active-pair
        loop short-circuits and runtime is closer to O(n + n_active_merges).
        """
        ids = self._text_to_ids(text)
        if not self.merges:
            return ids
        # Greedy lowest-id-merge-first: at each pass we find the
        # lowest-ranked applicable merge (= earliest learned, i.e. the
        # most-frequent pair seen during training) and fuse it. This is
        # the classical BPE inference rule: it guarantees the same
        # tokenisation regardless of where you start scanning.
        rank: dict[tuple[int, int], int] = {
            (a, b): i for i, (a, b, _new) in enumerate(self.merges)
        }
        new_id_for: dict[tuple[int, int], int] = {
            (a, b): new for (a, b, new) in self.merges
        }
        while True:
            best_rank = -1
            best_pair: tuple[int, int] | None = None
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                r = rank.get(pair)
                if r is not None and (best_pair is None or r < best_rank):
                    best_rank = r
                    best_pair = pair
            if best_pair is None:
                break
            new_id = new_id_for[best_pair]
            ids = self._merge_pair_in_sequence(ids, best_pair[0], best_pair[1], new_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode ids back to a string. Lossless for content tokens.

        Special-token ids are dropped silently (their string form is the
        responsibility of the higher-level `PluginferTokenizer.decode`).
        Out-of-vocabulary ids (which a model trained with `vocab_size`
        larger than the tokenizer's actual vocab can emit, since the
        tokenizer terminates BPE training early when no pair exceeds
        frequency 1) are also dropped. UTF-8 decode errors are mapped
        to U+FFFD via `errors='replace'` so a partial multi-byte
        sequence never raises.
        """
        chunks: list[bytes] = []
        for i in ids:
            piece = self.vocab.get(i)
            if piece:  # skip empty (special) and absent (OOV) entries
                chunks.append(piece)
        return b"".join(chunks).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the tokenizer state to a JSON file.

        We store ONLY the merges; the vocab is derivable. Specials are
        stored by ordered name list so a future bump to N_SPECIAL is
        backwards-detectable.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "format_version": TOKENIZER_FORMAT_VERSION,
            "vocab_size": self.vocab_size,
            "n_special": N_SPECIAL,
            "byte_offset": BYTE_OFFSET,
            "special_token_names": list(SPECIAL_TOKEN_NAMES),
            "merges": [[a, b, new] for (a, b, new) in self.merges],
            "training_time_seconds": self.training_time_seconds,
        }
        path.write_text(json.dumps(body, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETrainer":
        path = Path(path)
        body = json.loads(path.read_text(encoding="utf-8"))
        if body.get("format_version") != TOKENIZER_FORMAT_VERSION:
            raise ValueError(
                f"unsupported tokenizer format_version "
                f"{body.get('format_version')!r}; expected "
                f"{TOKENIZER_FORMAT_VERSION}"
            )
        if body.get("special_token_names") != list(SPECIAL_TOKEN_NAMES):
            raise ValueError(
                "special_token_names mismatch: refusing to load a tokenizer "
                "produced with a different special-token vocabulary"
            )
        trainer = cls(vocab_size=int(body["vocab_size"]))
        trainer.merges = [tuple(m) for m in body["merges"]]
        for a, b, new in trainer.merges:
            trainer._merge_lookup[(a, b)] = new
            trainer.vocab[new] = trainer.vocab[a] + trainer.vocab[b]
        trainer.training_time_seconds = float(body.get("training_time_seconds", 0.0))
        return trainer

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def actual_vocab_size(self) -> int:
        return len(self.vocab)

    def compression_ratio(self, texts: Iterable[str]) -> float:
        """Average chars-per-token on a held-out set. Higher = better BPE.

        Excludes empty strings; raises if no usable inputs.
        """
        total_chars = 0
        total_tokens = 0
        for t in texts:
            if not t:
                continue
            total_chars += len(t)
            total_tokens += len(self.encode(t))
        if total_tokens == 0:
            raise ValueError("no usable held-out texts for compression measurement")
        return total_chars / total_tokens
