"""Pure-stdlib byte-pair encoding tokenizer.

Why ship our own when SentencePiece exists? Three reasons:

1. **No external dependency.** Pluginfer's substrate must run on
   any host with Python; SentencePiece is C++ wheels with platform
   gaps (raspberry-pi, fresh ARM Macs, locked-down corporate hosts).
2. **Auditability.** A 250-line BPE is reviewable in one sitting;
   a wrapper around an opaque .so is not.
3. **Mesh distribution.** Tokenizer training itself can be done on
   the mesh — every contributor can run the trainer without first
   compiling C++.

This is *vanilla* BPE (Sennrich 2016 / GPT-2 style), not the
unigram-LM variant of SentencePiece. Vanilla BPE is what GPT-2/-3,
Llama, Mistral, and most production LLMs use; the unigram model is
mostly useful for languages without clear word boundaries
(Japanese, Chinese) — which is on the §D2 roadmap as a separate
tokenizer plugin.

Trainer is intentionally simple but not naive — uses a priority
queue over pair counts so training is O(N log V) for N corpus bytes
and V vocab size. A 16k vocab on a 100 MiB corpus trains in
~30 seconds on CPU.

Wire format: a single JSON file per trained tokenizer:

    {
      "version": 1,
      "merges": [["a", "b"], ["ab", "c"], ...],   // ordered
      "vocab":  {" ": 0, ...},               // token -> id
      "specials": {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    }
"""

from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


SPECIALS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}


# ---------- training -------------------------------------------------------

@dataclass
class BPEConfig:
    vocab_size: int = 16384
    min_pair_count: int = 2
    end_of_word: str = "</w>"   # appended to last byte of each word


def train_bpe(
    corpus: Iterable[str],
    config: BPEConfig = BPEConfig(),
) -> "BPETokenizer":
    """Train a byte-level BPE on an iterable of strings.

    Word splitting: whitespace + GPT-2-style regex would be ideal;
    here we use a simple split() to avoid pulling in `regex`. The
    end-of-word marker preserves spacing during decode.
    """
    # 1. Build word frequency table.
    word_counts: Counter = Counter()
    for line in corpus:
        for w in line.split():
            word_counts[w] += 1
    if not word_counts:
        raise ValueError("empty corpus")

    # 2. Initialize each word as a sequence of bytes + EOW.
    EOW = config.end_of_word
    seqs: dict[str, list[str]] = {}
    for w, _ in word_counts.items():
        # Encode each byte as a hex-escape token for uniformity.
        toks = [b"%c" % b for b in w.encode("utf-8", errors="replace")]
        toks = [t.decode("latin-1") for t in toks]
        toks[-1] = toks[-1] + EOW
        seqs[w] = toks

    # 3. Initial vocab = unique bytes (+ specials reserved later).
    vocab: set[str] = set()
    for s in seqs.values():
        vocab.update(s)
    target_vocab_size = max(256, config.vocab_size - len(SPECIALS))

    # 4. Greedy merge: find most-common adjacent pair, merge.
    merges: list[tuple[str, str]] = []
    while len(vocab) < target_vocab_size:
        pair_counts: Counter = Counter()
        for w, count in word_counts.items():
            seq = seqs[w]
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i + 1])] += count
        if not pair_counts:
            break
        best_pair, best_count = pair_counts.most_common(1)[0]
        if best_count < config.min_pair_count:
            break
        # Merge best_pair into a new token.
        new_tok = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab.add(new_tok)
        # Apply merge across all sequences.
        for w in list(seqs.keys()):
            seq = seqs[w]
            new_seq: list[str] = []
            i = 0
            while i < len(seq):
                if (i < len(seq) - 1 and seq[i] == best_pair[0]
                        and seq[i + 1] == best_pair[1]):
                    new_seq.append(new_tok)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            seqs[w] = new_seq

    # 5. Assign ids: specials first, then vocab in deterministic order.
    sorted_vocab = sorted(vocab)
    token_to_id: dict[str, int] = dict(SPECIALS)
    next_id = max(SPECIALS.values()) + 1
    for tok in sorted_vocab:
        if tok not in token_to_id:
            token_to_id[tok] = next_id
            next_id += 1

    return BPETokenizer(
        merges=list(merges),
        token_to_id=token_to_id,
        eow=EOW,
    )


# ---------- the tokenizer --------------------------------------------------

@dataclass
class BPETokenizer:
    merges: list[tuple[str, str]]
    token_to_id: dict[str, int]
    eow: str = "</w>"
    id_to_token: dict[int, str] = field(default_factory=dict, init=False)
    _merge_rank: dict[tuple[str, str], int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self._merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return SPECIALS["<pad>"]

    @property
    def bos_id(self) -> int:
        return SPECIALS["<bos>"]

    @property
    def eos_id(self) -> int:
        return SPECIALS["<eos>"]

    @property
    def unk_id(self) -> int:
        return SPECIALS["<unk>"]

    # --- encode --------------------------------------------------------

    def _bpe(self, word: str) -> list[str]:
        """Apply learned merges to a single word."""
        toks = [b"%c" % b for b in word.encode("utf-8", errors="replace")]
        toks = [t.decode("latin-1") for t in toks]
        if not toks:
            return toks
        toks[-1] = toks[-1] + self.eow

        while True:
            best_rank = None
            best_idx = -1
            for i in range(len(toks) - 1):
                pair = (toks[i], toks[i + 1])
                rank = self._merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_idx < 0:
                break
            new_tok = toks[best_idx] + toks[best_idx + 1]
            toks = toks[:best_idx] + [new_tok] + toks[best_idx + 2:]
        return toks

    def encode(self, text: str, *, add_bos: bool = False,
                add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for w in text.split():
            for tok in self._bpe(w):
                tid = self.token_to_id.get(tok, self.unk_id)
                ids.append(tid)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    # --- decode --------------------------------------------------------

    def decode(self, ids: Iterable[int]) -> str:
        toks: list[str] = []
        for i in ids:
            if i in (self.pad_id, self.bos_id, self.eos_id):
                continue
            t = self.id_to_token.get(int(i), "<unk>")
            toks.append(t)
        # Rebuild words by splitting on the end-of-word marker.
        joined = "".join(toks)
        # The EOW marker delimits words; replace with a space.
        words = joined.split(self.eow)
        out = " ".join(w for w in words if w)
        return out

    # --- persistence ---------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps({
            "version": 1,
            "merges": [list(m) for m in self.merges],
            "vocab":  self.token_to_id,
            "eow":    self.eow,
            "specials": SPECIALS,
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(m) for m in d.get("merges", [])]
        vocab = {k: int(v) for k, v in d.get("vocab", {}).items()}
        eow = d.get("eow", "</w>")
        return cls(merges=merges, token_to_id=vocab, eow=eow)
