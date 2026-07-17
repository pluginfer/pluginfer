"""Tokenize + render structured examples into LM training text.

Job router and provider quality both train via next-token prediction
on a structured rendering of (input, label). The rendering uses
special tokens so the model learns the schema:

  Job router:
    <BOS><JOB>{free-text}<SEP>{label_text}<EOS>

  Provider quality:
    <BOS><PROVIDER>{outcome-sequence}<SEP><QUALITY>{score}<ANOMALY>{flag}<EOS>

This module owns the rendering rules so dataset.py and the inference
server can stay in sync about how to format prompts.
"""

from __future__ import annotations

from typing import Iterable

from ai.tokenizer.special_tokens import SPECIAL_TOKEN_NAMES, SpecialTokens
from ai.tokenizer.tokenizer import PluginferTokenizer


def render_job_router_text(example: dict) -> str:
    """Convert a JobRouterExample dict to LM training text.

    The label_text already contains <GPU>...<VRAM>...<RUNTIME>...<PRICE>...
    special-token markers; we wrap with <JOB>/<SEP> here.
    """
    return f"<JOB>{example['input']}<SEP>{example['label_text']}"


def render_provider_text(example: dict) -> str:
    """Convert a ProviderSequenceExample dict to LM training text.

    The sequence is summarised compactly (per-event line); the label
    follows after <SEP>.
    """
    seq_lines = [
        f"{e['job_type']} d={e['duration_delta']:+.2f} v={int(e['verified'])} r={e['rep_delta']:+.3f}"
        for e in example["input"]
    ]
    body = "; ".join(seq_lines)
    label = example["label"]
    label_text = (
        f"<QUALITY>{label['quality_score']:.2f}"
        f"<ANOMALY>{1 if label['anomaly_flag'] else 0}"
    )
    if label["anomaly_reason"]:
        label_text += f" reason={label['anomaly_reason']}"
    return f"<PROVIDER>{body}<SEP>{label_text}"


def _split_specials(text: str) -> list[str]:
    """Split text into a list of (special-token-name | plain-text) pieces.

    Used by the preprocessor so we can keep specials as single tokens
    even when they appear inside a rendered string.
    """
    pieces: list[str] = []
    cursor = 0
    while cursor < len(text):
        # Find the earliest special-token marker
        next_idx = -1
        next_name: str | None = None
        for name in SPECIAL_TOKEN_NAMES:
            idx = text.find(name, cursor)
            if idx != -1 and (next_idx == -1 or idx < next_idx):
                next_idx = idx
                next_name = name
        if next_idx == -1:
            pieces.append(text[cursor:])
            break
        if next_idx > cursor:
            pieces.append(text[cursor:next_idx])
        pieces.append(next_name or "")
        cursor = next_idx + len(next_name or "")
    return [p for p in pieces if p]


class Preprocessor:
    """Renders structured examples to id sequences with special-token-aware encoding."""

    def __init__(self, tokenizer: PluginferTokenizer) -> None:
        self.tk = tokenizer
        self._special_set = set(SPECIAL_TOKEN_NAMES)
        self.specials = SpecialTokens()

    def encode_text_with_specials(self, text: str) -> list[int]:
        """Encode a string that may contain literal special-token markers.

        Splits on the special names so each marker becomes a single id
        rather than being decomposed by BPE.
        """
        ids: list[int] = []
        for piece in _split_specials(text):
            if piece in self._special_set:
                ids.append(self.tk.id_for(piece))
            else:
                ids.extend(self.tk.bpe.encode(piece))
        return ids

    def encode_example_for_lm(
        self,
        example: dict,
        kind: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        if kind == "job_router":
            text = render_job_router_text(example)
        elif kind == "provider_quality":
            text = render_provider_text(example)
        else:
            raise ValueError(f"unsupported kind for LM rendering: {kind}")
        ids: list[int] = []
        if add_bos:
            ids.append(self.specials.BOS)
        ids.extend(self.encode_text_with_specials(text))
        if add_eos:
            ids.append(self.specials.EOS)
        return ids

    def pack_into_context(
        self,
        sequences: Iterable[list[int]],
        context_length: int,
    ) -> list[list[int]]:
        """Pack variable-length sequences into fixed `context_length` chunks.

        We do NOT cross the EOS boundary - each packed chunk starts with
        a BOS and ends at most at the next EOS or the chunk boundary. If
        a single sequence is larger than context_length we drop the
        overflow rather than splitting (preserves cleanliness for
        multi-task training).
        """
        chunks: list[list[int]] = []
        buf: list[int] = []
        for seq in sequences:
            if len(seq) > context_length:
                # Single overflow: keep the head only.
                seq = seq[:context_length]
            if len(buf) + len(seq) > context_length:
                # Pad current buf to context_length and start a new one
                buf = buf + [self.specials.PAD] * (context_length - len(buf))
                chunks.append(buf)
                buf = []
            buf.extend(seq)
        if buf:
            buf = buf + [self.specials.PAD] * (context_length - len(buf))
            chunks.append(buf)
        return chunks
