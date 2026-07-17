"""Special-token vocabulary for the Pluginfer brain.

Design choice: the 13 special tokens occupy a contiguous block at the
START of the vocabulary (IDs 0-12). Raw bytes are then placed at IDs
13-268 (byte i -> id i+13), and BPE-learned merges populate IDs 269+.

Why not "bytes at 0-255, specials appended at the end" (the GPT-2
convention)? Two reasons:

1. Specials at the start makes <PAD>=0, which is what every PyTorch
   `pad_token_id` default expects. Saves a footgun.
2. Decode of any token > 12 is unambiguously textual content, so
   filtering specials out before stringification is a single
   `if id < N_SPECIAL: continue`.

The down-side is that byte 0x00 -> id 13 (not id 0). All places that
go bytes <-> ids must agree on the +13 offset; centralised in
`bpe.BYTE_OFFSET`.
"""

from __future__ import annotations

from dataclasses import dataclass

# Ordered list - position in this list = token ID. Order is load-bearing;
# changing it breaks every saved tokenizer file. Append new tokens at the
# end and bump tokenizer.format_version.
SPECIAL_TOKEN_NAMES: tuple[str, ...] = (
    "<PAD>",       # 0  - padding (cross_entropy ignores -100; we shift)
    "<BOS>",       # 1  - begin of sequence
    "<EOS>",       # 2  - end of sequence
    "<MASK>",      # 3  - MLM mask token (reserved; not used in causal LM)
    "<SEP>",       # 4  - segment separator (between packed examples)
    "<JOB>",       # 5  - start of job spec
    "<PROVIDER>",  # 6  - start of provider record
    "<GPU>",       # 7  - GPU class marker
    "<PRICE>",     # 8  - price value follows
    "<VRAM>",      # 9  - VRAM value follows
    "<RUNTIME>",   # 10 - runtime estimate follows
    "<QUALITY>",   # 11 - quality score follows
    "<ANOMALY>",   # 12 - anomaly flag follows
)

SPECIAL_TOKENS: dict[str, int] = {name: i for i, name in enumerate(SPECIAL_TOKEN_NAMES)}
N_SPECIAL = len(SPECIAL_TOKEN_NAMES)


@dataclass(frozen=True)
class SpecialTokens:
    """Convenience accessor object - avoids string-key lookups in hot paths."""

    PAD: int = SPECIAL_TOKENS["<PAD>"]
    BOS: int = SPECIAL_TOKENS["<BOS>"]
    EOS: int = SPECIAL_TOKENS["<EOS>"]
    MASK: int = SPECIAL_TOKENS["<MASK>"]
    SEP: int = SPECIAL_TOKENS["<SEP>"]
    JOB: int = SPECIAL_TOKENS["<JOB>"]
    PROVIDER: int = SPECIAL_TOKENS["<PROVIDER>"]
    GPU: int = SPECIAL_TOKENS["<GPU>"]
    PRICE: int = SPECIAL_TOKENS["<PRICE>"]
    VRAM: int = SPECIAL_TOKENS["<VRAM>"]
    RUNTIME: int = SPECIAL_TOKENS["<RUNTIME>"]
    QUALITY: int = SPECIAL_TOKENS["<QUALITY>"]
    ANOMALY: int = SPECIAL_TOKENS["<ANOMALY>"]


def is_special(token_id: int) -> bool:
    return 0 <= token_id < N_SPECIAL


def name_of(token_id: int) -> str | None:
    if 0 <= token_id < N_SPECIAL:
        return SPECIAL_TOKEN_NAMES[token_id]
    return None
