"""Token counting for pre-flight budget holds.

The audit flagged chars/4 as too crude — a hold sized on it can be off
by 2-3x for code, CJK, or heavy punctuation, which means a budget can
be over- or under-held. This module upgrades the ESTIMATE without
adding a hard dependency:

  * ``tiktoken`` backend when the package is installed (accurate BPE
    counts for OpenAI-family models; a reasonable proxy for others).
  * an improved heuristic fallback otherwise — word + punctuation +
    CJK-aware, closer than chars/4 across mixed text.

It is still an ESTIMATE used only to size the hold; SETTLEMENT always
uses the upstream's own reported usage, so an imperfect estimate never
mis-bills — it only affects how much headroom is reserved. The backend
in use is reported so nobody assumes more precision than they have.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_ENCODER = None
_BACKEND = "heuristic"


def _load_tiktoken() -> None:
    global _ENCODER, _BACKEND
    if _ENCODER is not None or _BACKEND == "tiktoken-failed":
        return
    try:
        import tiktoken
        _ENCODER = tiktoken.get_encoding("cl100k_base")
        _BACKEND = "tiktoken/cl100k_base"
    except Exception:
        _BACKEND = "tiktoken-failed"


_WORD = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")
_CJK = re.compile(r"[　-鿿가-힣]")


def count_text(text: str) -> int:
    """Estimated token count for one string."""
    if not text:
        return 0
    _load_tiktoken()
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text))
        except Exception:
            pass
    # Heuristic: word/punct tokens, +1 per CJK char (each ~1 token),
    # floored so it never under-counts to zero.
    cjk = len(_CJK.findall(text))
    non_cjk = _CJK.sub("", text)
    pieces = len(_WORD.findall(non_cjk))
    # BPE tends to split long words; nudge up for very long tokens.
    return max(1, pieces + cjk)


def backend() -> str:
    _load_tiktoken()
    return _BACKEND


def count_request(body: Dict[str, Any]) -> int:
    """Estimated INPUT tokens for a chat/messages body — every text
    part across messages + system, plus a small per-message overhead
    (role tags / formatting the API adds)."""
    total = 0
    msgs: List[Any] = body.get("messages") or []
    for m in msgs:
        total += 4  # per-message formatting overhead (OpenAI-ish)
        c = m.get("content")
        if isinstance(c, str):
            total += count_text(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    total += count_text(str(part.get("text", "")))
    sysp = body.get("system")
    if isinstance(sysp, str):
        total += count_text(sysp)
    return max(1, total)
