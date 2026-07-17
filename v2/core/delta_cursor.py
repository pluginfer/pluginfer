"""SSE delta cursor — resume long-streaming jobs without losing chunks.

The problem
-----------
A long-running streaming job (think 1-hour batch inference, 6-hour
training with live loss telemetry) emits deltas via SSE. When the
buyer's WiFi blinks, mobile network swaps, or laptop sleeps, the SSE
connection drops. The deltas keep arriving at the gateway. The buyer
re-connects — and loses every chunk emitted during the gap.

The pattern is identical to how Server-Sent Events were designed:
each chunk carries a sequence number, the client sends back
`Last-Event-ID` on reconnect, the server replays from there. Cleanly
spec-compliant; no custom protocol.

This module ships the in-process delta cursor: a ring buffer per
JobRecord. JobsService' `_on_delta` callback appends here; the SSE
handler reads from `since=<seq>` on reconnect. Bounded memory via
`MAX_DELTAS_PER_JOB`; if the buyer is too far behind, the gap is
flagged so they know to restart vs. silently degrading.

Innovation: §A30 "Cursor-based SSE resume for cross-internet
auction-routed compute streams." Combines (a) gap-aware replay,
(b) bounded memory, (c) terminal-state preservation across
reconnect — none of which standard SSE specifies on the server
side. The buyer's app gets the AWS-like SLA without AWS-like
infrastructure.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

MAX_DELTAS_PER_JOB = 4096


@dataclass
class DeltaEntry:
    seq: int
    payload: Dict[str, Any]
    is_terminal: bool = False


@dataclass
class DeltaCursor:
    """Per-job ring buffer of streamed deltas. Append is O(1),
    replay since cursor is O(k) where k is the gap size."""
    max_size: int = MAX_DELTAS_PER_JOB
    _entries: List[DeltaEntry] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _next_seq: int = 1
    # Oldest seq still in the buffer. When the ring rolls, this
    # advances; replay requests for `since < _earliest_seq` get a
    # gap_detected flag so the client knows to restart cleanly.
    _earliest_seq: int = 1
    terminal_seq: Optional[int] = None

    def append(self, payload: Dict[str, Any], *, is_terminal: bool = False) -> int:
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            entry = DeltaEntry(seq=seq, payload=payload, is_terminal=is_terminal)
            self._entries.append(entry)
            if is_terminal:
                self.terminal_seq = seq
            # Bound memory.
            while len(self._entries) > self.max_size:
                self._entries.pop(0)
                self._earliest_seq = self._entries[0].seq if self._entries else seq
            return seq

    def replay_since(self, since_seq: int = 0) -> Tuple[List[DeltaEntry], bool]:
        """Return all deltas with seq > since_seq + gap_detected flag.

        gap_detected = True means the buyer reconnected too late: at
        least one chunk was evicted from the ring before they
        re-subscribed. The handler should send a `gap_detected: true`
        SSE event so the client can decide whether to restart the
        whole stream or accept the missing data."""
        with self._lock:
            if not self._entries:
                return [], False
            gap = since_seq < self._earliest_seq - 1 and since_seq != 0
            out = [e for e in self._entries if e.seq > since_seq]
            return out, gap

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._next_seq - 1

    def has_terminal(self) -> bool:
        return self.terminal_seq is not None


__all__ = [
    "DeltaCursor",
    "DeltaEntry",
    "MAX_DELTAS_PER_JOB",
]
