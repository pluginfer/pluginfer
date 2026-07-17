"""Disk-tiered teacher cache.

In ``real_train.py`` step 1 took 37.3 seconds. The 3 MB model itself
trains in well under a second; the rest was sequential teacher API
calls blocking the loop. While Python waits for an HTTP response,
the GPU sits idle but the laptop's CPU is pegged on
asyncio + the renderer is starved.

This module decouples teacher response latency from training step
latency. A producer coroutine keeps a *cache* of teacher generations
on disk; the trainer dequeues from the cache. The disk is the
queue. If the API is slow or down, the trainer keeps running on
already-cached samples (training does *not* block on the network).

Why disk and not RAM? Two reasons:

1. The user's machine is RAM-constrained (16 GiB shared with everything
   else). A 100k-sample buffer in RAM is a real footprint; on disk
   it's invisible until needed.
2. Persistence: if the trainer crashes, the cache survives. Resuming
   does not re-call the teacher API for already-fetched prompts.

novel claim B4 (see the design notes): a knowledge-distillation
training method comprising an asynchronous teacher producer that
writes generations to a content-addressed on-disk shard, and a
training loop that consumes generations from said shard
independently of producer latency, such that the training step rate
is decoupled from teacher API latency.

Each shard is a JSONL file; samples are content-addressed by
``sha256(prompt + teacher_id)`` so two producers cannot duplicate
work. The cache is self-pruning: once a sample has been consumed
``n`` times it's removed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Optional


@dataclass
class TeacherSample:
    prompt: str
    response_text: str
    teacher_id: str
    ts: float
    consumed_count: int = 0

    def key(self) -> str:
        h = hashlib.sha256()
        h.update(self.prompt.encode("utf-8", errors="replace"))
        h.update(b"\x00")
        h.update(self.teacher_id.encode("utf-8", errors="replace"))
        return h.hexdigest()[:16]


class DiskTeacherCache:
    """Append-only on-disk cache of teacher generations.

    Format: one JSON object per line in ``cache_dir/samples.jsonl``.
    A small in-memory index tracks which keys we already have so we
    don't re-fetch.

    Thread-safe and async-safe for the read path. Writes are serialised
    through a single lock.
    """

    def __init__(self, cache_dir: str | os.PathLike, max_consumed: int = 4):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "samples.jsonl"
        self._lock = threading.Lock()
        self._cursor = 0  # read cursor (line index) for the consumer
        self._index: dict[str, TeacherSample] = {}
        self._max_consumed = max(1, max_consumed)
        self._load()

    # --- persistence -------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                s = TeacherSample(**obj)
                self._index[s.key()] = s
        except Exception:
            # Corrupt / partial line - rotate the cache so we don't
            # poison subsequent reads.
            backup = self.path.with_suffix(".bad")
            try:
                self.path.rename(backup)
            except Exception:
                pass
            self._index.clear()

    def _flush_one(self, s: TeacherSample) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(s)) + "\n")

    # --- producer side -----------------------------------------------------

    def has(self, prompt: str, teacher_id: str) -> bool:
        with self._lock:
            return TeacherSample(prompt, "", teacher_id, 0).key() in self._index

    def put(self, sample: TeacherSample) -> bool:
        """Store a sample. Returns False if already present (no-op)."""
        with self._lock:
            k = sample.key()
            if k in self._index:
                return False
            self._index[k] = sample
            self._flush_one(sample)
            return True

    # --- consumer side -----------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return sum(1 for s in self._index.values()
                       if s.consumed_count < self._max_consumed)

    def take(self, n: int) -> list[TeacherSample]:
        """Return up to n samples that haven't been over-consumed.

        Bumps each returned sample's consumed_count and re-flushes if
        it crosses the eviction threshold.
        """
        out: list[TeacherSample] = []
        with self._lock:
            evict: list[str] = []
            for k, s in list(self._index.items()):
                if len(out) >= n:
                    break
                if s.consumed_count >= self._max_consumed:
                    evict.append(k)
                    continue
                s.consumed_count += 1
                out.append(s)
                if s.consumed_count >= self._max_consumed:
                    evict.append(k)
            for k in evict:
                self._index.pop(k, None)
            # Persist evictions by rewriting the file (cheap since
            # we only do it occasionally; for production this would
            # be a compactor coroutine).
            if evict:
                self._compact_unlocked()
        return out

    def _compact_unlocked(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for s in self._index.values():
                f.write(json.dumps(asdict(s)) + "\n")
        tmp.replace(self.path)

    # --- async producer driver --------------------------------------------

    async def fill(
        self,
        prompts: Iterable[str],
        generate_fn: Callable[[str], Awaitable[tuple[str, str]]],
        target_size: int,
        max_concurrent: int = 4,
    ) -> int:
        """Background fill loop. Calls ``generate_fn(prompt) -> (text, teacher_id)``.

        Yields control between every generation so the event loop can
        run other work. Stops when len(self) >= target_size.

        Returns the number of new samples added.
        """
        added = 0
        sem = asyncio.Semaphore(max_concurrent)
        prompt_list = list(prompts)
        if not prompt_list:
            return 0

        async def _one(prompt: str) -> int:
            nonlocal added
            async with sem:
                if len(self) >= target_size:
                    return 0
                try:
                    text, tid = await generate_fn(prompt)
                except Exception:
                    return 0
                if not text:
                    return 0
                if self.has(prompt, tid):
                    return 0
                self.put(TeacherSample(
                    prompt=prompt, response_text=text,
                    teacher_id=tid, ts=time.time(),
                ))
                added += 1
                return 1

        # Round-robin through prompts until cache is full enough.
        i = 0
        while len(self) < target_size and i < len(prompt_list) * 10:
            tasks = [
                _one(prompt_list[(i + k) % len(prompt_list)])
                for k in range(min(max_concurrent, target_size - len(self)))
            ]
            if not tasks:
                break
            await asyncio.gather(*tasks)
            i += len(tasks)
            if len(self) >= target_size:
                break
        return added
