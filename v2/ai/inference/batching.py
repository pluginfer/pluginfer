"""Continuous batching scaffold for the inference engine.

Real continuous batching requires per-stream KV-cache slots in a
pre-allocated pool, and the ability to insert / evict streams between
decode steps. The current `InferenceEngine.generate()` is single-stream;
this module is the API the FastAPI server uses to enqueue requests.

For CP-AI-5 we ship the simplest possible scheduler: requests are
processed sequentially. The continuous batcher abstraction is here so
the server's request handler doesn't change when the real scheduler
lands.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

from .engine import GenerationParams, InferenceEngine


@dataclass
class GenerationRequest:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    params: GenerationParams = field(default_factory=GenerationParams)


class ContinuousBatcher:
    """Sequential request runner; stand-in for a real continuous batcher.

    The interface is async so the server doesn't need to change when the
    real scheduler lands.
    """

    def __init__(self, engine: InferenceEngine, max_concurrent: int = 1) -> None:
        self.engine = engine
        self.max_concurrent = max_concurrent
        self._lock = asyncio.Lock()

    async def submit(self, req: GenerationRequest) -> str:
        # Single-stream sequential execution under a lock. Real batcher
        # would interleave decode steps across streams.
        async with self._lock:
            return await asyncio.to_thread(
                self.engine.generate, req.prompt, req.params
            )
