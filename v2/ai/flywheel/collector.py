"""Inference event collector - JSONL logger for the flywheel dataset.

Every brain inference is logged to a structured JSONL file:
  {timestamp, request_id, module, input, output, latency_ms,
   model_checkpoint_hash}

The collector is per-process. The on-disk file is append-only and
flushed on every event so a process crash loses at most the in-flight
event. File rotation (size or date) is left to deployment - we keep
the writer dumb.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Iterable


@dataclass
class InferenceLogEvent:
    timestamp: float
    request_id: str
    module: str  # "parse_job" | "route_job" | ...
    input: Any
    output: Any
    latency_ms: float = 0.0
    model_checkpoint_hash: str = ""
    extra: dict = field(default_factory=dict)


class FlywheelCollector:
    """Append-only JSONL writer for inference events."""

    def __init__(
        self,
        log_path: str | Path,
        model_checkpoint_hash: str = "",
    ) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_checkpoint_hash = model_checkpoint_hash
        self._lock = Lock()
        self._buf_count = 0

    def log(
        self,
        module: str,
        input_value: Any,
        output_value: Any,
        latency_ms: float = 0.0,
        request_id: str | None = None,
        extra: dict | None = None,
    ) -> str:
        evt = InferenceLogEvent(
            timestamp=time.time(),
            request_id=request_id or str(uuid.uuid4()),
            module=module,
            input=input_value,
            output=output_value,
            latency_ms=latency_ms,
            model_checkpoint_hash=self.model_checkpoint_hash,
            extra=extra or {},
        )
        line = json.dumps(asdict(evt), default=str)
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")
            self._buf_count += 1
        return evt.request_id

    def count(self) -> int:
        return self._buf_count

    def replay(self) -> Iterable[InferenceLogEvent]:
        if not self.log_path.exists():
            return iter(())

        def _gen() -> Iterable[InferenceLogEvent]:
            with self.log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    body = json.loads(line)
                    yield InferenceLogEvent(**body)

        return _gen()
