"""Jobs API surface."""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict, Iterator, Optional

from ._http import HttpSession
from .types import Job, JobResult


class JobsAPI:
    def __init__(self, session: HttpSession) -> None:
        self._s = session

    def submit(
        self,
        *,
        kind: str,
        payload: Optional[Dict[str, Any]] = None,
        cost_ceiling_usd: float = 0.10,
        latency_ceiling_ms: int = 30_000,
        privacy_class: str = "public",
        quality_floor: float = 0.7,
        webhook_url: Optional[str] = None,
    ) -> Job:
        body = {
            "kind": kind,
            "payload": payload or {},
            "cost_ceiling_usd": cost_ceiling_usd,
            "latency_ceiling_ms": latency_ceiling_ms,
            "privacy_class": privacy_class,
            "quality_floor": quality_floor,
            "webhook_url": webhook_url,
        }
        return Job.from_dict(self._s.post("/v1/jobs", json=body))

    def get(self, job_id: str) -> Job:
        return Job.from_dict(self._s.get(f"/v1/jobs/{job_id}"))

    def result(self, job_id: str) -> JobResult:
        return JobResult.from_dict(self._s.get(f"/v1/jobs/{job_id}/result"))

    def cancel(self, job_id: str) -> Dict[str, Any]:
        return self._s.delete(f"/v1/jobs/{job_id}")

    def stream(self, job_id: str) -> Iterator[Dict[str, Any]]:
        for line in self._s.stream_lines(f"/v1/jobs/{job_id}/stream"):
            text = line.decode() if isinstance(line, (bytes, bytearray)) else line
            text = text.strip()
            if not text or text.startswith(":"):
                continue
            if text.startswith("data:"):
                payload = text[len("data:"):].strip()
                if not payload:
                    continue
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    yield {"raw": payload}

    def wait_for(
        self,
        job_id: str,
        *,
        terminal_states: tuple = ("completed", "failed", "cancelled", "timeout"),
        poll_interval_sec: float = 0.5,
        timeout_sec: float = 60.0,
    ) -> Job:
        deadline = time.monotonic() + timeout_sec
        while True:
            j = self.get(job_id)
            if j.state.state in terminal_states:
                return j
            if time.monotonic() >= deadline:
                return j
            time.sleep(poll_interval_sec)

    @staticmethod
    def decode_result(result: JobResult) -> bytes:
        """Convenience: base64-decode the result bytes if present."""
        if not result.result_b64:
            return b""
        return base64.b64decode(result.result_b64)
