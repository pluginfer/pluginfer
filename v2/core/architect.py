"""
TaskArchitect (deprecated — superseded by core.task_router.TaskRouter)
======================================================================
Original version had a TODO at the dispatch site:
    # In real imp: NetworkManager.send_task(target, task)
i.e. tasks were "assigned" by writing to a log line and never
actually transmitted to peers.

The real distributed dispatch lives in `core.task_router.TaskRouter`
(K-redundant, brain-gated, majority-vote reduce). This file stays
only because `complete_mesh_controller` imports it; new code should
prefer the router directly.
"""

from __future__ import annotations

import logging
import uuid
import warnings
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TaskArchitect:
    def __init__(self, scout):
        self.scout = scout
        self.active_jobs: Dict[str, List[Any]] = {}
        warnings.warn(
            "TaskArchitect is deprecated; use core.task_router.TaskRouter "
            "for K-redundant, brain-gated distributed dispatch.",
            DeprecationWarning, stacklevel=2,
        )

    def submit_complex_job(self, data: Dict[str, Any]) -> str:
        """Compatibility shim: returns a job_id but does no real sharding."""
        job_id = uuid.uuid4().hex[:8]
        self.active_jobs[job_id] = []
        logger.info("[ARCHITECT-deprecated] job %s registered (no-op shim).", job_id)
        return job_id

    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        return {"status": "deprecated", "use": "core.task_router.TaskRouter"}
