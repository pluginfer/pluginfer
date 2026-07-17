"""Compute jobs: submit, fetch, cancel, SSE-stream."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from ..middleware.auth import require_auth
from ..schemas import JobCreate, JobInfo, JobResult

router = APIRouter(prefix="/v1/jobs", tags=["compute"])


@router.post("", response_model=JobInfo, status_code=status.HTTP_202_ACCEPTED)
async def submit_job(
    body: JobCreate,
    request: Request,
    identity: str = Depends(require_auth),
) -> JobInfo:
    svc = request.app.state.jobs
    rec = await svc.submit(
        kind=body.kind,
        payload=body.payload,
        cost_ceiling_usd=body.cost_ceiling_usd,
        latency_ceiling_ms=body.latency_ceiling_ms,
        privacy_class=body.privacy_class,
        quality_floor=body.quality_floor,
        requester_identity=identity,
    )
    return JobInfo.model_validate(rec.to_info())


@router.get("/{job_id}", response_model=JobInfo)
def get_job(job_id: str, request: Request, identity: str = Depends(require_auth)) -> JobInfo:
    rec = request.app.state.jobs.get(job_id)
    if rec is None:
        raise HTTPException(404, "job_not_found")
    if rec.requester_identity != identity:
        raise HTTPException(403, "forbidden")
    return JobInfo.model_validate(rec.to_info())


@router.get("/{job_id}/result", response_model=JobResult)
def get_job_result(job_id: str, request: Request, identity: str = Depends(require_auth)) -> JobResult:
    rec = request.app.state.jobs.get(job_id)
    if rec is None:
        raise HTTPException(404, "job_not_found")
    if rec.requester_identity != identity:
        raise HTTPException(403, "forbidden")
    return JobResult.model_validate(rec.to_result())


@router.delete("/{job_id}", status_code=status.HTTP_200_OK)
async def cancel_job(job_id: str, request: Request, identity: str = Depends(require_auth)) -> dict:
    rec = request.app.state.jobs.get(job_id)
    if rec is None:
        raise HTTPException(404, "job_not_found")
    if rec.requester_identity != identity:
        raise HTTPException(403, "forbidden")
    cancelled = await request.app.state.jobs.cancel(job_id)
    return {"job_id": job_id, "cancelled": cancelled, "state": rec.state}


@router.get("/{job_id}/stream")
async def stream_job(job_id: str, request: Request, identity: str = Depends(require_auth)):
    svc = request.app.state.jobs
    rec = svc.get(job_id)
    if rec is None:
        raise HTTPException(404, "job_not_found")
    if rec.requester_identity != identity:
        raise HTTPException(403, "forbidden")
    q = await svc.subscribe(job_id)

    async def event_gen():
        try:
            terminal = {"completed", "failed", "cancelled", "timeout"}
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "{}"}
                    continue
                yield {"event": msg["event"], "data": json.dumps(msg)}
                cur = svc.get(job_id)
                if cur is not None and cur.state in terminal:
                    break
        finally:
            svc.unsubscribe(job_id, q)

    return EventSourceResponse(event_gen())
