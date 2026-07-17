"""Minimal waitlist-capture API for the landing page.

The landing.html POSTs `{email}` to `/api/waitlist`. This module is
the FastAPI router that backs it. Two surfaces:
  * append-only JSONL log (default: `~/.pluginfer/waitlist.jsonl`).
  * optional Slack / Discord webhook for instant pings when a new
    email lands.

NOT bundled into the main Pluginfer API by default — operator mounts
it explicitly on the marketing gateway, which is air-gapped from the
core compute / chain.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional
from urllib import error, request

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, EmailStr

router = APIRouter(prefix="/api/waitlist", tags=["growth"])


class WaitlistBody(BaseModel):
    email: EmailStr
    referral: Optional[str] = None
    # Optional self-described hardware / use-case so the operator can
    # prioritise admissions.
    hardware_class: Optional[str] = None
    use_case: Optional[str] = None
    model_config = ConfigDict(extra="ignore")


def _log_path() -> Path:
    return Path(os.environ.get(
        "PLUGINFER_WAITLIST_LOG_PATH",
        str(Path.home() / ".pluginfer" / "waitlist.jsonl"),
    ))


def _maybe_notify(entry: dict) -> None:
    """Best-effort webhook ping. Errors swallowed — never let a
    Slack outage kill the waitlist."""
    hook = os.environ.get("PLUGINFER_WAITLIST_WEBHOOK", "")
    if not hook:
        return
    body = json.dumps({
        "text": f"new waitlist: {entry['email']} "
                f"(use_case={entry.get('use_case', '?')})"
    }).encode("utf-8")
    try:
        req = request.Request(
            hook, data=body,
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=2.0):
            pass
    except (error.URLError, OSError):
        pass


@router.post("", status_code=status.HTTP_201_CREATED)
def signup(body: WaitlistBody, request_: Request) -> dict:
    entry = {
        "email": body.email,
        "referral": body.referral,
        "hardware_class": body.hardware_class,
        "use_case": body.use_case,
        "submitted_at_unix": time.time(),
        "remote_ip": (
            request_.headers.get("cf-connecting-ip")
            or (request_.client.host if request_.client else None)
        ),
        "user_agent": request_.headers.get("user-agent", ""),
    }
    p = _log_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
    except OSError as e:
        raise HTTPException(500, f"waitlist_log_write_failed: {e}") from e
    _maybe_notify(entry)
    return {"queued": True, "email": str(body.email)}
