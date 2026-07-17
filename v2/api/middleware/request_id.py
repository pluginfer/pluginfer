"""Request-ID propagation: every request gets an X-Request-ID and every
response carries one. Echo the inbound header if present, generate a
fresh uuid4 otherwise. Used by structured logging and SDK error
correlation."""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get(HEADER) or uuid.uuid4().hex
        # Stash on state so handlers can correlate logs.
        request.state.request_id = rid
        response = await call_next(request)
        response.headers[HEADER] = rid
        return response
