"""SDK exception hierarchy.

`PluginferError` is the base; everything else is a subclass so callers
can catch broadly or narrowly. All errors carry the request_id (when
available) so server-side logs can be correlated."""

from __future__ import annotations

from typing import Any, Optional


class PluginferError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.body = body


class AuthenticationError(PluginferError):
    """401: missing / invalid credentials, or expired wallet session."""


class JobNotFoundError(PluginferError):
    """404: requested job doesn't exist or belongs to another identity."""


class InsufficientBalanceError(PluginferError):
    """402-style: caller's PLG balance won't cover the locked auction
    price."""


class RateLimitError(PluginferError):
    """429: token bucket exhausted. `retry_after_sec` carries the
    server-side hint."""

    def __init__(self, message: str, *, retry_after_sec: float = 1.0, **kw) -> None:
        super().__init__(message, **kw)
        self.retry_after_sec = retry_after_sec
