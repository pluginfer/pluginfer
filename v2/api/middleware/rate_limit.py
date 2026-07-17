"""Token-bucket rate limiter.

In-memory by default — production swaps in a Redis-backed bucket via the
`store` arg without rewriting handlers. Keyed by API key when present,
falling back to client IP. The bucket math is conservative (refill ==
capacity / period) and is monotonic-clock based so DST / wall-clock
adjustments don't open or close the bucket.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


@dataclass
class TokenBucket:
    capacity: float
    refill_per_sec: float
    tokens: float = field(default=0.0)
    last_refill: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.tokens == 0.0:
            self.tokens = self.capacity

    def take(self, n: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = max(0.0, now - self.last_refill)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
        self.last_refill = now
        if self.tokens < n:
            return False
        self.tokens -= n
        return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-key token bucket. `key_fn` extracts the bucket key from the
    request (default: api key in `Authorization: Bearer <k>` header,
    fallback to client.host). Buckets live in `store` (default: dict)."""

    def __init__(
        self,
        app,
        *,
        capacity: float = 60.0,        # 60 requests
        refill_per_sec: float = 1.0,    # ... per second sustained
        key_fn: Optional[Callable[[Request], str]] = None,
        store: Optional[Dict[str, TokenBucket]] = None,
    ) -> None:
        super().__init__(app)
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self.key_fn = key_fn or self._default_key
        self.store: Dict[str, TokenBucket] = store if store is not None else {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _default_key(request: Request) -> str:
        auth = request.headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            return f"key:{auth.split(' ', 1)[1].strip()[:64]}"
        host = request.client.host if request.client else "unknown"
        return f"ip:{host}"

    async def dispatch(self, request: Request, call_next) -> Response:
        key = self.key_fn(request)
        async with self._lock:
            bucket = self.store.get(key)
            if bucket is None:
                bucket = TokenBucket(
                    capacity=self.capacity,
                    refill_per_sec=self.refill_per_sec,
                )
                self.store[key] = bucket
            allowed = bucket.take(1)
        if not allowed:
            return JSONResponse(
                {"error": "rate_limited",
                 "retry_after_sec": 1.0 / self.refill_per_sec},
                status_code=429,
                headers={"Retry-After": str(int(1.0 / self.refill_per_sec))},
            )
        return await call_next(request)
