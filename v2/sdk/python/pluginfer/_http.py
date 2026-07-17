"""Thin HTTP wrapper for the SDK.

We use httpx because:
- it handles HTTP/2 + connection pooling out of the box
- the `transport=` arg lets tests inject ASGITransport(app=...) so the
  full SDK -> API stack runs in-process under pytest with no socket
- timeout + retries are first-class.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from .exceptions import (
    AuthenticationError,
    JobNotFoundError,
    PluginferError,
    RateLimitError,
)


class HttpSession:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        wallet_session: Optional[str] = None,
        timeout: float = 30.0,
        transport: Optional[httpx.BaseTransport] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        headers: Dict[str, str] = {"User-Agent": "pluginfer-python/1.0.0"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if wallet_session:
            headers["X-Pluginfer-Session"] = wallet_session
        if client is not None:
            self._client = client
            for k, v in headers.items():
                self._client.headers.setdefault(k, v)
        else:
            self._client = httpx.Client(
                base_url=base_url.rstrip("/"),
                timeout=timeout,
                headers=headers,
                transport=transport,
            )

    @property
    def base_url(self) -> str:
        return str(self._client.base_url)

    def set_session(self, session_id: str) -> None:
        self._client.headers["X-Pluginfer-Session"] = session_id

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    def _check(self, r: httpx.Response) -> Any:
        rid = r.headers.get("X-Request-ID")
        if r.status_code == 200 or r.status_code == 202:
            return r.json() if r.content else {}
        if r.status_code == 401:
            raise AuthenticationError(
                _msg(r, "authentication_failed"),
                status_code=401, request_id=rid, body=_safe_body(r),
            )
        if r.status_code == 404:
            raise JobNotFoundError(
                _msg(r, "not_found"),
                status_code=404, request_id=rid, body=_safe_body(r),
            )
        if r.status_code == 429:
            try:
                retry_after = float(r.headers.get("Retry-After") or 1.0)
            except ValueError:
                retry_after = 1.0
            raise RateLimitError(
                "rate_limited",
                retry_after_sec=retry_after,
                status_code=429, request_id=rid, body=_safe_body(r),
            )
        raise PluginferError(
            _msg(r, f"http_{r.status_code}"),
            status_code=r.status_code, request_id=rid, body=_safe_body(r),
        )

    def get(self, path: str, **kw) -> Any:
        return self._check(self._client.get(path, **kw))

    def post(self, path: str, *, json: Any = None, **kw) -> Any:
        return self._check(self._client.post(path, json=json, **kw))

    def delete(self, path: str, **kw) -> Any:
        return self._check(self._client.delete(path, **kw))

    def stream_lines(self, path: str):
        with self._client.stream("GET", path) as r:
            if r.status_code != 200:
                self._check(r)
            for line in r.iter_lines():
                if line:
                    yield line


def _safe_body(r: httpx.Response) -> Any:
    try:
        return r.json()
    except Exception:
        return r.text[:512] if r.text else None


def _msg(r: httpx.Response, default: str) -> str:
    try:
        d = r.json()
        if isinstance(d, dict):
            return str(d.get("detail") or d.get("error") or default)
    except Exception:
        pass
    return default
