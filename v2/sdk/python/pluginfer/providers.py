"""Providers API surface."""

from __future__ import annotations

from typing import List

from ._http import HttpSession
from .types import Provider


class ProvidersAPI:
    def __init__(self, session: HttpSession) -> None:
        self._s = session

    def list(self) -> List[Provider]:
        return [
            Provider(
                pubkey=p["pubkey"], kind=p["kind"],
                quality_score=p.get("quality_score", 0.0),
                region=p.get("region", "unknown"),
            )
            for p in self._s.get("/v1/providers")
        ]
