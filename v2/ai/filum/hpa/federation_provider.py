"""LocalFederationProvider — bridge between the §J1 ModelFederation and
the `core.providers.Provider` ABC used by the auction layer.

This is the missing wire that closes "submit a job → it actually runs."
Today's `cli.py:_job_submit_via_auction` calls `auction.run(spec)` and
prints the winning bid, then ends with: *"execution dispatch happens
via the broker layer next session."* That is the gap. With this
provider registered, the same `Auction.run(spec)` returns a winner
that has a real `.execute(spec, bid)` method which (a) calls the
local federation, (b) hashes the output, (c) signs it (if a wallet is
provided) and (d) returns the `core.providers` standard response dict.

The same `JobsService` that the FastAPI router uses can therefore be
driven from the CLI without HTTP — same auction, same execution path,
same settlement contract. No fork.

Design rules:
  * The provider only bids on `kind in {"inference", "embed", "chat"}`.
    Training / fine-tune jobs need a different surface (mesh dispatch
    via TaskRouter); they're left for the `MeshGPUProvider` path.
  * The bid uses the federation's *cheapest available local backend*
    (Filum-Genesis if present; Ollama if running). It never bids
    cloud-API prices unless explicitly told to (cloud bids belong to
    `_CloudLLMProvider` so the auction can compare apples-to-apples
    by provider class).
  * Execution returns the standard `{status, result_text, result_b64,
    result_hash, ...}` dict so `JobsService._run_job` can settle it
    just like any other Provider.
  * If a wallet is passed, the result hash is signed and the pubkey
    PEM is returned alongside (matches `MeshGPUProvider.execute`).
    If no wallet, the response is unsigned but still has the result
    hash so the receiver can verify content integrity.

This provider is the load-bearing link for the local-only / no-internet
case. With Ollama installed (or Filum-Genesis trained), a user can
submit a job through the CLI and watch it round-trip without depending
on a deployed seed-relay or any peer.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from core.providers import (
    PRIVACY_PRIVATE,
    PRIVACY_PUBLIC,
    Bid,
    JobSpec,
    Provider,
)

logger = logging.getLogger(__name__)


_INFERENCE_KINDS = frozenset({"inference", "embed", "chat"})


@dataclass
class LocalFederationProvider(Provider):
    """A Provider that runs jobs through the local §J1 ModelFederation.

    Construction is lazy — the federation isn't probed until a job is
    submitted, so registering this provider is cheap.

    Args:
        provider_id: stable identifier used in bids and receipts.
            Defaults to ``"local-federation"``.
        wallet: optional ``core.tokenomics.Wallet`` used to ECDSA-sign
            the result hash. When None, the result is returned unsigned.
        federation_factory: optional callable that returns a
            ``ModelFederation`` instance. Default constructs a fresh
            federation with receipts enabled.
        privacy_grade: defaults to ``"private"`` because local-only
            execution is structurally more private than cloud API
            dispatch. Sensitive jobs that demand TEE-attested execution
            still won't match this provider (no TEE attestation).
        base_eta_ms: rough latency estimate used in bids. Real elapsed
            time is reported in the result dict.
        base_quality: self-reported quality floor of the federation
            (Filum + Ollama). 0.75 default — local Llama 8B is
            substantially weaker than GPT-4o on hard reasoning, so we
            don't claim 0.9+. Real quality comes from the §D1 receipt
            log in steady state.
    """

    provider_id: str = "local-federation"
    wallet: Optional[Any] = None
    federation_factory: Optional[Callable[[], Any]] = None
    privacy_grade: str = PRIVACY_PRIVATE
    base_eta_ms: int = 4_000
    base_quality: float = 0.75
    base_price_per_1k_tok_usd: float = 0.0
    _federation: Optional[Any] = field(default=None, init=False, repr=False)
    _last_probe_ts: float = field(default=0.0, init=False, repr=False)
    _last_probe_ok: bool = field(default=False, init=False, repr=False)

    def _ensure_federation(self):
        if self._federation is not None:
            return self._federation
        if self.federation_factory is not None:
            self._federation = self.federation_factory()
        else:
            from .model_federation import FederationConfig, ModelFederation
            self._federation = ModelFederation(
                config=FederationConfig(issue_receipts=True),
            )
        return self._federation

    def _has_any_backend(self) -> bool:
        # Cache for 5s — federation probes hit localhost so this is cheap,
        # but we don't need to hit Ollama on every bid call.
        now = time.monotonic()
        if now - self._last_probe_ts < 5.0:
            return self._last_probe_ok
        try:
            fed = self._ensure_federation()
            self._last_probe_ok = bool(fed.list_available())
        except Exception as e:
            logger.debug("federation probe failed: %s", e)
            self._last_probe_ok = False
        self._last_probe_ts = now
        return self._last_probe_ok

    def bid(self, job: JobSpec) -> Optional[Bid]:
        if job.kind not in _INFERENCE_KINDS:
            return None
        if not self._has_any_backend():
            return None
        approx_tokens = float(job.payload.get("max_tokens", 200))
        # Local execution is free at the marginal token level (the user
        # already paid for the GPU). We bid the configured base price
        # so the auction can still rank vs paid providers when both
        # exist; for the local-only case the price is 0.
        total_price = self.base_price_per_1k_tok_usd * (approx_tokens / 1000.0)
        return Bid(
            provider_id=self.provider_id,
            price_usd=total_price,
            eta_ms=self.base_eta_ms,
            expected_quality=self.base_quality,
            privacy_grade=self.privacy_grade,
            evidence={"backend": "local_federation"},
        )

    def execute(self, job: JobSpec, bid: Bid) -> Dict[str, Any]:
        from .model_federation import GenerationRequest

        prompt = str(
            job.payload.get("prompt") or job.payload.get("input") or ""
        )
        if not prompt:
            return {
                "status": "error",
                "code": "no_prompt",
                "provider_id": self.provider_id,
                "reason": "job.payload must include 'prompt' or 'input'",
                "refund_eligible": True,
            }
        max_tokens = int(job.payload.get("max_tokens", 256))
        privacy = self._map_privacy(job.privacy_class)
        req = GenerationRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            privacy_mode=privacy,
            require_receipt=True,
        )
        t0 = time.time()
        try:
            fed = self._ensure_federation()
            resp = fed.generate(req)
        except Exception as e:
            return {
                "status": "error",
                "code": "federation_error",
                "provider_id": self.provider_id,
                "reason": f"{type(e).__name__}: {e}",
                "refund_eligible": True,
            }
        text = resp.text or ""
        text_bytes = text.encode("utf-8")
        result_hash = hashlib.sha256(text_bytes).hexdigest()
        out: Dict[str, Any] = {
            "status": "executed",
            "provider_id": self.provider_id,
            "job_id": job.job_id,
            "result_text": text,
            "result_bytes_b64": base64.b64encode(text_bytes).decode("ascii"),
            "result_hash": result_hash,
            "execution_ms": int((time.time() - t0) * 1000),
            "exec_meta": {
                "path": "local_federation",
                "backend": resp.backend_name,
                "model_id": resp.model_id,
                "tokens_generated": resp.metadata.get("tokens_generated"),
                "receipt_id": resp.receipt_id,
            },
        }
        if self.wallet is not None:
            try:
                out["provider_sig"] = self.wallet.sign(result_hash)
                out["provider_pubkey_pem"] = self.wallet.public_key_pem
            except Exception as e:
                # Sign failure shouldn't kill execution; log and continue
                # without a sig (downstream may treat as unsigned).
                logger.warning("result-sign failed: %s", e)
        return out

    @staticmethod
    def _map_privacy(api_privacy: str) -> str:
        """Map auction-layer privacy_class to federation privacy_mode."""
        if api_privacy in ("private", "sensitive"):
            return "LOCAL_ONLY"
        return "HYBRID"
