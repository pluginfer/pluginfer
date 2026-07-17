"""Bridge provider: any OpenAI-compatible endpoint becomes auction supply.

The concrete target is a Mesh-LLM node (https://github.com/Mesh-LLM/mesh-llm)
— an open-source mesh that pools GPUs across machines and exposes the
pooled capacity as one OpenAI-compatible API (default `:9337/v1`). By
wrapping that endpoint as a `Provider`, an ENTIRE mesh-llm mesh bids in
the Pluginfer auction as a single supplier: Pluginfer supplies the
economics (sealed-bid auction, settlement, signed receipts) on top of
transport/inference layers built by others. The same class bridges LM
Studio, vLLM, llama.cpp-server, text-generation-inference, or another
Pluginfer node — anything speaking `/v1/chat/completions`.

Strategic note (positioning): Pluginfer does not compete with inference
meshes on plumbing; it monetizes and verifies them. This module is that
sentence as code.

Zero-config: `tools/auto_mesh.build_node_app` probes `:9337` at boot and
registers this provider automatically when a mesh-llm node is running
on the same machine (opt-out: PLUGINFER_DISABLE_MESHLLM=1; remote mesh:
PLUGINFER_MESHLLM_URL=http://host:9337/v1).

Honesty rules preserved:
  * probe() must succeed before the provider ever enters an auction —
    a dead endpoint abstains instead of winning and 502ing.
  * Results are hashed + wallet-signed like every other provider, so
    PNIS receipts stamp WHICH upstream model served the completion
    (`meshllm:<model>`); we never claim the compute as our own.
  * Public mesh == PRIVACY_PUBLIC: privacy-routed jobs will never be
    sent to a public mesh-llm swarm.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from core.providers import Bid, JobSpec, Provider, PRIVACY_PUBLIC

logger = logging.getLogger(__name__)

DEFAULT_MESHLLM_URL = "http://127.0.0.1:9337/v1"

# Mesh-llm serves quantized open weights on donated/peer GPUs; the
# marginal cost basis is electricity, so the default bid undercuts
# cloud APIs by construction. Operators override per deployment.
DEFAULT_PRICE_PER_1K_TOK_USD = 0.00005


class MeshLLMProvider(Provider):
    """One OpenAI-compatible endpoint (typically a whole mesh) as a bidder."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_MESHLLM_URL,
        wallet: Optional[Any] = None,
        provider_id: str = "",
        price_per_1k_tok_usd: float = DEFAULT_PRICE_PER_1K_TOK_USD,
        expected_quality: float = 0.68,
        eta_ms: int = 20_000,
        timeout_s: float = 300.0,
        privacy_grade: str = PRIVACY_PUBLIC,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.wallet = wallet
        self.provider_id = provider_id or f"meshllm-{hashlib.sha256(self.base_url.encode()).hexdigest()[:8]}"
        self.price_per_1k_tok_usd = float(price_per_1k_tok_usd)
        self.expected_quality = float(expected_quality)
        self.eta_ms = int(eta_ms)
        self.timeout_s = float(timeout_s)
        self.privacy_grade = privacy_grade
        self._models: List[str] = []
        self._probed_ok = False

    # -- discovery ---------------------------------------------------

    def probe(self) -> bool:
        """GET /models. True (and caches the served model list) when the
        endpoint is alive. Never raises — a dead mesh just abstains."""
        try:
            req = urllib.request.Request(f"{self.base_url}/models", method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
            self._models = [
                m.get("id", "") for m in data.get("data", []) if m.get("id")
            ]
            self._probed_ok = True
            return True
        except Exception as e:
            logger.debug("meshllm probe failed for %s: %s", self.base_url, e)
            self._probed_ok = False
            return False

    @property
    def models(self) -> List[str]:
        return list(self._models)

    # -- auction surface ----------------------------------------------

    def _serves(self, model: str) -> bool:
        if not model:
            return bool(self._models)
        if model in self._models:
            return True
        # mesh-llm's Mixture-of-Agents pseudo-model fans out to every
        # model in the mesh; it is always routable when the mesh is up.
        return model == "mesh"

    def bid(self, job: JobSpec) -> Optional[Bid]:
        if job.kind not in ("inference", "llm.completion", "llm.chat"):
            return None
        if not self._probed_ok and not self.probe():
            return None
        model = str((job.payload or {}).get("model", "") or "")
        if not self._serves(model):
            return None
        max_tokens = (job.payload or {}).get("max_tokens") or 256
        est_price = self.price_per_1k_tok_usd * (int(max_tokens) / 1000.0)
        if est_price > job.cost_ceiling_usd:
            return None
        if self.eta_ms > job.latency_ceiling_ms:
            return None
        return Bid(
            provider_id=self.provider_id,
            price_usd=max(est_price, 1e-8),
            eta_ms=self.eta_ms,
            expected_quality=self.expected_quality,
            privacy_grade=self.privacy_grade,
        )

    def execute(self, job: JobSpec, bid: Bid) -> Dict[str, Any]:
        t0 = time.monotonic()
        payload = job.payload or {}
        model = str(payload.get("model", "") or (self._models[0] if self._models else "mesh"))
        messages = payload.get("messages")
        if not messages:
            messages = [{"role": "user", "content": str(payload.get("prompt", ""))}]
        body: Dict[str, Any] = {"model": model, "messages": messages}
        for k in ("max_tokens", "temperature", "top_p"):
            v = payload.get(k)
            if v is not None:
                body[k] = v
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                resp = json.loads(r.read().decode("utf-8", "replace"))
            text = str(
                (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
            )
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                json.JSONDecodeError, KeyError, IndexError) as e:
            return {
                "status": "error",
                "code": "meshllm_upstream_error",
                "reason": f"{type(e).__name__}: {e}",
                "provider_id": self.provider_id,
                "job_id": job.job_id,
                "refund_eligible": True,
            }
        result_bytes = text.encode("utf-8")
        result_hash = hashlib.sha256(result_bytes).hexdigest()
        sig = ""
        if self.wallet is not None:
            try:
                sig = self.wallet.sign(result_hash)
            except Exception:
                sig = ""
        return {
            "status": "executed",
            "job_id": job.job_id,
            "result_bytes": base64.b64encode(result_bytes).decode("ascii"),
            "result_hash": result_hash,
            "provider_sig": sig,
            "provider_pubkey_pem": getattr(self.wallet, "public_key_pem", ""),
            "execution_ms": (time.monotonic() - t0) * 1000.0,
            "model_id": f"meshllm:{model}",
        }


def autodetect_meshllm(
    *, wallet: Optional[Any] = None, base_url: str = "",
) -> Optional[MeshLLMProvider]:
    """Zero-config detection: returns a probed, ready provider when a
    mesh-llm (or any OpenAI-compatible) endpoint is serving, else None.
    """
    import os
    if os.environ.get("PLUGINFER_DISABLE_MESHLLM") == "1":
        return None
    url = base_url or os.environ.get("PLUGINFER_MESHLLM_URL", DEFAULT_MESHLLM_URL)
    p = MeshLLMProvider(base_url=url, wallet=wallet)
    if p.probe():
        logger.info(
            "mesh-llm endpoint detected at %s serving %d model(s): %s",
            url, len(p.models), ", ".join(p.models[:5]) or "(none listed)",
        )
        return p
    return None
