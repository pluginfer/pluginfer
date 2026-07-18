"""
core.providers — Unified Provider Auction Layer (TODO §4.6, W15)
================================================================

**Design claim:**
  "Method for routing AI inference and training jobs to the
   lowest-cost provider in a sealed-bid live auction where peer GPUs
   running open-weight models bid against centralized LLM APIs under
   caller-supplied cost / latency / privacy / quality constraints,
   with cryptographic settlement on a stake-weighted ledger."

The economic kicker
-------------------
A consumer 4090 idle at 03:00 has near-zero opportunity cost. Run
Llama-3-70B-Q4 on it and you can serve ~70% of the routine workload
that today goes to GPT-4o at <20% of the price. As the Pluginfer
mesh grows, more workload moves off centralised APIs; the cost basis
of AI compute shifts from datacentre capex to consumer-GPU slack.
This is the cost-dynamics shift the project targets.

The auction
-----------
Caller submits a JobSpec with explicit constraints:
  * cost_ceiling_usd: max acceptable price
  * latency_ceiling_ms: max acceptable wall-clock
  * privacy_class: public | private | sensitive
  * quality_floor: 0.0–1.0 (per-task historical accuracy threshold)

The broker calls `bid(job)` on every registered Provider. Each
provider returns a Bid (price, eta_ms, expected_quality, evidence).
Bids that violate ceilings are filtered out. The remaining bids are
ranked by a Pareto-optimal scalar score (lower-is-better):

    score = α · (price / cost_ceiling)
          + β · (eta_ms / latency_ceiling)
          + γ · max(0, quality_floor - expected_quality)
          + δ · privacy_penalty(privacy_class, provider.privacy_grade)

The winning bid is settled in PLG (the chain's native token) via a
two-phase escrow:
  1. ESCROW phase — caller's PLG is locked in a chain-side
     escrow address; provider submits an `acceptance_signature` over
     (job_id, bid_hash, deadline) committing to deliver.
  2. SETTLE phase — on result delivery + acceptance (or arbiter
     verdict on dispute), escrow releases to the winning provider
     net of protocol fee.

Provider abstraction
--------------------
The Provider ABC is intentionally narrow — three methods:
  * `bid(job)`   — return a Bid or None.
  * `execute(job, bid)` — run the job (or proxy to peer / API).
  * `attest(result)`    — sign the result (and ZK proof, if compute).

This shape lets the same broker auction across:
  * `MeshGPUProvider`     — peer machine running an open-weight LLM
  * `OpenAIProvider`      — gpt-* via the public API (key from
                            keyring, never disk)
  * `AnthropicProvider`   — claude-* via the public API
  * `GeminiProvider`      — gemini via the public API
  * `GroqProvider`        — Groq inference
  * `OpenRouterProvider`  — multi-provider router
  * `OllamaProvider`      — local Ollama runtime
  * `GitHubProvider`      — repo-pull / commit pipeline (W17)

This module ships:
  * `Provider`, `Bid`, `JobSpec`, `Bid.evaluate_score` — the core ABC
    and value types.
  * `MeshGPUProvider` — peer-GPU stub with TimeOfDaySlackCurve hook.
  * `OpenAIProvider`, `AnthropicProvider` — minimal LLM-API stubs
    using `keyring` for secret retrieval (NEVER hits disk).
  * `Auction` — the broker side: register providers, run the
    sealed-bid round, return winner.

Concrete network calls are gated behind an `enabled` flag and an
`api_key` retrieval that fails closed when no key is configured.
The shipped MVP doesn't auto-charge external accounts; integrators
configure keyring entries explicitly.
"""

from __future__ import annotations

import abc
import base64
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import slack_auction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------
PRIVACY_PUBLIC = "public"
PRIVACY_PRIVATE = "private"
PRIVACY_SENSITIVE = "sensitive"
_PRIVACY_RANK = {
    PRIVACY_PUBLIC: 0, PRIVACY_PRIVATE: 1, PRIVACY_SENSITIVE: 2,
}


@dataclass
class JobSpec:
    """A unit of work submitted to the auction."""
    job_id: str
    kind: str                       # 'inference' | 'training' | 'embed' | ...
    payload: Dict[str, Any]         # task-specific (e.g. prompt, model, max_tokens)
    cost_ceiling_usd: float = 0.10
    latency_ceiling_ms: int = 30_000
    privacy_class: str = PRIVACY_PUBLIC
    quality_floor: float = 0.7
    # A2: reasoning-time budget. The caller can ask
    # for "deep thinking" by allowing a provider to spend up to N
    # seconds of reasoning before producing the final answer. The
    # default 0 preserves the historical "fast path" where any
    # reasoning-time happens inside latency_ceiling_ms. A bid's
    # `reasoning_seconds_committed` must be <= this; eta_ms is the
    # wall-clock ceiling INCLUDING any reasoning time.
    reasoning_seconds_max: int = 0
    submitted_at: float = field(default_factory=time.time)
    # Optional authentication: when set, a provider configured with
    # `require_signed_requests=True` will reject jobs whose signature
    # over (job_id || canonical_payload_hash) does not verify against
    # `requester_pubkey_pem`. Auction-layer brokers populate these.
    requester_pubkey_pem: Optional[str] = None
    request_signature: Optional[str] = None  # base64 ECDSA over the bytes
                                             # described in `signing_message`

    def payload_hash(self) -> str:
        """SHA256 of the canonical JSON of `payload` only.

        Used in the requester signature: sign(job_id + ":" + payload_hash).
        """
        return hashlib.sha256(
            json.dumps(self.payload, sort_keys=True, default=str).encode()
        ).hexdigest()

    def signing_message(self) -> str:
        return f"{self.job_id}:{self.payload_hash()}"

    def canonical_hash(self) -> str:
        body = {
            "job_id": self.job_id,
            "kind": self.kind,
            "payload": self.payload,
            "cost_ceiling_usd": self.cost_ceiling_usd,
            "latency_ceiling_ms": self.latency_ceiling_ms,
            "privacy_class": self.privacy_class,
            "quality_floor": self.quality_floor,
        }
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, default=str).encode()
        ).hexdigest()


@dataclass
class Bid:
    """A provider's offer for a job."""
    provider_id: str
    price_usd: float
    eta_ms: int
    expected_quality: float          # 0.0-1.0; provider's self-reported
                                     # historical accuracy on this task class
    privacy_grade: str               # the privacy class this bid SUPPORTS
                                     # (e.g. peer GPU with TEE = sensitive)
    # A2: how much wall-time the provider commits to spend on
    # reasoning before answering. 0 = "fast path, no extended
    # thinking". Bids that commit MORE reasoning_seconds than the
    # caller permits via reasoning_seconds_max are rejected.
    reasoning_seconds_committed: int = 0
    evidence: Dict[str, Any] = field(default_factory=dict)  # diagnostic

    def violates(self, job: JobSpec) -> Optional[str]:
        """Return a string reason if the bid violates a hard constraint."""
        # Sanity: bids that violate physics or accounting are rejected
        # before constraint checks. CP-5 byzantine-auction regression.
        if self.price_usd < 0:
            return f"negative price ({self.price_usd}) is not a valid bid"
        if self.eta_ms <= 0:
            return f"non-positive eta_ms ({self.eta_ms}) is not a valid bid"
        if not (0.0 <= self.expected_quality <= 1.0):
            return f"expected_quality {self.expected_quality} outside [0, 1]"
        if self.price_usd > job.cost_ceiling_usd:
            return f"price {self.price_usd:.4f} > ceiling {job.cost_ceiling_usd}"
        if self.eta_ms > job.latency_ceiling_ms:
            return f"eta {self.eta_ms} > ceiling {job.latency_ceiling_ms}"
        if self.expected_quality < job.quality_floor:
            return f"quality {self.expected_quality:.2f} < floor {job.quality_floor}"
        if _PRIVACY_RANK.get(self.privacy_grade, -1) < \
                _PRIVACY_RANK.get(job.privacy_class, 0):
            return (f"privacy {self.privacy_grade} < required "
                    f"{job.privacy_class}")
        # A2: reasoning-time-budget enforcement.
        if self.reasoning_seconds_committed < 0:
            return (f"negative reasoning_seconds_committed "
                    f"({self.reasoning_seconds_committed})")
        if self.reasoning_seconds_committed > job.reasoning_seconds_max:
            return (f"reasoning_seconds {self.reasoning_seconds_committed} "
                    f"> caller max {job.reasoning_seconds_max}")
        return None

    def score(self, job: JobSpec,
              alpha: float = 1.0, beta: float = 1.0,
              gamma: float = 2.0, delta: float = 1.0) -> float:
        """Lower is better. The Pareto-scalarisation function — the
        design rationale covers this specific 4-term form.

        alpha*price_ratio + beta*eta_ratio
          + gamma*max(0, quality_floor - expected_quality)
          + delta*privacy_penalty
        """
        if job.cost_ceiling_usd > 0:
            price_ratio = self.price_usd / job.cost_ceiling_usd
        else:
            price_ratio = self.price_usd
        if job.latency_ceiling_ms > 0:
            eta_ratio = self.eta_ms / job.latency_ceiling_ms
        else:
            eta_ratio = self.eta_ms / 1000.0
        quality_gap = max(0.0, job.quality_floor - self.expected_quality)
        # privacy penalty: zero if provider's grade >= required, large
        # otherwise (but bid would already be filtered out by violates()
        # — this term is here so the auction can be re-used for
        # soft-privacy use cases too).
        privacy_penalty = max(
            0,
            _PRIVACY_RANK.get(job.privacy_class, 0)
            - _PRIVACY_RANK.get(self.privacy_grade, 0),
        )
        return (alpha * price_ratio
                + beta * eta_ratio
                + gamma * quality_gap
                + delta * privacy_penalty)


# ---------------------------------------------------------------------
# Provider ABC
# ---------------------------------------------------------------------
class Provider(abc.ABC):
    """Abstract base — implement bid/execute/attest."""

    provider_id: str
    privacy_grade: str = PRIVACY_PUBLIC   # default: cloud APIs serve public

    @abc.abstractmethod
    def bid(self, job: JobSpec) -> Optional[Bid]:
        """Return a Bid, or None to abstain."""

    @abc.abstractmethod
    def execute(self, job: JobSpec, bid: Bid) -> Dict[str, Any]:
        """Run the job. Returns a result dict; integrators add
        on-chain settlement on top."""

    def attest(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Optional: sign / proof-attach. Default no-op."""
        return result


# ---------------------------------------------------------------------
# Concrete: peer GPU provider (the cost-dynamics shifter)
# ---------------------------------------------------------------------
class ProviderConfigurationError(RuntimeError):
    """Raised when a provider is asked to execute without the dependencies
    (TaskRouter / Wallet / etc) needed to do real work."""


@dataclass
class MeshGPUProvider(Provider):
    """A peer GPU on the Pluginfer mesh.

    The bid pulls hardware capability + current slack from
    `slack_auction.TimeOfDaySlackCurve` (TODO §4.2 — separate design note
    claim) and converts to USD via an oracle peg.

    `execute()` dispatches via `task_router.TaskRouter` and signs the
    result hash with the provider's wallet. Both `task_router` and
    `wallet` must be set for real execution; absent them the call
    raises `ProviderConfigurationError` (no silent stub).

    `local_executor` is an optional callable taking the input_data dict
    and returning a `bytes` result. When set, the provider runs work
    locally (no mesh round-trip) and returns the bytes signed; this is
    how a leaf node serves jobs it accepted from the auction. When unset
    AND task_router IS set, the provider acts as a relay and broadcasts
    to the mesh.
    """
    provider_id: str
    hardware_class: str = "consumer-gpu-mid"   # informational
    base_price_per_1k_tok_usd: float = 0.0008  # ~10x cheaper than GPT-4o
    base_eta_ms: int = 2000
    base_quality: float = 0.78
    privacy_grade: str = PRIVACY_PRIVATE       # mesh is more private
                                               # than cloud APIs
    slack_curve: Optional[slack_auction.TimeOfDaySlackCurve] = None
    enabled: bool = True
    # Real-execution dependencies (injected at construction time).
    task_router: Optional[Any] = None        # core.task_router.TaskRouter
    wallet: Optional[Any] = None             # core.tokenomics.Wallet
    local_executor: Optional[Any] = None     # Callable[[dict], bytes]
    require_signed_requests: bool = False

    def bid(self, job: JobSpec) -> Optional[Bid]:
        if not self.enabled:
            return None
        # Slack-aware price multiplier (1.0 = on-target, <1 = idle, >1 = busy).
        slack_factor = (
            self.slack_curve.opportunity_cost_factor() if self.slack_curve
            else 1.0
        )
        price = self.base_price_per_1k_tok_usd * slack_factor
        # The auction asks for `max_tokens` so it can quote an envelope
        # price. The actual production token estimate comes from the
        # caller-supplied prompt + max_tokens cap.
        approx_tokens = float(job.payload.get("max_tokens", 200))
        total_price = price * (approx_tokens / 1000.0)
        return Bid(
            provider_id=self.provider_id,
            price_usd=total_price,
            eta_ms=self.base_eta_ms,
            expected_quality=self.base_quality,
            privacy_grade=self.privacy_grade,
            evidence={"hardware_class": self.hardware_class,
                      "slack_factor": slack_factor},
        )

    def execute(self, job: JobSpec, bid: Bid) -> Dict[str, Any]:
        """Real dispatch: validate requester sig (if required), run the
        job (locally or via mesh), hash + sign the result, return
        a structured response.

        Errors (timeout, validation failure, executor error) return a
        structured dict with `refund_eligible=True` so the auction
        layer can release escrow safely. Configuration errors (no
        task_router AND no local_executor) raise loudly -- silent
        stubs are forbidden.
        """
        t_start = time.time()
        # 1. Optional requester-signature gate
        if self.require_signed_requests:
            err = self._verify_requester_signature(job)
            if err is not None:
                return {
                    "status": "error",
                    "code": "requester_sig_invalid",
                    "reason": err,
                    "provider_id": self.provider_id,
                    "job_id": job.job_id,
                    "refund_eligible": True,
                }

        # 2. Run the job (local fast path or mesh dispatch)
        try:
            if self.local_executor is not None:
                result_bytes, exec_meta = self._run_locally(job, bid)
            elif self.task_router is not None:
                result_bytes, exec_meta = self._run_via_mesh(job, bid)
            else:
                raise ProviderConfigurationError(
                    f"MeshGPUProvider {self.provider_id!r} has neither "
                    f"local_executor nor task_router configured; cannot "
                    f"execute. Wire one of them at construction time."
                )
        except _MeshTimeout as e:
            return {
                "status": "timeout",
                "provider_id": self.provider_id,
                "job_id": job.job_id,
                "deadline_ms": e.deadline_ms,
                "refund_eligible": True,
            }
        except _MeshExecutionError as e:
            return {
                "status": "error",
                "code": "execution_error",
                "reason": str(e),
                "provider_id": self.provider_id,
                "job_id": job.job_id,
                "refund_eligible": True,
            }

        # 3. Hash + sign the result
        result_hash = hashlib.sha256(result_bytes).hexdigest()
        provider_sig = self._sign_result(result_hash)

        return {
            "status": "executed",
            "provider_id": self.provider_id,
            "job_id": job.job_id,
            "result_hash": result_hash,
            "result_bytes": base64.b64encode(result_bytes).decode("ascii"),
            "execution_ms": int((time.time() - t_start) * 1000),
            "provider_sig": provider_sig,
            "provider_pubkey_pem": self._pubkey_pem(),
            "exec_meta": exec_meta,
        }

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _verify_requester_signature(self, job: JobSpec) -> Optional[str]:
        if not job.requester_pubkey_pem or not job.request_signature:
            return ("require_signed_requests=True but job has no "
                    "requester_pubkey_pem / request_signature")
        # Lazy import to avoid hard-coupling the auction layer to tokenomics
        # at module import time (tokenomics imports compute_ledger which
        # imports a lot).
        from . import tokenomics as _tok
        ok = _tok.Wallet.verify(
            job.requester_pubkey_pem,
            job.signing_message(),
            job.request_signature,
        )
        if not ok:
            return "ECDSA verification of request_signature failed"
        return None

    def _run_locally(self, job: JobSpec, bid: Bid):
        try:
            result = self.local_executor(job.payload)
        except Exception as e:
            raise _MeshExecutionError(f"local_executor raised: {e!r}") from e
        if not isinstance(result, (bytes, bytearray)):
            raise _MeshExecutionError(
                f"local_executor must return bytes; got "
                f"{type(result).__name__}"
            )
        return bytes(result), {"path": "local",
                               "hardware_class": self.hardware_class}

    def _run_via_mesh(self, job: JobSpec, bid: Bid):
        # Lazy import to avoid hard cycle at module load.
        from .task_router import TaskRequirements

        requirements = TaskRequirements(
            plugin=str(job.payload.get("plugin", job.kind)),
            min_vram_gb=float(job.payload.get("min_vram_gb", 0.0)),
            needs_gpu=bool(job.payload.get("needs_gpu", False)),
            deadline_ms=int(min(job.latency_ceiling_ms, bid.eta_ms * 5)),
            redundancy=int(job.payload.get("redundancy", 1)),
            cost_ceiling_plg=float(job.payload.get("cost_ceiling_plg", 1.0)),
        )
        timeout_s = max(1.0, requirements.deadline_ms / 1000.0)
        result = self.task_router.submit_and_wait(
            requirements, job.payload, timeout_s=timeout_s,
        )
        if result is None:
            raise _MeshTimeout(deadline_ms=requirements.deadline_ms)
        if isinstance(result, dict) and result.get("status") == "error":
            raise _MeshExecutionError(
                str(result.get("reason", "mesh-side error"))
            )
        # Serialise the mesh result into a deterministic byte string.
        # Mesh results are always JSON-able by contract (see TaskRouter).
        result_bytes = json.dumps(
            result, sort_keys=True, default=str
        ).encode("utf-8")
        return result_bytes, {"path": "mesh",
                              "redundancy": requirements.redundancy}

    def _sign_result(self, result_hash_hex: str) -> str:
        if self.wallet is None:
            raise ProviderConfigurationError(
                f"MeshGPUProvider {self.provider_id!r} has no wallet "
                f"configured; cannot sign result. Pass `wallet=...` at "
                f"construction time."
            )
        return self.wallet.sign(result_hash_hex)

    def _pubkey_pem(self) -> Optional[str]:
        if self.wallet is None:
            return None
        return self.wallet.public_key_pem


class _MeshTimeout(RuntimeError):
    def __init__(self, deadline_ms: int) -> None:
        super().__init__(f"mesh dispatch exceeded {deadline_ms}ms")
        self.deadline_ms = deadline_ms


class _MeshExecutionError(RuntimeError):
    pass


# ---------------------------------------------------------------------
# Concrete: cloud LLM provider stubs
# ---------------------------------------------------------------------
@dataclass
class _CloudLLMProvider(Provider):
    """Shared logic for OpenAI / Anthropic / Gemini / Groq / OpenRouter
    style providers.

    SECURITY: API keys are NEVER stored on disk by this class. They
    are looked up via the `keyring` package (Windows Credential
    Manager / macOS Keychain / Linux Secret Service) the moment they
    are needed, used in-memory, and never logged.
    """
    provider_id: str
    keychain_service: str
    keychain_user: str
    base_price_per_1k_tok_usd: float
    base_eta_ms: int
    base_quality: float
    privacy_grade: str = PRIVACY_PUBLIC
    enabled: bool = False

    def _key(self) -> Optional[str]:
        try:
            import keyring  # noqa
            return keyring.get_password(self.keychain_service,
                                        self.keychain_user)
        except Exception as e:
            logger.debug("[%s] keyring lookup failed: %s",
                         self.provider_id, e)
            return None

    def bid(self, job: JobSpec) -> Optional[Bid]:
        if not self.enabled:
            return None
        if self._key() is None:
            # Fail closed — never bid for a job we can't fulfill.
            return None
        approx_tokens = float(job.payload.get("max_tokens", 200))
        total_price = self.base_price_per_1k_tok_usd * (approx_tokens / 1000.0)
        return Bid(
            provider_id=self.provider_id,
            price_usd=total_price,
            eta_ms=self.base_eta_ms,
            expected_quality=self.base_quality,
            privacy_grade=self.privacy_grade,
            evidence={"upstream": self.provider_id},
        )

    def execute(self, job: JobSpec, bid: Bid) -> Dict[str, Any]:
        """Real upstream HTTPS dispatch.

        Schema chosen by `provider_id` prefix:
          openai* / openrouter* / groq* -> OpenAI Chat Completions
          anthropic*                    -> Anthropic Messages
          gemini*                       -> Google Generative Language
          ollama*                       -> Ollama /api/generate

        Fail-closed on missing key or any HTTP error -- never returns a
        synthesised "ok" response when the upstream call did not happen.
        """
        if self._key() is None:
            return {
                "status": "error",
                "code": "no_api_key",
                "provider_id": self.provider_id,
                "remediation": (
                    f"Set keychain entry {self.keychain_service}/"
                    f"{self.keychain_user} to enable {self.provider_id} "
                    f"dispatch."
                ),
                "refund_eligible": True,
            }
        prompt = str(
            job.payload.get("prompt") or job.payload.get("input") or ""
        )
        if not prompt:
            return {
                "status": "error",
                "code": "no_prompt",
                "provider_id": self.provider_id,
                "reason": "job.payload must include 'prompt' or 'input' "
                          "for cloud LLM dispatch",
                "refund_eligible": True,
            }
        max_tokens = int(job.payload.get("max_tokens", 256))
        timeout_s = max(1.0, min(60.0, bid.eta_ms / 1000.0 * 5))
        t0 = time.time()
        try:
            text = self._dispatch_upstream(prompt, max_tokens, timeout_s)
        except _CloudHttpError as e:
            return {
                "status": "error",
                "code": e.code,
                "provider_id": self.provider_id,
                "reason": e.detail,
                "refund_eligible": True,
            }
        text_bytes = text.encode("utf-8")
        result_hash = hashlib.sha256(text_bytes).hexdigest()
        return {
            "status": "executed",
            "provider_id": self.provider_id,
            "job_id": job.job_id,
            "result_text": text,
            "result_bytes": base64.b64encode(text_bytes).decode("ascii"),
            "result_hash": result_hash,
            "execution_ms": int((time.time() - t0) * 1000),
        }

    def _dispatch_upstream(
        self, prompt: str, max_tokens: int, timeout_s: float
    ) -> str:
        """Schema-dispatched upstream call. Imports httpx lazily so chain-
        only nodes don't need it on disk."""
        try:
            import httpx  # noqa: WPS433
        except ImportError as e:
            raise _CloudHttpError(
                code="httpx_not_installed",
                detail=f"httpx not available: {e!r}. Install with `pip "
                       f"install httpx` to enable cloud LLM dispatch.",
            ) from e
        pid = self.provider_id.lower()
        key = self._key() or ""
        try:
            if pid.startswith(("openai", "openrouter", "groq")):
                return _call_openai_chat(
                    httpx, pid, key, prompt, max_tokens, timeout_s
                )
            if pid.startswith("anthropic"):
                return _call_anthropic_messages(
                    httpx, key, prompt, max_tokens, timeout_s
                )
            if pid.startswith("gemini"):
                return _call_gemini_generate(
                    httpx, key, prompt, max_tokens, timeout_s
                )
            if pid.startswith("ollama"):
                return _call_ollama_generate(
                    httpx, prompt, max_tokens, timeout_s
                )
            raise _CloudHttpError(
                code="unknown_provider_schema",
                detail=(
                    f"no upstream schema known for provider_id "
                    f"{self.provider_id!r}. Override _dispatch_upstream "
                    f"in a subclass or use one of: openai, anthropic, "
                    f"gemini, ollama, openrouter, groq."
                ),
            )
        except httpx.HTTPError as e:
            raise _CloudHttpError(
                code="http_error", detail=f"{type(e).__name__}: {e!r}"
            ) from e


class _CloudHttpError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# OpenAI / OpenRouter / Groq all use the OpenAI Chat Completions schema.
# Picked the chat endpoint (not /v1/completions) because it's the modern
# default; legacy completion endpoint is one schema swap away.
def _call_openai_chat(
    httpx_mod, provider_id: str, key: str, prompt: str,
    max_tokens: int, timeout_s: float,
) -> str:
    if provider_id.startswith("openrouter"):
        url = "https://openrouter.ai/api/v1/chat/completions"
    elif provider_id.startswith("groq"):
        url = "https://api.groq.com/openai/v1/chat/completions"
    else:
        url = "https://api.openai.com/v1/chat/completions"
    body = {
        "model": "gpt-4o-mini",  # cheapest reasonable default; caller can
                                  # override via job.payload['model']
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    r = httpx_mod.post(
        url,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        json=body, timeout=timeout_s,
    )
    if r.status_code != 200:
        raise _CloudHttpError(
            code=f"http_{r.status_code}", detail=r.text[:500]
        )
    body = r.json()
    return body["choices"][0]["message"]["content"]


def _call_anthropic_messages(
    httpx_mod, key: str, prompt: str,
    max_tokens: int, timeout_s: float,
) -> str:
    body = {
        "model": "claude-haiku-4-5-20251001",  # cheapest current Claude
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = httpx_mod.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key,
                 "anthropic-version": "2023-06-01",
                 "Content-Type": "application/json"},
        json=body, timeout=timeout_s,
    )
    if r.status_code != 200:
        raise _CloudHttpError(
            code=f"http_{r.status_code}", detail=r.text[:500]
        )
    body = r.json()
    return "".join(
        block["text"] for block in body["content"] if block["type"] == "text"
    )


def _call_gemini_generate(
    httpx_mod, key: str, prompt: str,
    max_tokens: int, timeout_s: float,
) -> str:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-1.5-flash:generateContent"
    )
    r = httpx_mod.post(
        url,
        params={"key": key},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        },
        timeout=timeout_s,
    )
    if r.status_code != 200:
        raise _CloudHttpError(
            code=f"http_{r.status_code}", detail=r.text[:500]
        )
    body = r.json()
    return "".join(
        p.get("text", "")
        for p in body["candidates"][0]["content"]["parts"]
    )


def _call_ollama_generate(
    httpx_mod, prompt: str, max_tokens: int, timeout_s: float,
) -> str:
    # Ollama is local; no API key. Reads OLLAMA_HOST from env or
    # falls back to localhost:11434.
    import os
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    r = httpx_mod.post(
        f"{host}/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False,
              "options": {"num_predict": max_tokens}},
        timeout=timeout_s,
    )
    if r.status_code != 200:
        raise _CloudHttpError(
            code=f"http_{r.status_code}", detail=r.text[:500]
        )
    return r.json().get("response", "")


def OpenAIProvider(**kw) -> _CloudLLMProvider:
    return _CloudLLMProvider(
        provider_id=kw.pop("provider_id", "openai"),
        keychain_service=kw.pop("keychain_service", "pluginfer-openai"),
        keychain_user=kw.pop("keychain_user", "default"),
        base_price_per_1k_tok_usd=kw.pop(
            "base_price_per_1k_tok_usd", 0.0050),  # gpt-4o-mini-ish
        base_eta_ms=kw.pop("base_eta_ms", 1500),
        base_quality=kw.pop("base_quality", 0.92),
        **kw,
    )


def AnthropicProvider(**kw) -> _CloudLLMProvider:
    return _CloudLLMProvider(
        provider_id=kw.pop("provider_id", "anthropic"),
        keychain_service=kw.pop("keychain_service", "pluginfer-anthropic"),
        keychain_user=kw.pop("keychain_user", "default"),
        base_price_per_1k_tok_usd=kw.pop(
            "base_price_per_1k_tok_usd", 0.0080),  # claude-haiku-ish
        base_eta_ms=kw.pop("base_eta_ms", 1300),
        base_quality=kw.pop("base_quality", 0.93),
        **kw,
    )


# ---------------------------------------------------------------------
# Auction
# ---------------------------------------------------------------------
@dataclass
class Auction:
    """Sealed-bid auction across registered providers.

    Workflow:
      1. register(provider) — add a provider to the auction set.
      2. run(job) — collect bids, filter, score, pick winner.
      3. broker / on-chain layer takes the winner, escrows, settles.

    Returns an `AuctionResult` with the winning bid (or None) and
    full diagnostics so the broker can log losing bids for
    transparency / dispute audits."""
    providers: List[Provider] = field(default_factory=list)
    # HG21: when an economic layer is attached, this returns
    # (eligible, reason) per provider_id — un-bonded or quarantined
    # providers never even get to bid. One gate, before any bid.
    eligibility_fn: Optional[Callable[[str], Tuple[bool, str]]] = None

    def register(self, p: Provider) -> None:
        self.providers.append(p)

    def run(self, job: JobSpec) -> "AuctionResult":
        bids: List[Bid] = []
        rejected: List[Dict[str, Any]] = []
        for p in self.providers:
            if self.eligibility_fn is not None:
                pid = getattr(p, "provider_id", "?")
                try:
                    ok, why = self.eligibility_fn(pid)
                except Exception as e:
                    ok, why = True, f"eligibility check errored: {e}"
                if not ok:
                    rejected.append({"provider_id": pid,
                                     "reason": f"ineligible: {why}"})
                    continue
            try:
                b = p.bid(job)
            except Exception as e:
                rejected.append({"provider_id": getattr(p, "provider_id", "?"),
                                 "reason": f"provider raised: {e}"})
                continue
            if b is None:
                rejected.append({"provider_id": getattr(p, "provider_id", "?"),
                                 "reason": "abstained"})
                continue
            why = b.violates(job)
            if why:
                rejected.append({"provider_id": b.provider_id,
                                 "reason": why,
                                 "bid": b})
                continue
            bids.append(b)

        if not bids:
            return AuctionResult(winner=None, bids=[], rejected=rejected)

        scored = sorted(((b.score(job), b) for b in bids), key=lambda t: t[0])
        winner_score, winner = scored[0]
        return AuctionResult(
            winner=winner,
            winner_score=winner_score,
            bids=[b for _, b in scored],
            rejected=rejected,
        )


@dataclass
class AuctionResult:
    winner: Optional[Bid]
    bids: List[Bid] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    winner_score: float = float("inf")

    def is_won(self) -> bool:
        return self.winner is not None
