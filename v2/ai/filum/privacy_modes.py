"""Filum privacy modes: explicit per-query enforcement.

The user-visible promise: when a query is marked `LOCAL_ONLY`,
NO BYTE of that query, intermediate, or response leaves the device.
Period. The privacy enforcement layer is a hard gate at every
network boundary -- it's the only way to honestly claim "local AI"
when the same codebase ALSO supports mesh-distributed inference.

Three modes, in increasing exposure:

  1. LOCAL_ONLY  -- 100% on-device. No teachers, no RAG over remote
                    indices, no peer escalation. Filum's pure-weights
                    answer is what ships, period. Higher latency on
                    hard queries; total privacy.

  2. HYBRID      -- DEFAULT. Local Filum drafts; on low confidence
                    OR explicit `force_teacher`, escalate to a teacher
                    API (configured by user). Local RAG always on
                    (the index lives on the user's disk). Peer LoRAs
                    rented IF allowed by the user's adapter policy.
                    Most queries stay local; some metadata leaves
                    when escalation triggers.

  3. MESH_FULL   -- The user has explicitly opted IN to mesh inference
                    (e.g. they don't have a GPU, or they want the
                    fastest specialty adapter). Queries route to a
                    peer node. Encrypted in transit; peer can see
                    the query plaintext. Best perf for users without
                    local hardware; weakest privacy.

The boundary is enforced at THREE layers:

  * `speculative.SpeculativeRunner` -- gates teacher escalation.
  * `lora_pool.LoRAPool.route` -- gates remote-adapter rental.
  * `retrieval.RAGPipeline` -- gates remote-index lookup.

ALL three check the active `PrivacyMode` before any network call. A
LOCAL_ONLY query that requires teacher escalation does NOT escalate;
it returns Filum's best draft + a warning that the answer is
unverified.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class PrivacyMode(enum.Enum):
    """User-selectable privacy mode for a query (or a session)."""
    LOCAL_ONLY = "local_only"
    HYBRID = "hybrid"
    MESH_FULL = "mesh_full"


@dataclass
class PrivacyPolicy:
    """A snapshot of what a query is allowed to do.

    Created from a `PrivacyMode` + the user's per-feature opt-ins
    (e.g. they may be in HYBRID but have specifically forbidden
    Anthropic API calls)."""
    mode: PrivacyMode
    allow_teacher_escalation: bool
    allow_peer_lora_rental: bool
    allow_mesh_inference: bool
    allow_chain_receipt_logging: bool
    allow_remote_rag: bool
    forbidden_teachers: Tuple[str, ...] = ()
    forbidden_regions: Tuple[str, ...] = ()        # e.g. ("us-east", ...)

    @classmethod
    def from_mode(cls, mode: PrivacyMode, **overrides) -> "PrivacyPolicy":
        if mode == PrivacyMode.LOCAL_ONLY:
            base = cls(
                mode=mode,
                allow_teacher_escalation=False,
                allow_peer_lora_rental=False,
                allow_mesh_inference=False,
                allow_chain_receipt_logging=False,
                allow_remote_rag=False,
            )
        elif mode == PrivacyMode.HYBRID:
            base = cls(
                mode=mode,
                allow_teacher_escalation=True,
                allow_peer_lora_rental=True,
                allow_mesh_inference=False,            # opt-in only
                allow_chain_receipt_logging=True,
                allow_remote_rag=False,                # local index by default
            )
        elif mode == PrivacyMode.MESH_FULL:
            base = cls(
                mode=mode,
                allow_teacher_escalation=True,
                allow_peer_lora_rental=True,
                allow_mesh_inference=True,
                allow_chain_receipt_logging=True,
                allow_remote_rag=True,
            )
        else:
            raise ValueError(f"unknown mode: {mode}")
        # Apply overrides.
        for k, v in overrides.items():
            if hasattr(base, k):
                setattr(base, k, v)
        return base

    # ------------------------------------------------------------------

    def check_teacher(self, teacher_id: Optional[str] = None) -> bool:
        """Return True if a teacher API call is permitted."""
        if not self.allow_teacher_escalation:
            return False
        if teacher_id and teacher_id in self.forbidden_teachers:
            return False
        return True

    def check_peer_inference(self) -> bool:
        return self.allow_mesh_inference

    def check_peer_lora(self) -> bool:
        return self.allow_peer_lora_rental

    def check_remote_rag(self) -> bool:
        return self.allow_remote_rag

    def check_chain_logging(self) -> bool:
        return self.allow_chain_receipt_logging

    # ------------------------------------------------------------------

    def explain(self) -> str:
        return (
            f"PrivacyMode.{self.mode.name}: "
            f"teacher={'on' if self.allow_teacher_escalation else 'OFF'}, "
            f"peer_lora={'on' if self.allow_peer_lora_rental else 'OFF'}, "
            f"mesh={'on' if self.allow_mesh_inference else 'OFF'}, "
            f"remote_rag={'on' if self.allow_remote_rag else 'OFF'}, "
            f"chain_log={'on' if self.allow_chain_receipt_logging else 'OFF'}"
        )


# ---------------------------------------------------------------------------
# Hard-gate decorators
# ---------------------------------------------------------------------------


class PrivacyViolation(RuntimeError):
    """Raised when code path attempts to leave the device under a
    LOCAL_ONLY policy. The error is explicit so debugging is
    obvious (silent fall-back would be misleading)."""


def require_teacher(policy: PrivacyPolicy, teacher_id: Optional[str] = None) -> None:
    """Call this at the entry of any teacher-API code path. Raises
    PrivacyViolation if the policy forbids it."""
    if not policy.check_teacher(teacher_id):
        raise PrivacyViolation(
            f"teacher escalation forbidden under {policy.mode.name} "
            f"(teacher_id={teacher_id})"
        )


def require_peer_inference(policy: PrivacyPolicy) -> None:
    if not policy.check_peer_inference():
        raise PrivacyViolation(
            f"peer-mesh inference forbidden under {policy.mode.name}"
        )


def require_peer_lora(policy: PrivacyPolicy) -> None:
    if not policy.check_peer_lora():
        raise PrivacyViolation(
            f"peer LoRA rental forbidden under {policy.mode.name}"
        )


def require_remote_rag(policy: PrivacyPolicy) -> None:
    if not policy.check_remote_rag():
        raise PrivacyViolation(
            f"remote RAG forbidden under {policy.mode.name}"
        )


# ---------------------------------------------------------------------------
# User-facing helpers
# ---------------------------------------------------------------------------


DEFAULT_POLICY = PrivacyPolicy.from_mode(PrivacyMode.HYBRID)


def policy_for_kind(kind: Optional[str]) -> PrivacyPolicy:
    """Map a Pluginfer JobSpec.privacy_class -> PrivacyPolicy.
    Confidential = LOCAL_ONLY; internal = HYBRID; public = HYBRID."""
    if kind is None:
        return DEFAULT_POLICY
    k = kind.lower()
    if k in ("confidential", "private", "secret"):
        return PrivacyPolicy.from_mode(PrivacyMode.LOCAL_ONLY)
    if k == "public":
        return PrivacyPolicy.from_mode(PrivacyMode.HYBRID)
    if k in ("internal",):
        return PrivacyPolicy.from_mode(PrivacyMode.HYBRID)
    return DEFAULT_POLICY
