"""Universal AI Receipt Standard (PNIS-Receipt v1).

Every AI job that runs through Pluginfer produces a signed *receipt*:
a tamper-evident artifact recording exactly which model produced the
output, who signed for it, what it cost, how much energy it consumed,
and (optionally) a commitment to the dataset used. This is the
"AI bill-of-materials" that EU AI Act enforcement, SEC AI disclosures,
and enterprise compliance teams will demand within 24 months -- and
Pluginfer is the first network to produce one by default.

A receipt is a JSON document containing:

    {
      "schema": "pnis-receipt/v1",
      "job_id":          "...",
      "model":           {"id": "...", "hash": "<sha256>", "kind": "llm"},
      "provider":        {"id": "...", "pubkey": "<PEM>", "kind": "mesh|cloud"},
      "input_hash":      "<sha256>",
      "output_hash":     "<sha256>",
      "cost":            {"plg": "0.012345", "usd_estimate": "0.0008"},
      "energy_mj":       "0.42",
      "latency_ms":      813,
      "dataset_attestation_hash": null | "<sha256>",
      "timestamp_ns":    16859274859382000,
      "signature": {
        "alg":     "ecdsa-secp256k1-sha256",
        "value":   "<base64>",
        "pubkey":  "<PEM of provider key>"
      }
    }

The signature is over the canonical JSON of every other field
(sorted-keys, no whitespace) so any tampering is detected.

Why this matters
----------------
* **Regulators** can verify a Pluginfer-served output came from the
  model the operator claimed -- no opaque vendor black-box.
* **Enterprises** can include the receipt in their audit trail without
  having to trust the provider.
* **End users** can compare costs across providers like a fuel
  receipt: the same job ran on PROVIDER_A for $0.001 and PROVIDER_B
  for $0.0008 -- transparent market pressure pushes prices down.
* **Researchers** can prove a paper's experiment ran on the exact
  model checkpoint claimed (combined with §A9 verifiable-inference).

"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .tokenomics import Wallet

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "pnis-receipt/v1"


def _sha256(data: Union[bytes, str]) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _now_ns() -> int:
    return time.time_ns()


# ---------------------------------------------------------------------------
# Receipt body
# ---------------------------------------------------------------------------


@dataclass
class ModelRef:
    id: str                          # e.g. "filum-127m" or "openai/gpt-4o-mini"
    hash: str                        # sha256 of weights or canonical model id
    kind: str = "llm"                # llm | vision | audio | code | embedding


@dataclass
class ProviderRef:
    id: str                          # provider stable id (wallet addr or platform)
    pubkey: str                      # PEM
    kind: str = "mesh"               # mesh | cloud | enterprise


@dataclass
class CostBreakdown:
    plg: str                         # Decimal-as-str for chain payment
    usd_estimate: str = "0"          # Decimal-as-str for human comparison


@dataclass
class AIReceiptBody:
    """The unsigned receipt body. Every field (except `signature`) is
    folded into the canonical JSON that gets signed."""
    schema: str
    job_id: str
    model: ModelRef
    provider: ProviderRef
    input_hash: str
    output_hash: str
    cost: CostBreakdown
    energy_mj: str
    latency_ms: int
    dataset_attestation_hash: Optional[str]
    timestamp_ns: int

    def canonical_json(self) -> str:
        """Deterministic JSON used as signature input.

        Uses sort_keys=True + tightest separators so receivers on any
        platform get the same bytes the signer signed.
        """
        return json.dumps(asdict(self), sort_keys=True,
                          separators=(",", ":"), ensure_ascii=False)


@dataclass
class AIReceipt:
    """Signed receipt = body + signature block."""
    body: AIReceiptBody
    signature_alg: str               # "ecdsa-secp256k1-sha256"
    signature_value: str             # base64 ECDSA-SHA256
    signature_pubkey: str            # PEM matching provider.pubkey

    # ----------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self.body)
        d["signature"] = {
            "alg": self.signature_alg,
            "value": self.signature_value,
            "pubkey": self.signature_pubkey,
        }
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_file(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    # ----------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIReceipt":
        sig = data.pop("signature")
        body = AIReceiptBody(
            schema=data["schema"],
            job_id=data["job_id"],
            model=ModelRef(**data["model"]),
            provider=ProviderRef(**data["provider"]),
            input_hash=data["input_hash"],
            output_hash=data["output_hash"],
            cost=CostBreakdown(**data["cost"]),
            energy_mj=data["energy_mj"],
            latency_ms=int(data["latency_ms"]),
            dataset_attestation_hash=data.get("dataset_attestation_hash"),
            timestamp_ns=int(data["timestamp_ns"]),
        )
        return cls(
            body=body,
            signature_alg=sig["alg"],
            signature_value=sig["value"],
            signature_pubkey=sig["pubkey"],
        )

    @classmethod
    def from_json(cls, blob: str) -> "AIReceipt":
        return cls.from_dict(json.loads(blob))

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "AIReceipt":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # ----------------------------------------------------------------------

    def verify(self) -> bool:
        """Verify the receipt's signature against the embedded pubkey.

        A forged or tampered receipt fails here. The caller is
        additionally expected to verify the pubkey is one they trust
        (e.g. matches a registered provider in the chain).
        """
        if self.signature_alg != "ecdsa-secp256k1-sha256":
            return False
        if self.signature_pubkey != self.body.provider.pubkey:
            # The signature must be under the very key the receipt
            # claims as the provider's identity. Otherwise anyone who
            # holds ANY valid key could sign for ANY claimed provider.
            return False
        return Wallet.verify(
            self.signature_pubkey,
            self.body.canonical_json(),
            self.signature_value,
        )


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def make_receipt(
    *,
    job_id: str,
    model_id: str,
    model_hash: str,
    model_kind: str = "llm",
    provider: ProviderRef,
    input_bytes: bytes,
    output_bytes: bytes,
    cost_plg: Decimal,
    cost_usd_estimate: Decimal = Decimal("0"),
    energy_mj: Decimal = Decimal("0"),
    latency_ms: int = 0,
    dataset_attestation_hash: Optional[str] = None,
    timestamp_ns: Optional[int] = None,
    signer: Optional[Wallet] = None,
) -> AIReceipt:
    """Construct + sign a receipt in one call. `signer` must hold the
    private key matching `provider.pubkey`; if absent we raise rather
    than emit an unsigned receipt (no silent insecurity).
    """
    if signer is None:
        raise ValueError("make_receipt requires a signer Wallet")
    body = AIReceiptBody(
        schema=SCHEMA_VERSION,
        job_id=str(job_id),
        model=ModelRef(id=model_id, hash=model_hash, kind=model_kind),
        provider=provider,
        input_hash=_sha256(input_bytes),
        output_hash=_sha256(output_bytes),
        cost=CostBreakdown(
            plg=str(Decimal(cost_plg)),
            usd_estimate=str(Decimal(cost_usd_estimate)),
        ),
        energy_mj=str(Decimal(energy_mj)),
        latency_ms=int(latency_ms),
        dataset_attestation_hash=dataset_attestation_hash,
        timestamp_ns=timestamp_ns if timestamp_ns is not None else _now_ns(),
    )
    canonical = body.canonical_json()
    sig = signer.sign(canonical)
    pub_pem = signer.export_keys()["public"]
    if pub_pem != provider.pubkey:
        raise ValueError(
            "signer wallet pubkey does not match provider.pubkey; "
            "receipts must be signed by the very key claimed as the "
            "provider identity"
        )
    return AIReceipt(
        body=body,
        signature_alg="ecdsa-secp256k1-sha256",
        signature_value=sig,
        signature_pubkey=pub_pem,
    )


__all__ = [
    "SCHEMA_VERSION",
    "ModelRef",
    "ProviderRef",
    "CostBreakdown",
    "AIReceiptBody",
    "AIReceipt",
    "make_receipt",
]
