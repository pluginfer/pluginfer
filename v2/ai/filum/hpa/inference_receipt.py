"""§D1 Universal Inference Receipt Protocol — cryptographic provenance
for every AI output produced anywhere on the Pluginfer mesh.

The thesis: in 2026 you cannot tell if an image, a document, an
audio clip, or a chunk of code came from a human or an AI — and
even if you know it came from an AI, you cannot tell *which* AI,
*on what input*, *running on what weights*. The "AI provenance
crisis" is now a top-level concern of every regulator, every
publisher, every legal jurisdiction.

Centralised AI providers cannot solve this credibly because they
control both the model and the audit log; the fox guards the
henhouse. A *substrate-level* provenance protocol — built into the
compute mesh that runs the inference, signed by an independent
node, anchored to a public chain — is the only credible answer.

Pluginfer is positioned to be that substrate. Every inference run
on the Pluginfer mesh produces a Universal Inference Receipt:

    receipt = {
        "model_weights_sha256":  "a1b2…",   # which weights produced this
        "input_sha256":          "c3d4…",   # what input was given
        "output_sha256":         "e5f6…",   # what output was produced
        "model_metadata":        {...},      # name, version, parameter count
        "node_pubkey":           "…",        # which node ran the inference
        "timestamp":             1715300000.0,
        "compute_proof":         "…",        # optional: chain of layer hashes
        "policy_class":          "general",  # safety class assigned
    }
    signature = ed25519(node_privkey, canonical(receipt))

Receipts can be:

* **Verified** by anyone with the node's public key.
* **Anchored** on-chain (every N receipts -> Merkle root -> chain
  transaction). This is the §A1 receipt extension — same chain,
  larger payload type.
* **Searched** by any of {input_sha256, output_sha256,
  model_weights_sha256} — the user can ask "did model X ever
  produce this output?" without revealing the input.
* **Repudiated** by the node *only* by proving the input was
  malformed (the receipt itself is non-repudiable; only the
  inference's *validity* can be challenged via §A8 quorum
  inference).

Privacy considerations: input/output hashes leak nothing about
content. If the buyer wants to retain the right to *prove later*
that they ran a specific input, they keep their copy. If they
prefer *full forgetfulness*, they discard it; the hash on the
chain is then meaningless to anyone without the original.

This module ships:

* ``InferenceReceipt`` dataclass
* ``issue_receipt(...)`` for inference nodes
* ``verify_receipt(...)`` for any third party
* ``ReceiptLog`` — append-only log + Merkle aggregator for chain
  anchoring (every 256 receipts -> 1 root -> 1 chain tx)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey, Ed25519PublicKey,
    )
    _HAS_ED25519 = True
except Exception:                                                  # pragma: no cover
    _HAS_ED25519 = False


RECEIPT_PROTOCOL_VERSION = 1


# ---------- the receipt ----------------------------------------------------

@dataclass
class InferenceReceipt:
    protocol: int = RECEIPT_PROTOCOL_VERSION
    receipt_id: str = ""             # sha256-prefix of the canonical body
    model_weights_sha256: str = ""
    input_sha256: str = ""
    output_sha256: str = ""
    model_metadata: dict = field(default_factory=dict)
    node_pubkey: str = ""            # hex
    timestamp: float = 0.0
    compute_proof: str = ""          # optional: hex sha256 of layer-activation chain
    policy_class: str = "general"
    signature: str = ""              # hex

    def canonical_body(self) -> bytes:
        d = asdict(self)
        d.pop("receipt_id", None)
        d.pop("signature", None)
        return json.dumps(d, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")

    def compute_id(self) -> str:
        return hashlib.sha256(self.canonical_body()).hexdigest()[:32]


# ---------- issue + verify -------------------------------------------------

def issue_receipt(
    *,
    model_weights_bytes: Optional[bytes] = None,
    model_weights_sha256: Optional[str] = None,
    input_text: Optional[str] = None,
    input_bytes: Optional[bytes] = None,
    output_text: Optional[str] = None,
    output_bytes: Optional[bytes] = None,
    model_metadata: Optional[dict] = None,
    node_pubkey_hex: str,
    node_priv_seed: bytes,
    compute_proof_hex: str = "",
    policy_class: str = "general",
    timestamp: Optional[float] = None,
) -> InferenceReceipt:
    """Compose, sign, and return an InferenceReceipt.

    Either ``model_weights_bytes`` OR ``model_weights_sha256`` must be
    supplied; same for input/output. The function hashes whatever is
    raw; what's already a hash is taken at face value.
    """
    weights_h = (
        model_weights_sha256
        if model_weights_sha256
        else (hashlib.sha256(model_weights_bytes or b"").hexdigest()
              if model_weights_bytes is not None else "")
    )
    in_h = _hash_arg(input_text, input_bytes)
    out_h = _hash_arg(output_text, output_bytes)
    r = InferenceReceipt(
        model_weights_sha256=weights_h,
        input_sha256=in_h,
        output_sha256=out_h,
        model_metadata=dict(model_metadata or {}),
        node_pubkey=node_pubkey_hex,
        timestamp=timestamp if timestamp is not None else time.time(),
        compute_proof=compute_proof_hex,
        policy_class=policy_class,
    )
    r.receipt_id = r.compute_id()
    body = r.canonical_body()
    if _HAS_ED25519 and len(node_priv_seed) >= 32:
        priv = Ed25519PrivateKey.from_private_bytes(node_priv_seed[:32])
        sig = priv.sign(body)
        r.signature = sig.hex()
    else:
        # Soft mode: HMAC under the same secret as a fallback for
        # environments without `cryptography`. Tests can run; production
        # runs cryptography.
        sig = hmac.new(node_priv_seed, body, hashlib.sha256).digest()
        r.signature = sig.hex()
    return r


def verify_receipt(
    receipt: InferenceReceipt,
    node_pubkey_or_secret: bytes,
) -> bool:
    """Returns True iff the signature is valid.

    ``node_pubkey_or_secret`` is the 32-byte raw Ed25519 public key when
    cryptography is available; otherwise it's the HMAC secret used at
    issue time.
    """
    if not receipt.signature:
        return False
    body = receipt.canonical_body()
    try:
        sig = bytes.fromhex(receipt.signature)
    except ValueError:
        return False
    if _HAS_ED25519 and len(node_pubkey_or_secret) == 32 and len(sig) == 64:
        try:
            pub = Ed25519PublicKey.from_public_bytes(node_pubkey_or_secret)
            pub.verify(sig, body)
            return True
        except Exception:
            return False
    expected = hmac.new(node_pubkey_or_secret, body, hashlib.sha256).digest()
    return hmac.compare_digest(expected, sig)


def _hash_arg(text: Optional[str], blob: Optional[bytes]) -> str:
    if text is not None:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    if blob is not None:
        return hashlib.sha256(blob).hexdigest()
    return ""


# ---------- the receipt log + Merkle aggregator ----------------------------

@dataclass
class ReceiptLogConfig:
    log_path: str = "ai/filum/_work/receipts.jsonl"
    merkle_batch_size: int = 256
    anchor_path: str = "ai/filum/_work/anchors.jsonl"


class ReceiptLog:
    """Append-only log of receipts + Merkle batching for chain anchoring.

    Every ``merkle_batch_size`` receipts -> compute Merkle root ->
    record an anchor entry that says "this root commits to receipts
    [first_id, last_id]". The anchor entry is what gets posted on
    chain. Verifying a single receipt then reduces to:
    1. Hash the receipt.
    2. Walk the Merkle path from leaf to the anchored root.
    3. Confirm the root is the one on chain.

    All pure-Python; the chain integration is left to caller (the
    existing ``core/anchored_bootstrap.py`` and §A10 bitcoin_anchor
    primitives plug here directly).
    """

    def __init__(self, config: ReceiptLogConfig = ReceiptLogConfig()):
        self.cfg = config
        Path(self.cfg.log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.cfg.anchor_path).parent.mkdir(parents=True, exist_ok=True)
        self._batch: list[InferenceReceipt] = []

    def append(self, receipt: InferenceReceipt) -> Optional[dict]:
        """Persist + buffer for the next Merkle root. Returns an anchor
        record when a batch is sealed; otherwise None."""
        with open(self.cfg.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(receipt), ensure_ascii=False) + "\n")
        self._batch.append(receipt)
        if len(self._batch) >= self.cfg.merkle_batch_size:
            return self._seal_batch()
        return None

    def _seal_batch(self) -> dict:
        leaves = [bytes.fromhex(r.receipt_id) if len(r.receipt_id) % 2 == 0
                  and all(c in "0123456789abcdef" for c in r.receipt_id)
                  else hashlib.sha256(r.canonical_body()).digest()
                  for r in self._batch]
        # Pad to power-of-two for a balanced tree.
        leaves_padded = list(leaves)
        while len(leaves_padded) & (len(leaves_padded) - 1):
            leaves_padded.append(b"\x00" * 32)
        layer = [hashlib.sha256(b).digest() for b in leaves_padded]
        while len(layer) > 1:
            new_layer = []
            for i in range(0, len(layer), 2):
                new_layer.append(hashlib.sha256(layer[i] + layer[i + 1]).digest())
            layer = new_layer
        root_hex = layer[0].hex() if layer else ""
        anchor = {
            "merkle_root": root_hex,
            "first_receipt_id": self._batch[0].receipt_id,
            "last_receipt_id":  self._batch[-1].receipt_id,
            "count": len(self._batch),
            "ts": time.time(),
        }
        with open(self.cfg.anchor_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(anchor) + "\n")
        self._batch.clear()
        return anchor

    def flush(self) -> Optional[dict]:
        """Force-seal whatever's pending, even if below batch size."""
        if not self._batch:
            return None
        return self._seal_batch()
