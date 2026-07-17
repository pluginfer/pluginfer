"""
Gossip Protocol Handler
=======================
Floods (Block, TX, custom) messages across the mesh with seen-cache
deduplication and TTL hop-limit.

W24 hardening: every envelope is now ECDSA-signed by its origin
node's wallet. Receivers verify the signature against the embedded
pubkey before accepting or re-flooding the envelope. Unsigned or
invalid-signature envelopes are rejected.

Why: the previous version's `origin` field was attacker-controlled.
A LAN attacker could forge a gossip claiming to be from any node
(e.g. "I am the coordinator at 192.168.1.42:9999"), and workers
would re-broadcast and act on it. Mesh hijack was one packet away.
"""

import hashlib
import json
import logging
import threading
import time
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


class GossipProtocol:
    def __init__(self, node_id: str, broadcast_callback,
                 wallet=None, require_signed: bool = True):
        """
        :param node_id: Local Node ID
        :param broadcast_callback: function(payload: dict, exclude_sender: bool) -> None
        :param wallet: core.tokenomics.Wallet — used to sign outgoing
            envelopes. If None, outgoing envelopes are unsigned (alpha
            mode); receivers running with require_signed=True will drop
            them.
        :param require_signed: if True, incoming envelopes without a
            valid signature are dropped. Default True for production
            safety; set False only for transitional networks where
            some peers are still on the unsigned protocol.
        """
        self.node_id = node_id
        self.broadcast_callback = broadcast_callback
        self.wallet = wallet
        self.require_signed = require_signed
        self.seen_messages: Set[str] = set()
        # Sliding-window TTL for the seen-cache: each entry stamped
        # with insertion time so cleanup evicts old ones rather than
        # nuking the whole set on overflow (TODO sec6 finding).
        self._seen_at: Dict[str, float] = {}
        self.lock = threading.Lock()

        # Cleanup thread
        threading.Thread(target=self._cleanup_loop, daemon=True).start()

    # ---- canonical bytes for signing -------------------------------------
    @staticmethod
    def _canonical_body(envelope: Dict[str, Any]) -> bytes:
        """Sort-keys, separators-tight JSON of the body fields ONLY
        (excluding sig + pubkey + ttl, which mutate per hop)."""
        body = {k: v for k, v in envelope.items()
                if k not in ("signature", "signer_pubkey", "ttl")}
        return json.dumps(body, sort_keys=True, separators=(",", ":"),
                          default=str).encode("utf-8")

    def _sign(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        """Attach signer_pubkey + ECDSA signature over canonical body."""
        if self.wallet is None:
            return envelope
        canon = self._canonical_body(envelope)
        envelope["signer_pubkey"] = self.wallet.public_key_pem
        envelope["signature"] = self.wallet.sign(canon.decode("utf-8"))
        return envelope

    @staticmethod
    def _verify(envelope: Dict[str, Any]) -> bool:
        """Verify the envelope's signature against its embedded pubkey."""
        from .tokenomics import Wallet
        sig = envelope.get("signature")
        pubkey_pem = envelope.get("signer_pubkey")
        if not sig or not pubkey_pem:
            return False
        canon = GossipProtocol._canonical_body(envelope).decode("utf-8")
        try:
            return Wallet.verify(pubkey_pem, canon, sig)
        except Exception as e:
            logger.debug("gossip signature verify error: %s", e)
            return False

    # ---- producer side ---------------------------------------------------
    def broadcast(self, message_type: str, payload: Dict[str, Any],
                  origin: Optional[str] = None) -> None:
        """Broadcast a signed message to all connected peers."""
        if origin is None:
            origin = self.node_id

        # Content-derived id so identical messages dedupe.
        msg_content = json.dumps(payload, sort_keys=True, default=str)
        msg_hash = hashlib.sha256(
            f"{message_type}:{msg_content}".encode()
        ).hexdigest()

        with self.lock:
            if msg_hash in self.seen_messages:
                return
            self.seen_messages.add(msg_hash)
            self._seen_at[msg_hash] = time.time()

        envelope: Dict[str, Any] = {
            "type": "GOSSIP",
            "gossip_type": message_type,
            "origin": origin,
            "id": msg_hash,
            "payload": payload,
            "ttl": 10,
            "ts": time.time(),
        }
        envelope = self._sign(envelope)

        logger.debug("[GOSSIP] Broadcasting %s %s",
                     message_type, msg_hash[:8])
        self.broadcast_callback(envelope, exclude_sender=False)

    # ---- consumer side ---------------------------------------------------
    def handle_gossip(self, envelope: Dict[str, Any]):
        """
        Process incoming gossip. Returns (is_new, payload).

        Drops the envelope (returns (False, None)) if:
          * signature missing / invalid (when require_signed=True);
          * already seen;
          * TTL expired.
        """
        if self.require_signed and not self._verify(envelope):
            logger.warning(
                "[GOSSIP] dropping envelope id=%s from origin=%r — "
                "missing or invalid signature",
                str(envelope.get("id"))[:8], envelope.get("origin"),
            )
            return False, None

        msg_id = envelope.get("id")
        ttl = envelope.get("ttl", 0)

        with self.lock:
            if msg_id in self.seen_messages:
                return False, None
            self.seen_messages.add(msg_id)
            self._seen_at[msg_id] = time.time()

        if ttl <= 0:
            logger.debug("[GOSSIP] %s expired (TTL=0)", str(msg_id)[:8])
            return False, None

        # Decrement TTL and re-flood. NOTE: the signature was made over
        # the body fields excluding `ttl`, so re-broadcasting with a
        # decremented TTL keeps the signature valid for downstream peers.
        envelope["ttl"] = ttl - 1
        self.broadcast_callback(envelope, exclude_sender=True)

        return True, envelope.get("payload")

    # ---- background cleanup ----------------------------------------------
    _SEEN_RETENTION_S = 1800     # 30 minutes — long enough that the
                                 # network has fully drained any in-flight
                                 # message at TTL=10.

    def _cleanup_loop(self):
        """Sliding-window seen-cache cleanup."""
        while True:
            time.sleep(60)
            cutoff = time.time() - self._SEEN_RETENTION_S
            with self.lock:
                stale = [k for k, t in self._seen_at.items() if t < cutoff]
                for k in stale:
                    self.seen_messages.discard(k)
                    self._seen_at.pop(k, None)
            if stale:
                logger.debug("[GOSSIP] cleaned %d stale seen-cache entries",
                             len(stale))
