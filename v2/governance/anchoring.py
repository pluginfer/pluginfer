"""External anchoring of the receipt-chain head (HG13j).

The signed hash chain (see ``signing.py``) stops outsiders and naive
edits, but a motivated OPERATOR who holds the signing key can still
rewrite the whole chain and re-sign it. The fix is publication: commit
the chain head, on a schedule, to a public append-only medium the
operator does not control. After that, any rewrite contradicts a head
the world already witnessed.

Method: OpenTimestamps (https://opentimestamps.org). Free public
calendar servers aggregate submitted digests into a Merkle tree whose
root is committed into a Bitcoin transaction. The proof artifact
(``.ots``) is a standard, widely-supported format verifiable with the
independent ``opentimestamps-client`` tooling — the operator cannot
forge or backdate one without breaking Bitcoin itself.

Honest scope, stated plainly:

* OPT-IN. Anchoring sends the 32-byte chain head (and nothing else —
  no spend data, no receipt contents) to public calendar servers.
  Airgapped deployments simply leave it off.
* FAIL-OPEN. An anchoring failure never blocks spend enforcement or
  receipt emission — this is audit hardening, not a spend control.
  Every attempt, success or failure, is journaled.
* A fresh ``.ots`` proof is PENDING: calendars batch digests into
  Bitcoin within hours. ``ots upgrade`` later completes the proof to a
  full Bitcoin attestation. Records say "pending" rather than claiming
  "on the blockchain" the moment the digest leaves the building.
* One ``.ots`` file is written PER calendar that accepted the digest
  (independent proofs, no tree merging) — any single one suffices.

Verification (third party, no trust in this gateway):

    pip install opentimestamps-client
    ots upgrade <proof>.ots          # completes once Bitcoin-attested
    ots verify -d <chain_head_hex> <proof>.ots

then confirm the same head via ``GET /v1/receipts/verify`` and check
each receipt's Ed25519 signature against the published public key.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("pluginfer.governance.anchoring")


# --------------------------------------------------------------------------
# OpenTimestamps detached-proof serialization
# --------------------------------------------------------------------------

# These constants are the OpenTimestamps file format v1, byte-for-byte
# (opentimestamps-client `DetachedTimestampFile`): header magic, varuint
# major version 1, the SHA256 file-hash op tag, then the 32-byte digest,
# then the calendar's serialized timestamp operations verbatim.
OTS_HEADER_MAGIC = (b"\x00OpenTimestamps\x00\x00Proof\x00"
                    b"\xbf\x89\xe2\xe8\x84\xe8\x92\x94")
OTS_VERSION = b"\x01"
OTS_SHA256_TAG = b"\x08"

DEFAULT_CALENDARS: Tuple[str, ...] = (
    "https://a.pool.opentimestamps.org",
    "https://b.pool.opentimestamps.org",
    "https://a.pool.eternitywall.com",
    "https://ots.btc.catallaxy.com",
)


class AnchorSubmitError(RuntimeError):
    """A calendar server did not accept the digest."""


def build_detached_ots(digest: bytes, calendar_ops: bytes) -> bytes:
    """Assemble a standard detached ``.ots`` proof file for a sha256
    digest from a calendar's response bytes."""
    if len(digest) != 32:
        raise ValueError(f"digest must be 32 bytes, got {len(digest)}")
    if not calendar_ops:
        raise ValueError("calendar response is empty")
    return (OTS_HEADER_MAGIC + OTS_VERSION + OTS_SHA256_TAG + digest
            + calendar_ops)


def submit_to_calendar(url: str, digest: bytes, *,
                       timeout: float = 10.0) -> bytes:
    """POST the digest to one calendar's ``/digest`` endpoint and return
    its serialized timestamp operations."""
    # Calendar URLs are operator config, but a config typo must not turn
    # into a file:// read or a custom-scheme surprise — http(s) only.
    scheme = url.split(":", 1)[0].lower()
    if scheme not in ("http", "https"):
        raise AnchorSubmitError(f"{url}: calendar URL must be http(s)")
    req = urllib.request.Request(
        url.rstrip("/") + "/digest",
        data=digest,
        headers={
            "Accept": "application/vnd.opentimestamps.v1",
            "User-Agent": "pluginfer-signet-anchor/1.0",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(  # nosec B310 — scheme allowlisted above
                req, timeout=timeout) as resp:
            body = resp.read()
    except (urllib.error.URLError, TimeoutError, ConnectionError,
            OSError) as e:
        raise AnchorSubmitError(f"{url}: {e}") from e
    if not body:
        raise AnchorSubmitError(f"{url}: empty response")
    return body


# --------------------------------------------------------------------------
# AnchorManager — journaled, signed anchor records + proof files
# --------------------------------------------------------------------------

_SIG_FIELDS = ("signature", "algorithm", "public_key_pem")


class AnchorManager:
    """Anchors chain heads and keeps the journal + proof files.

    ``state_dir`` persistence mirrors the receipt store: proofs land in
    ``<state_dir>/anchors/`` and the journal in ``anchors.jsonl`` there.
    With no state dir (tests / dev) everything is memory-only and says
    so on the record. ``submit_fn`` is the injection point for hermetic
    tests — production uses :func:`submit_to_calendar`.
    """

    def __init__(self, state_dir: Optional[os.PathLike] = None, *,
                 signer: Optional[Any] = None,
                 calendars: Optional[List[str]] = None,
                 submit_fn: Optional[Callable[[str, bytes], bytes]] = None,
                 timeout_s: float = 10.0):
        self.signer = signer
        self.calendars = [c.rstrip("/") for c in
                          (calendars or list(DEFAULT_CALENDARS))]
        self._submit = submit_fn or (
            lambda url, digest: submit_to_calendar(
                url, digest, timeout=self.timeout_s))
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._records: List[Dict[str, Any]] = []
        self.anchors_dir: Optional[Path] = None
        if state_dir is not None:
            self.anchors_dir = Path(state_dir) / "anchors"
            self.anchors_dir.mkdir(parents=True, exist_ok=True)
            self._load_journal()

    # -- persistence -------------------------------------------------------

    def _journal_path(self) -> Optional[Path]:
        return (self.anchors_dir / "anchors.jsonl"
                if self.anchors_dir else None)

    def _load_journal(self) -> None:
        path = self._journal_path()
        if path is None or not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("skipping malformed anchor journal "
                                       "line")
        except OSError as e:
            logger.warning("anchor journal unreadable: %s", e)

    def _append_journal(self, rec: Dict[str, Any]) -> None:
        path = self._journal_path()
        if path is None:
            return
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        except OSError as e:
            logger.warning("could not persist anchor record: %s", e)

    # -- anchoring ---------------------------------------------------------

    def anchor(self, head_hex: str, receipt_count: int) -> Dict[str, Any]:
        """Submit ``head_hex`` to every configured calendar. Never
        raises — failures are recorded on the returned record."""
        rec: Dict[str, Any] = {
            "anchor_id": "anc-" + secrets.token_urlsafe(9),
            "ts": time.time(),
            "chain_head_sha256": head_hex,
            "receipt_count": receipt_count,
            "method": "opentimestamps",
            "status": "pending",   # -> Bitcoin-attested after `ots upgrade`
            "calendars": [],
            "persisted": self.anchors_dir is not None,
            "verify_hint": ("pip install opentimestamps-client; "
                            "ots upgrade <proof>.ots; "
                            f"ots verify -d {head_hex} <proof>.ots"),
        }
        try:
            digest = bytes.fromhex(head_hex)
        except ValueError:
            rec["ok"] = False
            rec["error"] = "chain head is not valid hex"
            return self._finish(rec)
        if len(digest) != 32:
            rec["ok"] = False
            rec["error"] = "chain head is not a sha256 digest"
            return self._finish(rec)

        ok_count = 0
        for i, cal in enumerate(self.calendars):
            entry: Dict[str, Any] = {"url": cal, "index": i}
            try:
                ops = self._submit(cal, digest)
                proof = build_detached_ots(digest, ops)
                fname = f"{rec['anchor_id']}_{i}.ots"
                if self.anchors_dir is not None:
                    (self.anchors_dir / fname).write_bytes(proof)
                    entry["proof_file"] = fname
                else:
                    entry["proof_file"] = None
                entry["ok"] = True
                entry["proof_bytes"] = len(proof)
                ok_count += 1
            except (AnchorSubmitError, ValueError, OSError) as e:
                entry["ok"] = False
                entry["error"] = str(e)
                logger.warning("anchor submit failed: %s", e)
            rec["calendars"].append(entry)
        rec["ok"] = ok_count > 0
        if ok_count == 0:
            rec["error"] = "no calendar accepted the digest"
        return self._finish(rec)

    def _finish(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        if self.signer is not None:
            try:
                body = json.dumps(
                    {k: v for k, v in rec.items() if k not in _SIG_FIELDS},
                    sort_keys=True, default=str)
                rec["signature"] = self.signer.sign(body)
                rec["algorithm"] = getattr(self.signer, "algorithm",
                                           "unknown")
                rec["public_key_pem"] = getattr(self.signer,
                                                "public_key_pem", None)
            except Exception as e:
                logger.warning("anchor record signing failed: %s", e)
        with self._lock:
            self._records.append(rec)
        self._append_journal(rec)
        return rec

    def anchor_if_new(self, head_hex: str,
                      receipt_count: int) -> Optional[Dict[str, Any]]:
        """Anchor only when the head moved past the last SUCCESSFUL
        anchor — a quiet gateway costs the calendars nothing."""
        if head_hex == "0" * 64:
            return None            # genesis: nothing to prove yet
        last = self.last_success()
        if last is not None and last.get("chain_head_sha256") == head_hex:
            return None
        return self.anchor(head_hex, receipt_count)

    # -- queries -----------------------------------------------------------

    def records(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._records[-max(1, min(limit, 1000)):])

    def last_success(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            for rec in reversed(self._records):
                if rec.get("ok"):
                    return rec
        return None

    def find_proof(self, anchor_id: str,
                   index: int) -> Optional[Path]:
        """Resolve a proof file via the journal — the filename comes
        from OUR record, never from a caller-supplied path."""
        if self.anchors_dir is None:
            return None
        with self._lock:
            recs = list(self._records)
        for rec in recs:
            if rec.get("anchor_id") != anchor_id:
                continue
            for entry in rec.get("calendars", []):
                if entry.get("index") == index and entry.get("ok"):
                    fname = entry.get("proof_file")
                    if not fname:
                        return None
                    path = self.anchors_dir / fname
                    return path if path.exists() else None
        return None


# --------------------------------------------------------------------------
# AnchorScheduler — periodic anchoring on its own daemon thread
# --------------------------------------------------------------------------


class AnchorScheduler(threading.Thread):
    """Anchors the current head every ``interval_s`` when it changed.

    ``head_fn`` returns ``(head_hex, receipt_count)`` under the
    gateway's own lock. The thread is a daemon and holds only weak
    coupling to the app — ``stop()`` ends it promptly.
    """

    def __init__(self, manager: AnchorManager,
                 head_fn: Callable[[], Tuple[str, int]],
                 *, interval_s: float = 3600.0, tick_s: float = 15.0):
        super().__init__(name="signet-anchor", daemon=True)
        self.manager = manager
        self.head_fn = head_fn
        self.interval_s = max(1.0, float(interval_s))
        self.tick_s = max(0.05, min(tick_s, self.interval_s))
        self._stop = threading.Event()
        self._last_attempt = 0.0

    def run(self) -> None:
        while not self._stop.wait(self.tick_s):
            now = time.monotonic()
            if now - self._last_attempt < self.interval_s:
                continue
            self._last_attempt = now
            try:
                head, count = self.head_fn()
                self.manager.anchor_if_new(head, count)
            except Exception as e:      # fail-open: audit hardening only
                logger.warning("scheduled anchoring failed: %s", e)

    def stop(self) -> None:
        self._stop.set()


__all__ = [
    "AnchorManager",
    "AnchorScheduler",
    "AnchorSubmitError",
    "DEFAULT_CALENDARS",
    "OTS_HEADER_MAGIC",
    "build_detached_ots",
    "submit_to_calendar",
]
