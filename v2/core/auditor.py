"""
The Auditor (Compliance Agent)

Self-audit of node integrity:
  1. Ledger consistency (verify_chain).
  2. File-system integrity (W30): when a release-signed manifest is
     available, file hashes are compared against the manifest. The
     manifest itself is ECDSA-verified using the same release pubkey
     infrastructure as the updater (W31). Without a signed manifest,
     the auditor falls back to a self-signed runtime snapshot AND
     marks the report attestation as 'self-signed-snapshot' so the
     reader knows not to trust it as a tamper-evidence proof.
  3. Subsystem loadability — purely informational.

The compliance-report generator returns measured facts only — no
'Gold Standard' / 'ENFORCED' claims (those were stripped in 89ad0af).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Where to look for the release-signed manifest. Override via
# `PLUGINFER_RELEASE_MANIFEST_PATH` env var; default is alongside the
# core package so PyInstaller bundles ship it.
_DEFAULT_MANIFEST_PATH = os.environ.get(
    "PLUGINFER_RELEASE_MANIFEST_PATH",
    os.path.join(os.path.dirname(__file__), "_release_manifest.json"),
)


def _verify_manifest_signature(manifest: Dict[str, Any],
                               pubkey_pem: str) -> bool:
    """Verify the manifest's `manifest_signature` field against
    `pubkey_pem`. Returns True only if the canonical body (manifest
    minus signature) hashes to the signed payload."""
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
    except Exception as e:
        logger.error("[AUDITOR] cryptography missing: %s", e)
        return False

    sig_b64 = manifest.get("manifest_signature")
    if not sig_b64:
        return False
    body = {k: v for k, v in manifest.items()
            if k != "manifest_signature"}
    canonical = json.dumps(body, sort_keys=True,
                           separators=(",", ":")).encode()
    try:
        pub = serialization.load_pem_public_key(pubkey_pem.encode())
        sig = base64.b64decode(sig_b64)
        pub.verify(sig, canonical, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False
    except Exception as e:
        logger.error("[AUDITOR] manifest signature verify failed: %s", e)
        return False


class SystemAuditor:
    def __init__(self, ledger, core_path: str = "./core",
                 manifest_path: Optional[str] = None,
                 release_pubkey_pem: Optional[str] = None):
        self.ledger = ledger
        self.core_path = core_path
        self.manifest_path = manifest_path or _DEFAULT_MANIFEST_PATH
        # Production deploy MUST set the env var below to a build-time
        # release pubkey so a tampered manifest can't be silently
        # accepted (mirrors updater.py W31 design).
        self.release_pubkey_pem = (
            release_pubkey_pem
            or os.environ.get("PLUGINFER_RELEASE_PUBKEY_PEM")
        )
        self.known_checksums: Dict[str, str] = {}
        self.attestation_mode: str = "uninitialised"
        self.last_audit_time = 0.0
        self.audit_interval = 300

        self._load_attestation_source()

    # ------------------------------------------------------------------
    # Initialisation: prefer signed manifest, fall back to snapshot.
    # ------------------------------------------------------------------
    def _load_attestation_source(self) -> None:
        if self._try_load_signed_manifest():
            return
        # Honest fallback — the runtime snapshot can be tampered before
        # we ever ran. Record the attestation mode loudly.
        self._snapshot_core_files()
        self.attestation_mode = "self-signed-snapshot"
        if self.release_pubkey_pem:
            logger.warning(
                "[AUDITOR] release pubkey configured but no manifest "
                "found at %s; falling back to runtime snapshot. A "
                "pre-launch tamper would NOT be detected.",
                self.manifest_path,
            )
        else:
            logger.info(
                "[AUDITOR] no signed manifest configured (alpha mode); "
                "running with runtime-snapshot integrity. Set "
                "PLUGINFER_RELEASE_MANIFEST_PATH and "
                "PLUGINFER_RELEASE_PUBKEY_PEM in production."
            )

    def _try_load_signed_manifest(self) -> bool:
        """Returns True iff a signed manifest was loaded and verified."""
        if not (self.manifest_path and os.path.exists(self.manifest_path)):
            return False
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            logger.error("[AUDITOR] manifest unreadable: %s", e)
            return False

        files = manifest.get("files") or {}
        if not isinstance(files, dict) or not files:
            logger.error("[AUDITOR] manifest missing 'files' dict")
            return False

        # Signature verification is mandatory if a pubkey is configured.
        if self.release_pubkey_pem:
            if not _verify_manifest_signature(
                manifest, self.release_pubkey_pem,
            ):
                logger.critical(
                    "[AUDITOR] manifest at %s FAILED signature "
                    "verification — refusing to trust it.",
                    self.manifest_path,
                )
                return False
            self.attestation_mode = "release-manifest-verified"
        else:
            # Manifest present but no pubkey to verify it. Loud warning
            # but accept the file's hashes — strictly weaker than full
            # release-signed mode.
            self.attestation_mode = "manifest-unverified"
            logger.warning(
                "[AUDITOR] manifest loaded WITHOUT signature "
                "verification (no PLUGINFER_RELEASE_PUBKEY_PEM). "
                "Set the env var in production builds."
            )

        # Resolve manifest paths relative to core_path's parent so the
        # manifest can be authored in repo-relative form.
        base = os.path.dirname(os.path.abspath(self.core_path))
        self.known_checksums = {
            os.path.normpath(os.path.join(base, p)): h
            for p, h in files.items()
        }
        logger.info(
            "[AUDITOR] loaded %d hashes from signed manifest (%s)",
            len(self.known_checksums), self.attestation_mode,
        )
        return True

    def _snapshot_core_files(self) -> None:
        """Calculate checksums of all .py files under core_path."""
        try:
            for root, _, files in os.walk(self.core_path):
                for f in files:
                    if f.endswith(".py"):
                        path = os.path.join(root, f)
                        self.known_checksums[os.path.normpath(path)] = \
                            self._get_file_hash(path)
            logger.info("[AUDITOR] snapshotted %d core files for "
                        "integrity monitoring",
                        len(self.known_checksums))
        except Exception as e:
            logger.error("[AUDITOR] snapshot failed: %s", e)

    @staticmethod
    def _get_file_hash(path: str) -> str:
        sha256 = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception:
            return "ERROR"

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------
    def perform_audit(self) -> Dict[str, Any]:
        start = time.time()
        report = {
            "status": "PASS",
            "attestation_mode": self.attestation_mode,
            "ledger_integrity": "UNKNOWN",
            "file_integrity": "UNKNOWN",
            "issues": [],
        }

        # 1. Ledger.
        try:
            ok = self.ledger.verify_chain()
        except Exception as e:
            report["issues"].append(f"verify_chain raised: {e}")
            ok = False
        report["ledger_integrity"] = "PASS" if ok else "FAIL"
        if not ok:
            report["status"] = "FAIL"
            report["issues"].append("blockchain integrity check failed")

        # 2. File integrity.
        tampered = []
        for path, expected in self.known_checksums.items():
            if not os.path.exists(path):
                tampered.append(f"MISSING: {path}")
                continue
            actual = self._get_file_hash(path)
            if actual != expected:
                tampered.append(f"MODIFIED: {path}")
        if tampered:
            report["file_integrity"] = "FAIL"
            report["status"] = "FAIL"
            report["issues"].extend(tampered)
        else:
            report["file_integrity"] = "PASS"

        report["duration_ms"] = int((time.time() - start) * 1000)
        self.last_audit_time = time.time()
        logger.info("[AUDITOR] audit complete in %d ms; status=%s",
                    report["duration_ms"], report["status"])
        if report["status"] == "FAIL":
            logger.critical("[AUDITOR] integrity issue: %s",
                            report["issues"])
        return report

    def start_background_audit(self) -> None:
        """Run audit loop in background."""
        def _loop():
            while True:
                time.sleep(self.audit_interval)
                self.perform_audit()
        threading.Thread(target=_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # Compliance report (no marketing claims)
    # ------------------------------------------------------------------
    def generate_compliance_report(self) -> str:
        """
        Produce a measured-facts report on this node's current state.

        This used to claim 'Gold Standard Listing Requirements' /
        'ENFORCED swarm isolation' / 'Active Sybil protection'
        regardless of actual state — fabricated certification text
        actionable under securities-fraud rules. Stripped 89ad0af.

        New report contains only facts derivable from the live audit
        plus the attestation mode (release-manifest vs runtime
        snapshot) so the reader knows the strength of the claim.
        """
        report = self.perform_audit()
        sandbox_present = self._module_loadable("core.secure_sandbox")
        sentinel_present = self._module_loadable("core.ai_sentinel")
        consensus_present = self._module_loadable("core.bft_consensus")
        privacy_present = self._module_loadable("core.privacy")

        compliance_doc = {
            "timestamp": time.time(),
            "node_version": "3.0-alpha",
            "attestation_mode": self.attestation_mode,
            "audit_result": report["status"],
            "measured_metrics": {
                "blockchain_height": self.ledger.get_height(),
                "ledger_integrity": report["ledger_integrity"],
                "file_system_integrity": report["file_integrity"],
                "files_monitored": len(self.known_checksums),
                "issues_count": len(report.get("issues", [])),
            },
            "subsystems_loadable": {
                "secure_sandbox": sandbox_present,
                "ai_sentinel": sentinel_present,
                "bft_consensus": consensus_present,
                "zk_privacy": privacy_present,
            },
            "disclaimers": [
                "This report is a self-audit of the local node only.",
                "Subsystem 'loadable' status reflects only that the "
                "module imports — NOT that it is correctly configured "
                "or providing the security properties it claims.",
                "Attestation mode 'self-signed-snapshot' means a "
                "pre-launch tamper would NOT have been detected; only "
                "'release-manifest-verified' is tamper-evident.",
                "This document is NOT a third-party certification, NOT "
                "an exchange listing report, and MUST NOT be presented "
                "as either.",
            ],
        }
        return json.dumps(compliance_doc, indent=4)

    @staticmethod
    def _module_loadable(modname: str) -> bool:
        try:
            __import__(modname)
            return True
        except Exception:
            return False


# ----------------------------------------------------------------------
# Manifest authoring helper — used by build pipeline
# ----------------------------------------------------------------------
def build_signed_manifest(
    core_path: str,
    private_key_pem: str,
    output_path: str,
    version: str = "0.0.0",
) -> Dict[str, Any]:
    """Build a release manifest for a build pipeline.

    Walks `core_path`, computes SHA-256 for every .py file, signs the
    canonical body with `private_key_pem` (PEM ECDSA SECP256K1 / P-256
    private key), writes to `output_path`. Returns the manifest dict.

    This is the build-time half of W30; the runtime auditor verifies
    the result via `_verify_manifest_signature`.
    """
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    files = {}
    base = os.path.abspath(core_path)
    for root, _, fs in os.walk(core_path):
        for f in fs:
            if f.endswith(".py"):
                p = os.path.join(root, f)
                rel = os.path.relpath(p, os.path.dirname(base))
                files[rel.replace(os.sep, "/")] = \
                    SystemAuditor._get_file_hash(p)
    body = {"version": version, "files": files,
            "built_at": time.time()}
    canonical = json.dumps(body, sort_keys=True,
                           separators=(",", ":")).encode()
    priv = serialization.load_pem_private_key(
        private_key_pem.encode(), password=None,
    )
    signature = priv.sign(canonical, ec.ECDSA(hashes.SHA256()))
    manifest = dict(body)
    manifest["manifest_signature"] = base64.b64encode(signature).decode()
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(manifest, out, indent=2)
    return manifest
