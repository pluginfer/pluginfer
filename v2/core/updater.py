"""
Auto-Updater
============
Real implementation: polls GitHub Releases (or any compatible JSON
manifest), compares semver, and downloads to a staging directory
when an update is available. Application of the update (replacing
the running binary) is left to a separate side-loaded helper
because Windows can't replace a running .exe in-process.

Previous version was a mock at every step:
    * `REMOTE_VERSION_URL = "https://api.pluginfer.network/latest_version"  # Mock`
    * `MOCK REMOTE CHECK ... mock_remote_info = {...}`
    * `# Simulate download / for i in range(11): time.sleep(0.1)`
    * `[Mock] Renaming ...`

A "self-updater" that mock-reports a fake new version and pretends
to install it isn't a feature — it's a security misdirection. Now
it queries a real source and refuses to do anything we can't verify.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CURRENT_VERSION = "0.9.0"
DEFAULT_MANIFEST_URL = os.environ.get(
    "PLUGINFER_UPDATE_MANIFEST",
    "https://raw.githubusercontent.com/pluginfer/releases/main/latest.json",
)

# Release pubkey baked into the binary at build time. The expected wire
# format is a SECP256K1 public key in PEM. Anyone publishing an update
# must sign the manifest *body* (the JSON minus the "manifest_signature"
# field, sorted-keys canonicalised) with the matching private key.
#
# At v3.0-alpha there is no production release pubkey yet — set this to
# None to mean "not configured". When None, the updater still checks
# the artifact SHA-256 against the manifest's claim, but the manifest
# itself is unsigned (the original behaviour). To enable signed-update
# verification for a release, set:
#     PLUGINFER_RELEASE_PUBKEY_PEM=<full PEM-encoded public key>
# in the build environment (or override at runtime).
RELEASE_PUBKEY_PEM: Optional[str] = os.environ.get("PLUGINFER_RELEASE_PUBKEY_PEM")


class AutoUpdater:
    """
    Manifest format expected at PLUGINFER_UPDATE_MANIFEST:
        {
          "version":        "1.0.0",
          "release_notes":  "...",
          "download_url":   "https://.../pluginfer-1.0.0.exe",
          "sha256":         "...",
          "min_version":    "0.5.0",      // optional
          "critical":       false
        }
    Manifest must be served over HTTPS. SHA-256 in the manifest is
    cross-checked against the downloaded artifact before staging.
    """

    DOWNLOAD_TIMEOUT_S = 60.0
    MANIFEST_TIMEOUT_S = 5.0

    def __init__(self, current_version: str = CURRENT_VERSION,
                 manifest_url: str = DEFAULT_MANIFEST_URL,
                 staging_dir: Optional[str] = None):
        self.current_version = current_version
        self.manifest_url = manifest_url
        self.staging_dir = staging_dir or os.path.join(
            tempfile.gettempdir(), "pluginfer-update")
        os.makedirs(self.staging_dir, exist_ok=True)
        self.pending_update: Optional[Dict[str, Any]] = None

    def check_for_updates(self) -> Optional[Dict[str, Any]]:
        """
        Returns the manifest if a newer, **signature-verified** version
        is available, else None.

        Layered defense:
          1. Manifest must be served over HTTPS.
          2. Manifest's `manifest_signature` field must verify against
             the configured `RELEASE_PUBKEY_PEM` (if one is set). If a
             release pubkey is configured but the manifest is unsigned
             or the signature fails, the manifest is REJECTED — even
             if the artifact's SHA-256 matches its claim. This closes
             the MITM-swap-both-fields attack.
          3. If RELEASE_PUBKEY_PEM is None (alpha / unsigned mode),
             the updater logs an explicit warning that the manifest
             is unverified and only sha256 against artifact is checked.
          4. Artifact SHA-256 is verified against the manifest claim
             at download time (`download_update`).
        """
        if not self.manifest_url.startswith("https://"):
            logger.warning("Refusing to fetch update manifest over non-HTTPS URL.")
            return None
        try:
            with urllib.request.urlopen(self.manifest_url,
                                         timeout=self.MANIFEST_TIMEOUT_S) as r:
                manifest = json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as e:
            logger.info("Update check failed (%s); will retry later.", e)
            return None
        except Exception as e:
            logger.warning("Update manifest invalid: %s", e)
            return None

        if not isinstance(manifest, dict) or "version" not in manifest:
            return None

        # Manifest signature verification.
        if not self._verify_manifest_signature(manifest):
            return None

        if self._compare_versions(manifest["version"], self.current_version) <= 0:
            logger.info("System is up to date (current: %s)", self.current_version)
            return None

        logger.info("Update available: %s -> %s",
                    self.current_version, manifest["version"])
        self.pending_update = manifest
        return manifest

    @staticmethod
    def _verify_manifest_signature(manifest: Dict[str, Any]) -> bool:
        """
        Verify ECDSA signature over the manifest body using the
        baked-in `RELEASE_PUBKEY_PEM`. Returns True if verified, OR if
        the alpha-mode unsigned-manifest concession applies (no
        pubkey configured).
        """
        sig_b64 = manifest.get("manifest_signature")

        if RELEASE_PUBKEY_PEM is None:
            # Alpha mode. Loudly warn but allow through.
            logger.warning(
                "Updater: RELEASE_PUBKEY_PEM not configured; manifest "
                "signature cannot be verified. MITM swapping both "
                "download_url and sha256 would be undetectable. Set "
                "PLUGINFER_RELEASE_PUBKEY_PEM before any production "
                "rollout (W31)."
            )
            return True

        if not sig_b64:
            logger.error(
                "Updater: RELEASE_PUBKEY_PEM is configured but manifest "
                "carries no `manifest_signature` field — REJECTING."
            )
            return False

        # Strip the signature field, canonicalise the body, verify.
        body = {k: v for k, v in manifest.items() if k != "manifest_signature"}
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))

        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.exceptions import InvalidSignature
            pubkey = serialization.load_pem_public_key(
                RELEASE_PUBKEY_PEM.encode("utf-8")
            )
            sig = base64.b64decode(sig_b64)
            pubkey.verify(sig, canonical.encode("utf-8"),
                          ec.ECDSA(hashes.SHA256()))
            logger.info("Updater: manifest signature verified.")
            return True
        except InvalidSignature:
            logger.error("Updater: manifest signature INVALID — REJECTING.")
            return False
        except Exception as e:
            logger.error("Updater: signature verification failed (%s) "
                         "— REJECTING.", e)
            return False

    def download_update(self, manifest: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Download to staging_dir; verify SHA-256; return staged file path."""
        manifest = manifest or self.pending_update
        if not manifest:
            return None
        url = manifest.get("download_url")
        expected_hash = manifest.get("sha256")
        if not url or not url.startswith("https://"):
            logger.warning("Refusing to download from non-HTTPS URL.")
            return None
        if not expected_hash:
            logger.warning("Refusing to download update without sha256 in manifest.")
            return None

        version = manifest.get("version", "unknown")
        target = os.path.join(self.staging_dir, f"pluginfer-{version}.staged")
        try:
            with urllib.request.urlopen(url, timeout=self.DOWNLOAD_TIMEOUT_S) as r, \
                 open(target, "wb") as f:
                hasher = hashlib.sha256()
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    hasher.update(chunk)
                    f.write(chunk)
            actual_hash = hasher.hexdigest()
        except Exception as e:
            logger.error("Download failed: %s", e)
            return None

        if actual_hash != expected_hash.lower():
            logger.error("SHA-256 mismatch on update; deleting. expected=%s got=%s",
                         expected_hash, actual_hash)
            try:
                os.remove(target)
            except OSError:
                pass
            return None
        logger.info("Update %s staged at %s", version, target)
        return target

    def perform_update(self, manifest: Dict[str, Any], force: bool = False) -> bool:
        """
        We DOWNLOAD to staging but never rename the running binary. Replacing
        a running executable safely requires a side-loaded helper (Windows
        can't replace itself; macOS code-signs reset; Linux needs perms).
        Real shipping requires a separate `pluginfer-updater.exe` helper.

        Returns True if downloaded and ready for swap, False otherwise.
        """
        if not force and not manifest.get("critical"):
            logger.info("Non-critical update available. Awaiting user approval.")
            return False
        staged = self.download_update(manifest)
        return staged is not None

    @staticmethod
    def _compare_versions(v1: str, v2: str) -> int:
        def parse(v: str): return [int(x) for x in v.split(".") if x.isdigit()]
        a, b = parse(v1), parse(v2)
        return (a > b) - (a < b)
