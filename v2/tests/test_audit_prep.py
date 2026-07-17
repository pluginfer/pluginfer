"""G9 — audit-prep packager."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from tools.audit_prep import build_audit_package, CRYPTO_SURFACE  # noqa: E402


def test_audit_package_bundles_every_crypto_surface_file(tmp_path):
    out = tmp_path / "audit.zip"
    manifest = build_audit_package(out_path=out, include_tests=False)
    counted = sum(1 for c in manifest["contents"] if "sha256" in c)
    # Every claim listed in CRYPTO_SURFACE should resolve to a real
    # file. A missing one is a regression — the file was renamed or
    # deleted without updating audit_prep.
    missing = [
        c["path"] for c in manifest["contents"]
        if c.get("status") == "MISSING"
    ]
    assert not missing, f"missing audit-surface files: {missing}"
    assert counted == len(CRYPTO_SURFACE)


def test_audit_package_includes_contents_md_and_json(tmp_path):
    out = tmp_path / "audit.zip"
    build_audit_package(out_path=out, include_tests=False)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert "CONTENTS.md" in names
        assert "CONTENTS.json" in names
        # CONTENTS.md should be human-readable Markdown.
        body = z.read("CONTENTS.md").decode("utf-8")
        assert body.startswith("# Pluginfer audit package")


def test_audit_package_sha256_matches_unpacked_file(tmp_path):
    """The SHA in the manifest should equal the SHA you'd compute on
    the unpacked file."""
    import hashlib
    out = tmp_path / "audit.zip"
    manifest = build_audit_package(out_path=out, include_tests=False)
    with zipfile.ZipFile(out) as z:
        for entry in manifest["contents"]:
            if "sha256" not in entry:
                continue
            data = z.read(entry["path"])
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]


def test_audit_package_with_tests_includes_test_files(tmp_path):
    out = tmp_path / "audit.zip"
    build_audit_package(out_path=out, include_tests=True)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert any(n.startswith("tests/") for n in names)
