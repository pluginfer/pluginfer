"""W39 — `installer/build_macos.{spec,sh}` + `entitlements.plist` hygiene.

The actual build only runs on macOS (PyInstaller bundles + codesign +
notarytool are platform-specific). What we CAN do hermetically:

  * Verify the spec parses as Python (PyInstaller's `exec()` over the
    spec would catch syntax errors at build time, but we want them
    caught in CI on every commit, on every platform).
  * Verify all paths the spec references exist (the
    `('../v2/ai/filum/architecture.py', 'ai/filum')` style).
  * Verify the hidden-import names exist as importable modules (so a
    `pip install pyinstaller && pyinstaller build_macos.spec` on a
    macOS dev box doesn't trip on a typo three steps in).
  * Verify the entitlements plist parses + has the right
    capabilities declared.
  * Verify the build_macos.sh shell script has the structure we
    expect (the env-var fallback path, the --arch arg, etc.).

Off-keyboard verification (real macOS build, codesign, notarize) is
the MH1+W39 deliverable; these tests catch the configuration regressions
that have nothing to do with macOS hardware.
"""

from __future__ import annotations

import plistlib
import re
import sys
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parents[2] / "installer"
REPO_ROOT = INSTALLER.parent


def _load_spec_text(name: str) -> str:
    p = INSTALLER / name
    assert p.exists(), f"missing spec file: {p}"
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. spec parses as Python
# ---------------------------------------------------------------------------


def test_macos_spec_compiles_as_python():
    """A PyInstaller .spec is just Python that the bootloader exec()s.
    If our spec has a syntax error, no PyInstaller run will catch it
    later -- the user just sees a confusing traceback. compile() here
    so CI catches it on every push."""
    src = _load_spec_text("build_macos.spec")
    compile(src, "build_macos.spec", "exec")


def test_windows_spec_still_compiles_as_python():
    """Sanity check: the existing Windows spec also compiles. Anchors
    the new macOS spec test as not-spurious."""
    src = _load_spec_text("build_windows.spec")
    compile(src, "build_windows.spec", "exec")


# ---------------------------------------------------------------------------
# 2. spec references real paths
# ---------------------------------------------------------------------------


def _extract_relpath_pairs(spec_text: str) -> list[tuple[str, str]]:
    """The datas= list in our specs uses ('../v2/...path...', 'dst').
    Extract every src as a relative path the spec is going to read."""
    # Tolerant regex: capture both single and double quoted strings.
    return re.findall(r"\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]", spec_text)


def test_macos_spec_data_paths_all_exist():
    src = _load_spec_text("build_macos.spec")
    pairs = _extract_relpath_pairs(src)
    # The spec runs from `installer/` so `../v2/...` resolves relative
    # to the installer directory.
    missing = []
    for rel, _dst in pairs:
        if rel.startswith("../"):
            resolved = (INSTALLER / rel).resolve()
        else:
            resolved = (REPO_ROOT / rel).resolve()
        if not resolved.exists():
            missing.append(str(resolved))
    assert not missing, "spec references nonexistent paths:\n  " + "\n  ".join(missing)


# ---------------------------------------------------------------------------
# 3. hidden imports resolve
# ---------------------------------------------------------------------------


def _extract_hidden_imports(spec_text: str) -> list[str]:
    """Locate `hiddenimports=[...]` and return the strings inside."""
    m = re.search(r"hiddenimports\s*=\s*\[([^\]]*)\]", spec_text, re.DOTALL)
    if not m:
        return []
    body = m.group(1)
    return re.findall(r"['\"]([^'\"]+)['\"]", body)


def test_macos_spec_hidden_imports_exist_or_are_third_party():
    """Each hiddenimport must either be a real module on disk in
    `v2/` (our own code) or be a known third-party package (torch,
    numpy, etc.) which PyInstaller will pull from site-packages.

    We assert: every name starting with 'ai.' or 'core.' resolves to
    a .py under v2/. (Third-party names are not checked here -- pip
    + PyInstaller handle those at build time.)
    """
    v2 = REPO_ROOT / "v2"
    if str(v2) not in sys.path:
        sys.path.insert(0, str(v2))
    src = _load_spec_text("build_macos.spec")
    names = _extract_hidden_imports(src)
    # Filter to the project namespaces.
    project = [n for n in names if n.startswith(("ai.", "core."))]
    missing = []
    for n in project:
        path = v2 / Path(n.replace(".", "/")).with_suffix(".py")
        if not path.exists():
            missing.append(f"{n} -> {path}")
    assert not missing, "hidden imports point at missing files:\n  " + "\n  ".join(missing)


# ---------------------------------------------------------------------------
# 4. entitlements plist parses + declares the right capabilities
# ---------------------------------------------------------------------------


def test_entitlements_plist_parses():
    p = INSTALLER / "entitlements.plist"
    assert p.exists()
    with p.open("rb") as f:
        plist = plistlib.load(f)
    assert isinstance(plist, dict)


def test_entitlements_plist_has_required_capabilities():
    """The entitlements.plist must enable network + JIT (required for
    PyTorch) and must NOT enable App Sandbox (we ship Developer ID,
    not App Store)."""
    p = INSTALLER / "entitlements.plist"
    plist = plistlib.loads(p.read_bytes())
    # Required to be present + true.
    required_true = [
        "com.apple.security.network.client",
        "com.apple.security.network.server",
        "com.apple.security.cs.allow-jit",
        "com.apple.security.cs.allow-unsigned-executable-memory",
    ]
    for k in required_true:
        assert plist.get(k) is True, f"{k} must be present and True"
    # App Sandbox MUST NOT be enabled.
    assert plist.get("com.apple.security.app-sandbox") is not True, (
        "App Sandbox must NOT be enabled — we ship Developer ID outside "
        "the App Store and need filesystem access to the mesh state dir"
    )


# ---------------------------------------------------------------------------
# 5. build_macos.sh shell script has the env-var fallback structure
# ---------------------------------------------------------------------------


def test_build_macos_sh_present_and_executable_marker():
    sh = INSTALLER / "build_macos.sh"
    assert sh.exists(), "installer/build_macos.sh missing"
    # First line should be a shebang.
    first = sh.read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith("#!"), f"missing shebang: {first!r}"


def test_build_macos_sh_has_unsigned_fallback():
    """The §13 honest-fallback contract: when no Apple Dev ID is set,
    the script still produces an .app (ad-hoc) and warns clearly."""
    sh = (INSTALLER / "build_macos.sh").read_text(encoding="utf-8")
    assert 'PLUGINFER_APPLE_DEV_ID' in sh
    assert 'SKIPPED' in sh                           # the codesign-skipped branch
    assert 'UNSIGNED (ad-hoc)' in sh                 # the final-summary warning
    # The notarize step must also be opt-in.
    assert 'PLUGINFER_APPLE_KEYCHAIN_PROFILE' in sh


def test_build_macos_sh_supports_arch_argument():
    sh = (INSTALLER / "build_macos.sh").read_text(encoding="utf-8")
    # --arch flag handled in the loop.
    assert '--arch' in sh
    assert 'PLUGINFER_TARGET_ARCH' in sh


def test_build_macos_sh_refuses_non_macos_host():
    """Running on Linux/Windows must error out cleanly (uname check)."""
    sh = (INSTALLER / "build_macos.sh").read_text(encoding="utf-8")
    assert '"$(uname)" != "Darwin"' in sh
