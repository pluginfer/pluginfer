#!/usr/bin/env bash
# ============================================================================
#   build_macos.sh — produce a signed, notarized Pluginfer.app + .pkg.
#
#   Usage:
#       ./build_macos.sh                              # default: host arch
#       ./build_macos.sh --arch universal2            # Apple Silicon + Intel
#       ./build_macos.sh --arch x86_64                # Intel-only
#       ./build_macos.sh --arch arm64                 # Apple Silicon-only
#
#   Signing + notarization (optional, enabled when env vars are set):
#       PLUGINFER_APPLE_DEV_ID="Developer ID Application: NAME (TEAMID)"
#       PLUGINFER_APPLE_INSTALLER_ID="Developer ID Installer: NAME (TEAMID)"
#       PLUGINFER_APPLE_KEYCHAIN_PROFILE=AC_PROFILE_NAME    # for notarytool
#
#   When those env vars are absent, the script:
#       * still produces a working .app (unsigned ad-hoc),
#       * skips the codesign + notarytool steps,
#       * prints a clear "ad-hoc only" warning at the end so the
#         developer knows the artefact won't pass Gatekeeper for
#         distribution outside the local machine.
#
#   This is the "honest unsigned fallback" referenced in §13 audit.
# ============================================================================
set -e
set -o pipefail

ARCH="${PLUGINFER_TARGET_ARCH:-}"
SIGN_ID="${PLUGINFER_APPLE_DEV_ID:-}"
INSTALLER_SIGN_ID="${PLUGINFER_APPLE_INSTALLER_ID:-}"
NOTARIZE_PROFILE="${PLUGINFER_APPLE_KEYCHAIN_PROFILE:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        --arch)             ARCH="$2"; shift 2 ;;
        --sign-id)          SIGN_ID="$2"; shift 2 ;;
        --installer-id)     INSTALLER_SIGN_ID="$2"; shift 2 ;;
        --notarize-profile) NOTARIZE_PROFILE="$2"; shift 2 ;;
        --help|-h)
            grep '^#' "$0" | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "unknown arg: $1"
            exit 2
            ;;
    esac
done

# ---------- 0. Sanity check ----------------------------------------------
if [ "$(uname)" != "Darwin" ]; then
    echo "ERROR: build_macos.sh runs on macOS only. (uname=$(uname))"
    exit 1
fi

INSTALLER_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$INSTALLER_DIR/.." && pwd)"

# ---------- 1. Python + PyInstaller --------------------------------------
PY=$(command -v python3)
if [ -z "$PY" ]; then
    echo "ERROR: python3 not found on PATH."
    exit 1
fi

echo "  arch         : ${ARCH:-host-default}"
echo "  python       : $PY ($($PY --version))"
echo "  installer    : $INSTALLER_DIR"
echo "  repo root    : $REPO_ROOT"

# Install build deps quietly. Ad-hoc dev boxes may already have these.
$PY -m pip install --quiet --upgrade pip
$PY -m pip install --quiet pyinstaller

# ---------- 2. PyInstaller -----------------------------------------------
cd "$INSTALLER_DIR"
export PLUGINFER_TARGET_ARCH="$ARCH"
export PLUGINFER_ENTITLEMENTS="$INSTALLER_DIR/entitlements.plist"
echo "  building Pluginfer.app via PyInstaller (arch=${ARCH:-host})..."
$PY -m PyInstaller --noconfirm --clean build_macos.spec

APP="$INSTALLER_DIR/dist/Pluginfer/Pluginfer.app"
if [ ! -d "$APP" ]; then
    echo "ERROR: PyInstaller did not produce $APP"
    exit 1
fi

echo "  built        : $APP"

# ---------- 3. Codesign --------------------------------------------------
if [ -n "$SIGN_ID" ]; then
    echo "  codesigning  : $SIGN_ID"
    # --deep walks every nested binary; --force re-signs any bundled
    # frameworks; --options=runtime enables the hardened runtime
    # required for notarization on macOS 10.15+.
    codesign --force --deep --options=runtime \
        --entitlements "$INSTALLER_DIR/entitlements.plist" \
        --sign "$SIGN_ID" \
        "$APP"

    # Verify.
    codesign --verify --deep --strict --verbose=2 "$APP" 2>&1 | tail -5
    spctl --assess --type execute --verbose "$APP" 2>&1 || \
        echo "  (Gatekeeper assessment not yet OK — notarization pending)"
else
    echo "  codesigning  : SKIPPED (no PLUGINFER_APPLE_DEV_ID set)"
    # Best-effort ad-hoc sign so the .app at least runs locally on
    # the developer's machine without 'damaged app' Gatekeeper warnings.
    codesign --force --deep --sign - "$APP" 2>/dev/null || true
fi

# ---------- 4. Notarize --------------------------------------------------
if [ -n "$SIGN_ID" ] && [ -n "$NOTARIZE_PROFILE" ]; then
    ZIP="$INSTALLER_DIR/dist/Pluginfer.zip"
    /usr/bin/ditto -c -k --keepParent "$APP" "$ZIP"
    echo "  notarizing   : $ZIP via profile $NOTARIZE_PROFILE"
    if xcrun notarytool submit "$ZIP" \
        --keychain-profile "$NOTARIZE_PROFILE" \
        --wait \
        --output-format json > "$INSTALLER_DIR/dist/notary.json"; then
        STATUS=$(/usr/bin/python3 -c \
            "import json; print(json.load(open('$INSTALLER_DIR/dist/notary.json')).get('status', '?'))")
        echo "  notary status: $STATUS"
        if [ "$STATUS" = "Accepted" ]; then
            xcrun stapler staple "$APP"
            spctl --assess --type execute --verbose "$APP" 2>&1 | tail -3
        else
            echo "  notarization FAILED — fetch log:"
            xcrun notarytool log \
                "$(/usr/bin/python3 -c "import json; print(json.load(open('$INSTALLER_DIR/dist/notary.json'))['id'])")" \
                --keychain-profile "$NOTARIZE_PROFILE" 2>&1 | tail -40 || true
        fi
    else
        echo "  notarytool failed — see $INSTALLER_DIR/dist/notary.json"
    fi
else
    echo "  notarize     : SKIPPED (PLUGINFER_APPLE_KEYCHAIN_PROFILE unset)"
fi

# ---------- 5. .pkg installer (optional) ---------------------------------
if [ -n "$INSTALLER_SIGN_ID" ]; then
    PKG_OUT="$INSTALLER_DIR/dist/Pluginfer.pkg"
    echo "  building pkg : $PKG_OUT"
    pkgbuild --root "$INSTALLER_DIR/dist/Pluginfer" \
        --identifier com.pluginfer.gui \
        --version 0.1.0 \
        --install-location /Applications \
        --sign "$INSTALLER_SIGN_ID" \
        "$PKG_OUT.unsigned" 2>&1 | tail -5

    productbuild --package "$PKG_OUT.unsigned" \
        --sign "$INSTALLER_SIGN_ID" \
        "$PKG_OUT" 2>&1 | tail -5
    rm -f "$PKG_OUT.unsigned"

    if [ -n "$NOTARIZE_PROFILE" ]; then
        echo "  notarizing pkg ..."
        xcrun notarytool submit "$PKG_OUT" \
            --keychain-profile "$NOTARIZE_PROFILE" --wait || true
        xcrun stapler staple "$PKG_OUT" || true
    fi
    echo "  pkg done     : $PKG_OUT"
else
    echo "  pkg          : SKIPPED (no PLUGINFER_APPLE_INSTALLER_ID set)"
fi

# ---------- 6. Summary ----------------------------------------------------
echo
echo "  ==================================================================="
if [ -n "$SIGN_ID" ]; then
    echo "    Signed Pluginfer.app at: $APP"
else
    echo "    UNSIGNED (ad-hoc) Pluginfer.app at: $APP"
    echo "    For distribution, set PLUGINFER_APPLE_DEV_ID + run again."
    echo "    (Apple Developer Program: \$99/yr; see docs/SIGNING_SETUP.md)"
fi
echo "  ==================================================================="
echo
