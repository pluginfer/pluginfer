# Pluginfer signing setup

Pluginfer ships signed installers on three platforms. Each platform has
its own signing chain. **None of them are free**; budget a few hundred
dollars per year and a couple of weeks of paperwork before launch.

This doc walks through what to obtain, where to plug it in, and how the
build pipeline (`v2/build/`) consumes each secret.

> ⚠ **Never commit any private key, .pfx, .p12, or password to the
> repo.** Every secret listed here belongs in a CI secret store (GitHub
> Actions encrypted secrets, your password manager, your HSM). The
> `.gitignore` already excludes `*.pem` / `*.pfx` / `*.p12` patterns.

---

## 1. Pluginfer release key (manifest signing)

The auto-updater (`core/updater.py`, hardened in W31) verifies the
release manifest against `PLUGINFER_RELEASE_PUBKEY_PEM` baked into the
binary. The build pipeline signs the manifest with the matching
private key. **You generate this key yourself; it never leaves your
trusted environment.**

### Generate

```sh
# One-time, on an offline / hardened machine:
openssl ecparam -name secp256k1 -genkey -noout -out release.priv.pem
openssl ec -in release.priv.pem -pubout -out release.pub.pem

# Bake the public key into the binary by exporting it as an env var
# during the build:
export PLUGINFER_RELEASE_PUBKEY_PEM="$(cat release.pub.pem)"

# Store the private key in your CI secret store (GitHub: Settings ->
# Secrets -> Actions). Never on a developer laptop.
```

### Used by

| Side    | Code path                            | Env var                              |
| ------- | ------------------------------------ | ------------------------------------ |
| Build   | `v2/build/manifest.py:sign_manifest` | `PLUGINFER_RELEASE_PRIVKEY_PEM`      |
| Runtime | `core/updater.py:_verify_manifest`   | `PLUGINFER_RELEASE_PUBKEY_PEM`       |

The build/release pipeline is in `v2/build/manifest.py`. Run it after
all artefacts are produced; it walks the dist/ directory, computes
SHA-256s, and emits a signed `manifest.json`.

---

## 2. Windows Authenticode (`.exe` code signing)

Windows Defender / SmartScreen will quarantine an unsigned installer
the first time tens of users download it. Get a code-signing
certificate.

### EV vs OV

| Type | Cost (yr) | SmartScreen reputation | Hardware token |
| ---- | --------- | ---------------------- | -------------- |
| EV   | ~$300-500 | Immediate trust        | Required       |
| OV   | ~$100-200 | Builds over weeks      | None           |

EV is recommended for launch (we want zero install friction).
DigiCert / Sectigo / GlobalSign all sell them.

### Acquire

1. Buy from a CA. They will mail you a hardware token (USB) or a
   download link with a `.pfx`.
2. The CA performs identity verification against your business or
   personal name.
3. You receive a `.pfx` file and a password.

### Configure CI

```yaml
# .github/workflows/release.yml (excerpt)
env:
  PLUGINFER_AUTHENTICODE_PFX: ${{ secrets.PLUGINFER_AUTHENTICODE_PFX_PATH }}
  PLUGINFER_AUTHENTICODE_PASS: ${{ secrets.PLUGINFER_AUTHENTICODE_PASS }}
```

### Used by

`v2/build/windows/build_windows.py:_maybe_sign` -> `signtool.exe sign`
with SHA-256 + RFC 3161 timestamping. Without these env vars the
build still completes (UNSIGNED .exe), but SmartScreen will block.

### Submit to Microsoft Defender

After each release, also submit the signed installer to:
<https://www.microsoft.com/en-us/wdsi/filesubmission>

---

## 3. macOS notarization (.app + .pkg)

Apple Notary Service checks every binary distributed outside the App
Store. Without it, Gatekeeper blocks even Cmd+Click bypass on some
recent macOS versions.

### Prerequisites

1. **Apple Developer Program** membership ($99/year). Sign up at
   <https://developer.apple.com>. Identity verification takes 1-2
   weeks.
2. **Developer ID Application** certificate. Created via Xcode or
   `developer.apple.com/account/resources/certificates`. Installs
   into your login keychain.
3. **App-specific password** for notarization. Generated at
   <https://appleid.apple.com> -> Sign In and Security -> App-Specific
   Passwords.

### Store credentials

```sh
# One-time, on the build mac:
xcrun notarytool store-credentials pluginfer-notary \
    --apple-id you@example.com \
    --team-id ABCD123456 \
    --password <app-specific password>

# This writes to the keychain. The build script just references the
# profile name 'pluginfer-notary'.
```

### Configure CI

```yaml
env:
  APPLE_DEVELOPER_TEAM_ID: ABCD123456
  APPLE_NOTARY_PROFILE: pluginfer-notary
```

### Used by

- `installer/build_macos.sh` — the canonical build script. Reads
  these env vars and runs `codesign --sign` with
  `installer/entitlements.plist` (hardened runtime), then submits to
  `xcrun notarytool` and staples on success.
- `installer/build_macos.spec` — the PyInstaller spec. Honours
  `PLUGINFER_TARGET_ARCH=universal2|arm64|x86_64` and
  `PLUGINFER_ENTITLEMENTS=path/to/entitlements.plist`.
- `installer/entitlements.plist` — the hardened-runtime
  entitlements (network client+server, JIT, dyld env vars). App Sandbox
  is intentionally NOT enabled because we ship via Developer ID
  outside the App Store and need filesystem access to the mesh state
  directory.

End-to-end command:

```sh
export PLUGINFER_APPLE_DEV_ID="Developer ID Application: NAME (TEAMID)"
export PLUGINFER_APPLE_INSTALLER_ID="Developer ID Installer: NAME (TEAMID)"
export PLUGINFER_APPLE_KEYCHAIN_PROFILE="pluginfer-notary"
./installer/build_macos.sh --arch universal2
# Output: installer/dist/Pluginfer/Pluginfer.app (signed + notarized)
#         installer/dist/Pluginfer.pkg (signed + notarized installer)
```

When the env vars are absent the script still produces a working
unsigned ad-hoc .app that runs on the developer's local machine and
prints a clear "ad-hoc only — won't pass Gatekeeper for distribution"
warning. This is the §13 honest-fallback contract: developers without
an Apple Developer ID can still build + smoke-test locally without
the script silently failing.

### Verify locally

```sh
spctl --assess --verbose=4 dist/Pluginfer.app
# Should print: accepted, source=Notarized Developer ID
```

---

## 4. Build host requirements

| Target | Host         | Required tools                                |
| ------ | ------------ | --------------------------------------------- |
| Linux  | any (CI ok)  | `python3`, `pyinstaller`, optionally `dpkg-deb` |
| Windows| Windows host | `python3`, `pyinstaller`, NSIS (`makensis`), Windows SDK (`signtool.exe`) |
| macOS  | macOS host   | `python3`, `pyinstaller`, Xcode CLT, `productbuild`, `codesign`, `xcrun`  |

`v2/build/build_all.py --platform all` skips targets whose host doesn't
match.

---

## 5. Smoke test the chain end-to-end

```sh
# In a clean checkout, with all env vars set:
python -m build.build_all --platform host
python -m build.manifest \
    --version 1.0.0 --git-sha "$(git rev-parse --short HEAD)" \
    --linux-deb v2/build/dist/pluginfer_1.0.0_amd64.deb \
    --output v2/build/dist/manifest.json

# Verify the signed manifest with the pubkey baked into the runtime:
python -c "
from build.manifest import verify_manifest
import json, os
m = json.loads(open('v2/build/dist/manifest.json').read())
pub = os.environ['PLUGINFER_RELEASE_PUBKEY_PEM']
print('verify =', verify_manifest(m, pubkey_pem=pub))
"
```

If `verify = True`, the auto-updater will accept this manifest at
runtime. If `False`, double-check that the build host's `_PRIVKEY_PEM`
is the matching pair of the runtime's `_PUBKEY_PEM`.
