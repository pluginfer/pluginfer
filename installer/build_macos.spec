# ============================================================================
#   PyInstaller spec for Pluginfer macOS .app (Apple Silicon + Intel)
#
#   Build with:
#       cd installer
#       pip install pyinstaller
#       pyinstaller --clean build_macos.spec
#
#   Output:
#       installer/dist/Pluginfer/Pluginfer.app   - the bundle
#       installer/dist/Pluginfer/                - the COLLECT directory
#
#   Cross-arch notes:
#     * On an Apple Silicon dev box, set PLUGINFER_TARGET_ARCH=universal2
#       in the environment to produce a fat binary that runs on both
#       arm64 (M1/M2/M3/M4) and x86_64 (Intel). Requires Python ≥3.11
#       built with universal2 wheels (see https://www.python.org).
#     * On Intel macOS, set PLUGINFER_TARGET_ARCH=x86_64 to produce a
#       single-arch x86_64 build. Apple Silicon Macs run x86_64 binaries
#       under Rosetta 2 — works but slower than native arm64.
#     * Default (no env var): single-arch matching the build host. The
#       wrapper script `build_macos.sh` sets the env var explicitly so
#       this default is rarely hit.
#
#   Signing notes:
#     This spec leaves codesign_identity=None. The shell wrapper
#     (`build_macos.sh`) post-processes the .app with `codesign --deep`
#     and `xcrun notarytool` after PyInstaller exits. Doing it outside
#     the spec lets us reuse one .app for ad-hoc test signing on a dev
#     machine and for proper notarized signing on a CI signer.
# ============================================================================

# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None
target_arch = os.environ.get("PLUGINFER_TARGET_ARCH") or None    # None = host
entitlements = os.environ.get("PLUGINFER_ENTITLEMENTS") or None  # None = unset

a = Analysis(
    ['../v2/ai/filum/gui_launcher.py'],
    pathex=['../v2'],
    binaries=[],
    datas=[
        # Bundle the architecture + critical configs alongside the .app.
        ('../v2/ai/filum/architecture.py',   'ai/filum'),
        ('../v2/ai/filum/auto_setup.py',     'ai/filum'),
        ('../v2/ai/filum/agent_mode.py',     'ai/filum'),
        ('../v2/ai/filum/self_context.py',   'ai/filum'),
        ('../v2/ai/filum/decision_engine.py','ai/filum'),
        ('../v2/ai/filum/lr_schedule.py',    'ai/filum'),
        ('../v2/ai/filum/optimizer_8bit.py', 'ai/filum'),
        ('../v2/ai/filum/tokenizer_bpe.py',  'ai/filum'),
        ('../v2/ai/filum/hpa',               'ai/filum/hpa'),
        # The whole repo as searchable corpus + provenance docs.
        ('../v2',            'v2'),
        ('../README.md',     '.'),
    ],
    hiddenimports=[
        'tkinter',
        'numpy',
        'torch',
        'psutil',
        'ai.filum.gui_launcher',
        'ai.filum.auto_setup',
        'ai.filum.service_mode',
        'ai.filum.agent_mode',
        'ai.filum.first_run',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'pytest',
        'IPython',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Pluginfer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                   # codesign + UPX don't mix on macOS
    console=False,               # GUI app — no terminal window
    disable_windowed_traceback=False,
    argv_emulation=True,         # macOS open-document/url events
    target_arch=target_arch,     # universal2 / arm64 / x86_64 / None
    codesign_identity=None,      # done out-of-band by build_macos.sh
    entitlements_file=entitlements,
    # icon='Pluginfer.icns',     # uncomment when icon ships
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Pluginfer',
)

app = BUNDLE(
    coll,
    name='Pluginfer.app',
    # icon='Pluginfer.icns',     # uncomment when icon ships
    bundle_identifier='com.pluginfer.gui',
    info_plist={
        'CFBundleName':                 'Pluginfer',
        'CFBundleDisplayName':          'Pluginfer',
        'CFBundleShortVersionString':   '0.1.0',
        'CFBundleVersion':              '1',
        'NSHighResolutionCapable':      True,
        'LSMinimumSystemVersion':       '11.0',     # Big Sur
        'NSHumanReadableCopyright':     'Copyright (c) 2026 Pluginfer.',
        # Background-friendly: don't hide other windows.
        'LSUIElement':                  False,
        # Network entitlements live in the entitlements .plist; here we
        # only declare the user-facing capability surface.
        'NSCameraUsageDescription':     'Pluginfer does not use the camera.',
        'NSMicrophoneUsageDescription': 'Pluginfer does not use the mic.',
    },
)
