# ============================================================================
#   PyInstaller spec for Pluginfer Windows .exe
#
#   Build with:
#       cd installer
#       pip install pyinstaller
#       pyinstaller build_windows.spec
#
#   Output:  installer/dist/Pluginfer/Pluginfer.exe + bundled deps
#
#   This produces a self-contained directory the user can zip + ship to
#   any Windows machine — no Python install required on the target.
# ============================================================================

# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['../v2/ai/filum/gui_launcher.py'],
    pathex=['../v2'],
    binaries=[],
    datas=[
        # Bundle the architecture + critical config alongside the exe.
        ('../v2/ai/filum/architecture.py', 'ai/filum'),
        ('../v2/ai/filum/auto_setup.py',   'ai/filum'),
        ('../v2/ai/filum/agent_mode.py',   'ai/filum'),
        ('../v2/ai/filum/self_context.py', 'ai/filum'),
        ('../v2/ai/filum/decision_engine.py', 'ai/filum'),
        ('../v2/ai/filum/lr_schedule.py',  'ai/filum'),
        ('../v2/ai/filum/optimizer_8bit.py', 'ai/filum'),
        ('../v2/ai/filum/tokenizer_bpe.py', 'ai/filum'),
        ('../v2/ai/filum/hpa', 'ai/filum/hpa'),
        # The whole repo as searchable corpus.
        ('../v2',          'v2'),
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
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',  # not needed
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
    upx=True,
    console=False,            # GUI app — no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='Pluginfer.ico',   # uncomment when icon ships
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Pluginfer',
)
