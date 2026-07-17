@echo off
REM ============================================================================
REM   Pluginfer Setup -- one-file Windows installer.
REM   Double-click this file. Everything else is automatic.
REM   The GUI opens when setup is complete -- no second double-click needed.
REM ============================================================================
setlocal enabledelayedexpansion

echo.
echo  ===================================================================
echo                          PLUGINFER SETUP
echo                Earn from your idle GPU. Train AI for free.
echo  ===================================================================
echo.

REM 1. Find the repo root (the parent of this installer/ folder).
set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%" >nul
set "REPO_ROOT=%CD%"
popd >nul
echo  [1/6] Repo root        : %REPO_ROOT%

REM 2. Probe Python.
set "PY_CMD="
for %%P in (python py python3) do (
    where %%P >nul 2>nul && set "PY_CMD=%%P" && goto :py_found
)
:py_found
if "%PY_CMD%"=="" (
    echo  [ERROR] Python not found.
    echo          Install Python 3.10+ from https://python.org and re-run.
    echo          Direct download: https://www.python.org/downloads/
    start "" "https://www.python.org/downloads/"
    pause
    exit /b 1
)
echo  [2/6] Python           : %PY_CMD%

REM 3. Detect accelerator vendor BEFORE installing torch so we pull
REM    the right wheel. NVIDIA -> CUDA wheel; everyone else -> CPU/MPS
REM    (ROCm-on-Windows is not officially supported by upstream torch
REM    as of this writing, so AMD users get the CPU wheel and can swap
REM    in their preferred build later).
set "VENDOR=cpu"
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    set "VENDOR=cuda"
)

echo  [3/6] Accelerator probe: %VENDOR%

REM 4. Install dependencies. One-time; pip is idempotent so re-runs are cheap.
echo         Installing dependencies (one-time step)...
%PY_CMD% -m pip install --quiet --upgrade pip 2>nul
%PY_CMD% -m pip install --quiet psutil numpy 2>nul

%PY_CMD% -c "import torch" >nul 2>nul
if errorlevel 1 (
    if "%VENDOR%"=="cuda" (
        echo         Installing PyTorch with CUDA 12.1 support...
        %PY_CMD% -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
        if errorlevel 1 (
            echo         CUDA wheel install failed; falling back to CPU torch...
            %PY_CMD% -m pip install --quiet torch
        )
    ) else (
        echo         Installing PyTorch (CPU build)...
        %PY_CMD% -m pip install --quiet torch
    )
)
echo         dependencies ready.

REM 5. Run §H3 first_run -- handles auto_setup (idempotent) AND launches
REM    the GUI in a single step. The user does NOT need to double-click
REM    anything else after this.
echo  [4/6] First-run orchestrator: configure + launch GUI...
cd /d "%REPO_ROOT%\v2"

REM Optional: create a Start Menu shortcut to the launcher for repeat opens.
echo  [5/6] Creating Start Menu shortcut...
set "SHORTCUT_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Pluginfer"
if not exist "%SHORTCUT_DIR%" mkdir "%SHORTCUT_DIR%"
set "LAUNCHER=%REPO_ROOT%\installer\Pluginfer.bat"
%PY_CMD% -c "from win32com.client import Dispatch; sh=Dispatch('WScript.Shell'); s=sh.CreateShortcut(r'%SHORTCUT_DIR%\Pluginfer.lnk'); s.Targetpath=r'%LAUNCHER%'; s.WorkingDirectory=r'%REPO_ROOT%\v2'; s.IconLocation=r'%LAUNCHER%'; s.save()" >nul 2>nul
if errorlevel 1 (
    echo         (pywin32 not installed; Start Menu shortcut skipped.
    echo          You can re-launch Pluginfer via: %LAUNCHER%)
) else (
    echo         shortcut: %SHORTCUT_DIR%\Pluginfer.lnk
)

REM 6. Hand off to the GUI. `start /b` so this .bat exits and the
REM    GUI window owns the foreground.
echo  [6/6] Opening Pluginfer GUI...
echo.
start "" /b %PY_CMD% -m ai.filum.first_run
exit /b 0
