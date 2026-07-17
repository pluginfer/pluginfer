@echo off
REM ============================================================================
REM   Pluginfer -- double-click launcher (Windows).
REM   Runs the §H3 first-run orchestrator. On second-and-later opens the
REM   auto_setup step is a no-op so this is effectively just "open GUI."
REM ============================================================================
setlocal

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%" >nul
set "REPO_ROOT=%CD%"
popd >nul

set "PY_CMD="
for %%P in (python py python3) do (
    where %%P >nul 2>nul && set "PY_CMD=%%P" && goto :go
)
:go
if "%PY_CMD%"=="" (
    echo Python not found. Run Pluginfer-Setup.bat first.
    pause
    exit /b 1
)

cd /d "%REPO_ROOT%\v2"
start "" /b %PY_CMD% -m ai.filum.first_run

REM The GUI is windowed; this launcher can exit immediately.
exit /b 0
