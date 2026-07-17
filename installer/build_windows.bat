@echo off
REM Build Pluginfer.exe using PyInstaller. Run once on a Windows dev box.
REM Output: installer/dist/Pluginfer/ — zip and ship.
setlocal

set "INSTALLER_DIR=%~dp0"
cd /d "%INSTALLER_DIR%"

echo Installing PyInstaller...
pip install --quiet pyinstaller

echo Building Pluginfer.exe ...
pyinstaller --noconfirm --clean build_windows.spec

if errorlevel 1 (
    echo Build failed. See above output.
    pause
    exit /b 1
)

echo.
echo Build complete: %INSTALLER_DIR%dist\Pluginfer\
echo Zip that folder and ship to any Windows user.
echo No Python install required on the target machine.
echo.
pause
