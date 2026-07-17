@echo off
REM ============================================
REM Pluginfer Windows Executable Builder
REM ============================================

echo ============================================
echo Pluginfer - Building Windows Executable
echo ============================================
echo.

echo Step 1: Dependencies assumed installed via pip...
echo.

echo.
echo Step 3: Building executable (FOLDER MODE for INSTANT START)...
echo.

REM Clean previous build artifacts
if exist build rmdir /S /Q build
if exist dist rmdir /S /Q dist
REM Build main application
echo Building PluginferNode...
python -m PyInstaller --noconfirm --clean PluginferNode.spec

if errorlevel 1 (
    echo ERROR: Build failed
    exit /b 1
)

echo.
echo Step 4: Building Installer UI (Setup.exe)...
echo.
python -m PyInstaller --noconfirm ^
    --onefile ^
    --windowed ^
    --name="Setup" ^
    --icon=NONE ^
    installer_gui.py

if errorlevel 1 (
    echo ERROR: Installer Build failed
    exit /b 1
)

echo.
echo ============================================
echo Build Complete!
echo ============================================
echo.

REM Prepare the Distribution Folder
echo Organizing release files...

REM Define Release Folder
set RELEASE_DIR=Pluginfer_v2_Secure_Installer

REM Clean previous release
rmdir /S /Q %RELEASE_DIR% 2>nul
mkdir %RELEASE_DIR%

REM Copy the Built Application Folder
REM We rename it to 'PluginferNode' inside the release so the installer finds it
xcopy /E /I /Y dist\PluginferNode %RELEASE_DIR%\PluginferNode

REM Copy the Installer
copy dist\Setup.exe %RELEASE_DIR%\

REM Copy Documentation
copy README.md %RELEASE_DIR%\

echo.
echo ========================================================
echo  SECURE RELEASE READY: %RELEASE_DIR%
echo ========================================================
echo.
echo  INSTRUCTIONS:
echo  1. Open the '%RELEASE_DIR%' folder.
echo  2. Double-click 'Setup.exe' (The Game-Style Installer).
echo  3. It will simulate installation and launch the Node.
echo  4. The actual code is inside the 'PluginferNode' folder.
echo.
echo ========================================================

