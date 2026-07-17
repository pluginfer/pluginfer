@echo off
echo [1/2] Setting up Dependencies (Downloading FFmpeg)...
python setup_dependencies.py

if %ERRORLEVEL% NEQ 0 (
    echo Error downloading dependencies. Exiting.
    exit /b %ERRORLEVEL%
)

echo.
echo [2/2] Building Zero-Config Executable...
pyinstaller examples/PluginferNode.spec --clean --noconfirm

if %ERRORLEVEL% NEQ 0 (
    echo Error during build.
    exit /b %ERRORLEVEL%
)

echo.
echo ==============================================
echo  BUILD SUCCESSFUL
echo  Binary: dist\PluginferNode.exe
echo ==============================================
pause
