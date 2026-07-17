@echo off
echo ===================================================
echo   PLUGINFER - CLEAN REPAIR & REBUILD
echo ===================================================
echo.

echo 1. Stopping any running instances...
taskkill /F /IM PluginferNode.exe 2>nul
taskkill /F /IM Setup.exe 2>nul

echo.
echo 2. Wiping old build artifacts...
if exist build rmdir /S /Q build
if exist dist rmdir /S /Q dist
if exist Pluginfer_v2_Secure_Installer rmdir /S /Q Pluginfer_v2_Secure_Installer

echo.
echo 3. Starting Fresh Build...
call build_windows.bat
