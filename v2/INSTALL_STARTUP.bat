
@echo off
title Pluginfer - Install Startup
color 0B
echo "Installing Pluginfer to Windows Startup..."

set "SCRIPT_PATH=%~dp0run_immortal.bat"
set "STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT_PATH=%STARTUP_FOLDER%\PluginferNode.lnk"

echo "Target Script: %SCRIPT_PATH%"
echo "Startup Folder: %STARTUP_FOLDER%"

:: Create Shortcut using PowerShell
powershell "$s=(New-Object -COM WScript.Shell).CreateShortcut('%SHORTCUT_PATH%');$s.TargetPath='%SCRIPT_PATH%';$s.WorkingDirectory='%~dp0';$s.Save()"

if exist "%SHORTCUT_PATH%" (
    echo "✅ SUCCESS: Pluginfer will now start automatically when you turn on your PC."
) else (
    echo "❌ ERROR: Failed to create shortcut."
)

pause
