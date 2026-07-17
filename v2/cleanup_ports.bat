@echo off
REM Cleanup script to kill all Pluginfer processes and free ports
echo ================================================
echo   PLUGINFER PORT CLEANUP UTILITY
echo ================================================
echo.
echo Killing all Python processes...
taskkill /F /IM python.exe /T 2>nul
if %errorlevel% equ 0 (
    echo [OK] Python processes terminated
) else (
    echo [INFO] No Python processes found
)

echo.
echo Killing all PluginferNode processes...
taskkill /F /IM PluginferNode.exe /T 2>nul
if %errorlevel% equ 0 (
    echo [OK] PluginferNode processes terminated
) else (
    echo [INFO] No PluginferNode processes found
)

echo.
echo Waiting for ports to be released...
timeout /t 2 /nobreak >nul

echo.
echo ================================================
echo   PORT STATUS CHECK
echo ================================================
netstat -ano | findstr ":8000 :8001 :9000 :9001" >nul
if %errorlevel% equ 0 (
    echo [WARNING] Some ports still in use:
    netstat -ano | findstr ":8000 :8001 :9000 :9001"
) else (
    echo [OK] All ports are FREE
)

echo.
echo Cleanup complete!
pause
