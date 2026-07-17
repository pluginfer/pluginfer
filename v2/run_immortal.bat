
@echo off
title Pluginfer V2 - IMMORTAL NODE
color 0A

:START
cls
echo "=================================================="
echo "      PLUGINFER V2: AUTONOMOUS AI NODE            "
echo "      STATUS: IMMORTAL (AUTO-RESTART ON)          "
echo "=================================================="
echo "System Time: %time%"
echo "Starting Node Controller..."

:: Run the Python Controller
:: using pythonw to hide console window if preferred, but here we want visibility
:: Fix Unicode Logging Error
set PYTHONUTF8=1

:: Try to run with Python 3.11 explicitly (if available via py launcher)
py -3.11 pluginfer_node.py
if %ERRORLEVEL% NEQ 0 (
    echo "Python 3.11 not found via py launcher. Trying default python..."
    python pluginfer_node.py
)

:: If it crashes, wait 5 seconds and restart
echo "!!! CRITICAL CRASH DETECTED !!!"
echo "Re-spawning in 5 seconds..."
pause
goto START
