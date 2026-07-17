@echo off
REM Quick port checker before launching Pluginfer
echo ================================================
echo   PLUGINFER PORT CHECKER
echo ================================================
echo.
echo Checking if ports are available...
echo.

set "PORTS_FREE=1"

REM Check port 8000
netstat -ano | findstr ":8000 " | findstr "LISTENING" >nul
if %errorlevel% equ 0 (
    echo [BUSY] Port 8000 is in use
    set "PORTS_FREE=0"
) else (
    echo [FREE] Port 8000
)

REM Check port 8001
netstat -ano | findstr ":8001 " | findstr "LISTENING" >nul
if %errorlevel% equ 0 (
    echo [BUSY] Port 8001 is in use
    set "PORTS_FREE=0"
) else (
    echo [FREE] Port 8001
)

REM Check port 9000
netstat -ano | findstr ":9000 " | findstr "LISTENING" >nul
if %errorlevel% equ 0 (
    echo [BUSY] Port 9000 is in use
    set "PORTS_FREE=0"
) else (
    echo [FREE] Port 9000
)

REM Check port 9001
netstat -ano | findstr ":9001 " | findstr "LISTENING" >nul
if %errorlevel% equ 0 (
    echo [BUSY] Port 9001 is in use
    set "PORTS_FREE=0"
) else (
    echo [FREE] Port 9001
)

echo.
if "%PORTS_FREE%"=="1" (
    echo [OK] All ports are available!
    echo You can safely launch Pluginfer.
) else (
    echo [WARNING] Some ports are in use!
    echo Run cleanup_ports.bat to free them before launching.
)
echo.
pause
