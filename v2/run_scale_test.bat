@echo off
echo [SETUP] Cleaning up environment...
powershell -Command "Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.Id -ne $PID} | Stop-Process -Force"
timeout /t 2 /nobreak >nul
if exist stress_*.log del /Q stress_*.log
echo [SETUP] Deleted old logs.
echo [TEST] Launching 20-Node Cluster...
python tests\stress_test_mesh.py
echo [TEST] Complete.
pause
