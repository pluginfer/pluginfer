@echo off
echo [TEST] Launching 20-Node Cluster (No Cleanup)...
if exist stress_*.log del /Q stress_*.log
python tests\stress_test_mesh.py
echo [TEST] Complete.
pause
