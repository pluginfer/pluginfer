
@echo off
title Pluginfer V2 - BUILD GAMING INSTALLER
color 0B
echo "=========================================="
echo "   BUILDING GAMING STYLE INSTALLER (.EXE) "
echo "=========================================="

echo "Checking PyInstaller..."
pip install pyinstaller
echo "Installing Dependencies..."
pip install -r requirements.txt

echo "Building Setup.exe..."
pyinstaller --noconfirm --onefile --windowed --name "Setup_v6_Perfect" --distpath "Pluginfer_v2_Secure_Installer" --icon "NONE" installer_gui.py

echo "=========================================="
echo "   BUILD COMPLETE "
echo "   LOCATION: Pluginfer_v2_Secure_Installer/Setup_v6_Perfect.exe"
echo "=========================================="
pause
