#!/bin/bash
# ============================================
# Pluginfer Linux/Mac Executable Builder
# ============================================

echo "============================================"
echo "Pluginfer - Building Executable"
echo "============================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ first"
    exit 1
fi

echo "Step 1: Installing PyInstaller..."
echo
pip3 install pyinstaller --break-system-packages || pip3 install pyinstaller --user
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install PyInstaller"
    exit 1
fi

echo
echo "Step 2: Installing project dependencies..."
echo
pip3 install torch numpy py-cpuinfo cryptography pyjwt requests psutil --break-system-packages 2>/dev/null || \
pip3 install torch numpy py-cpuinfo cryptography pyjwt requests psutil --user 2>/dev/null || \
echo "WARNING: Some dependencies may be missing"

echo
echo "Step 3: Building executable..."
echo

# Build main application
pyinstaller --noconfirm \
    --onefile \
    --name="pluginfer" \
    --add-data="plugins:plugins" \
    --add-data="core:core" \
    --add-data="README.md:." \
    --hidden-import=core \
    --hidden-import=core.plugin_base \
    --hidden-import=core.plugin_registry \
    --hidden-import=core.inference_engine \
    --hidden-import=core.hardware_detector \
    --hidden-import=core.qal_controller \
    --hidden-import=core.license_validator \
    --hidden-import=core.mesh_controller \
    pluginfer.py

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed"
    exit 1
fi

echo
echo "============================================"
echo "Build Complete!"
echo "============================================"
echo
echo "Executable created: dist/pluginfer"
echo
echo "To run:"
echo "  1. Copy dist/pluginfer to your desired location"
echo "  2. Make sure plugins folder is in the same directory"
echo "  3. Run: ./pluginfer --test"
echo
echo "Note: The executable includes Python interpreter and dependencies"
echo "      Size: ~50-100 MB"
echo

# Create a standalone package
echo "Creating standalone package..."
mkdir -p pluginfer_standalone
cp dist/pluginfer pluginfer_standalone/
cp -r plugins pluginfer_standalone/
cp README.md pluginfer_standalone/
cp QUICK_START.md pluginfer_standalone/ 2>/dev/null
cp MESH_NETWORKING.md pluginfer_standalone/ 2>/dev/null
chmod +x pluginfer_standalone/pluginfer

echo
echo "Standalone package created: pluginfer_standalone/"
echo "This folder contains everything needed to run Pluginfer"
echo
echo "To use:"
echo "  cd pluginfer_standalone"
echo "  ./pluginfer --test"
echo
