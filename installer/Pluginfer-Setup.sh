#!/bin/bash
# ============================================================================
#   Pluginfer Setup -- one-file macOS / Linux installer.
#   Run once. Detects vendor (CUDA / ROCm / MPS / CPU), installs the right
#   torch wheel, configures the node, and OPENS the GUI. No second step.
# ============================================================================
set -e

echo
echo "  ==================================================================="
echo "                          PLUGINFER SETUP"
echo "                Earn from your idle GPU. Train AI for free."
echo "  ==================================================================="
echo

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."
OS="$(uname)"
ARCH="$(uname -m)"

echo "  [1/6] Repo root  : $REPO_ROOT"
echo "         OS / arch : $OS $ARCH"

# 1. Probe Python 3.
if ! command -v python3 >/dev/null 2>&1; then
    echo "  [ERROR] Python 3 not found."
    if [ "$OS" = "Darwin" ]; then
        echo "          On macOS:  brew install python3"
        echo "          Or download from https://www.python.org/downloads/"
    else
        echo "          On Linux:  sudo apt install python3 python3-pip"
    fi
    exit 1
fi
PY=$(command -v python3)
echo "  [2/6] Python     : $PY"

# 2. Detect accelerator vendor so we install the correct torch wheel.
#    Order: NVIDIA -> AMD/ROCm -> Apple Silicon -> Intel iGPU -> CPU.
VENDOR="cpu"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    VENDOR="cuda"
elif [ "$OS" = "Linux" ] && (command -v rocm-smi >/dev/null 2>&1 \
       || lspci 2>/dev/null | grep -qi 'amd.*radeon\|amd/ati\|advanced micro devices.*display'); then
    VENDOR="rocm"
elif [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    VENDOR="mps"
elif [ "$OS" = "Linux" ] && lspci 2>/dev/null | grep -qi 'intel.*arc\|intel corporation.*display'; then
    VENDOR="xpu"
fi
echo "  [3/6] Accelerator: $VENDOR"

# 3. Install deps.
echo "  [4/6] Installing dependencies..."
$PY -m pip install --quiet --upgrade pip 2>/dev/null || true
$PY -m pip install --quiet psutil numpy 2>/dev/null || true

if ! $PY -c "import torch" >/dev/null 2>&1; then
    case "$VENDOR" in
        cuda)
            echo "         Installing PyTorch with CUDA 12.1 support..."
            $PY -m pip install --quiet torch \
                --index-url https://download.pytorch.org/whl/cu121 2>/dev/null \
                || $PY -m pip install --quiet torch
            ;;
        rocm)
            echo "         Installing PyTorch with ROCm support..."
            $PY -m pip install --quiet torch \
                --index-url https://download.pytorch.org/whl/rocm6.0 2>/dev/null \
                || $PY -m pip install --quiet torch
            ;;
        mps|xpu|cpu)
            # Stock pip wheel covers Apple Silicon (MPS auto-detected), Intel
            # XPU experimental, and CPU baseline.
            echo "         Installing PyTorch (covers $VENDOR backend)..."
            $PY -m pip install --quiet torch
            ;;
    esac
fi
echo "         dependencies ready."

# 4. Make the relaunch helpers executable.
chmod +x "$SCRIPT_DIR/Pluginfer.command" 2>/dev/null || true
chmod +x "$SCRIPT_DIR/Pluginfer-Setup.sh" 2>/dev/null || true

# 5. Register a desktop entry on Linux for repeat launches.
if [ "$OS" = "Linux" ] && [ -d "$HOME/.local/share/applications" ]; then
    # Rewrite the .desktop with an absolute Exec path so it works
    # regardless of working directory at launch time.
    cat > "$HOME/.local/share/applications/Pluginfer.desktop" <<DESKEOF
[Desktop Entry]
Name=Pluginfer
Comment=Earn from your idle GPU. Train AI for free.
Exec=$PY -m ai.filum.first_run
Path=$REPO_ROOT/v2
Icon=
Terminal=false
Type=Application
Categories=Utility;Network;
StartupNotify=true
DESKEOF
    chmod +x "$HOME/.local/share/applications/Pluginfer.desktop" 2>/dev/null || true
    echo "  [5/6] Desktop entry registered: ~/.local/share/applications/Pluginfer.desktop"
else
    echo "  [5/6] Desktop entry: skipped ($OS)"
fi

# 6. Hand off to first-run orchestrator (auto_setup + GUI launch).
echo "  [6/6] Opening Pluginfer GUI..."
echo
cd "$REPO_ROOT/v2"

if [ "$OS" = "Darwin" ]; then
    # On macOS Tk is fine; just launch detached so the shell can exit.
    nohup $PY -m ai.filum.first_run >/dev/null 2>&1 &
    disown 2>/dev/null || true
elif [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
    nohup $PY -m ai.filum.first_run >/dev/null 2>&1 &
    disown 2>/dev/null || true
else
    # Headless Linux (no $DISPLAY) — service mode is the right path.
    echo "         (No display detected — starting headless service.)"
    nohup $PY -m ai.filum.service_mode >/dev/null 2>&1 &
    disown 2>/dev/null || true
fi

echo
echo "  ==================================================================="
echo "    Pluginfer is now running."
echo "    Re-launch later with: $SCRIPT_DIR/Pluginfer.command   (macOS)"
echo "                       or: $PY -m ai.filum.first_run        (Linux)"
echo "  ==================================================================="
echo
