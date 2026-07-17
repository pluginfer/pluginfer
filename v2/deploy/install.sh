#!/usr/bin/env bash
# Pluginfer one-command installer — Mac (Intel + Apple Silicon) and
# Linux (Ubuntu, Debian, Fedora, Arch). Brings up a working auto-mesh
# node serving real model inference via Ollama.
#
# Usage:
#   curl -fsSL https://get.pluginfer.network/install.sh | bash
#
# Or with options:
#   curl -fsSL https://get.pluginfer.network/install.sh | bash -s -- \
#       --seed-host seed.pluginfer.network \
#       --seed-port 9000 \
#       --node-port 8101 \
#       --model qwen2.5:1.5b
#
# Verify after install:
#   curl http://localhost:8101/peers | jq
#   curl http://localhost:8101/v1/hardware | jq '.runtime'
#
# The installer:
#   1. Detects the platform (Mac / Linux distro)
#   2. Installs Python 3.12 + git + jq via the right package manager
#   3. Installs Ollama (cross-platform)
#   4. Pulls the model
#   5. Clones Pluginfer (or downloads release tarball)
#   6. Sets up venv + deps
#   7. Generates per-node wallet
#   8. Boots auto_mesh in the foreground via the supervisor
#
# It WILL NOT silently fall back to the echo runner. If Ollama can't
# load the model, the installer exits non-zero with a clear message.

set -euo pipefail

PLUGINFER_VERSION="${PLUGINFER_VERSION:-main}"
PLUGINFER_REPO="${PLUGINFER_REPO:-https://github.com/pluginfer/pluginfer.git}"
PLUGINFER_RELEASE_URL="${PLUGINFER_RELEASE_URL:-}"
INSTALL_DIR="${PLUGINFER_INSTALL_DIR:-$HOME/.pluginfer}"
SEED_HOST="${PLUGINFER_SEED_HOST:-127.0.0.1}"
SEED_PORT="${PLUGINFER_SEED_PORT:-9000}"
NODE_PORT="${PLUGINFER_NODE_PORT:-8101}"
OLLAMA_PORT="${PLUGINFER_OLLAMA_PORT:-11435}"
MODEL_ID="${PLUGINFER_MODEL:-qwen2.5:1.5b}"
RUN_FOREGROUND="${PLUGINFER_RUN_FOREGROUND:-1}"

# Allow non-root install on Mac/personal Linux; auto-detect sudo when
# we need a system package. The repo + venv go under $HOME, so no
# root is needed for the auto_mesh layer itself.
SUDO=""
if [[ $EUID -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    fi
fi

# --- CLI parsing -----------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed-host)  SEED_HOST="$2"; shift 2 ;;
        --seed-port)  SEED_PORT="$2"; shift 2 ;;
        --node-port)  NODE_PORT="$2"; shift 2 ;;
        --model)      MODEL_ID="$2"; shift 2 ;;
        --version)    PLUGINFER_VERSION="$2"; shift 2 ;;
        --release-url) PLUGINFER_RELEASE_URL="$2"; shift 2 ;;
        --background) RUN_FOREGROUND=0; shift ;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# \?//' | head -40
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# --- Platform detection ---------------------------------------------
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Darwin) PLATFORM="mac" ;;
    Linux)
        PLATFORM="linux"
        if [[ -r /etc/os-release ]]; then
            . /etc/os-release
            DISTRO="$ID"
        else
            DISTRO="unknown"
        fi
        ;;
    *) echo "unsupported OS: $OS — use install.ps1 on Windows" >&2; exit 1 ;;
esac

echo "[pluginfer] detected $PLATFORM ($ARCH)"

# --- Step 1: install Python 3.12 + git + jq + openssl ---------------
echo "[pluginfer] step 1/8: system dependencies"
ensure_cmd() {
    local cmd="$1"; shift
    if command -v "$cmd" >/dev/null 2>&1; then return 0; fi
    if [[ "$PLATFORM" == "mac" ]]; then
        if ! command -v brew >/dev/null 2>&1; then
            echo "[pluginfer] installing Homebrew (needed for $cmd)"
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install "$@"
    else
        case "$DISTRO" in
            ubuntu|debian)
                $SUDO apt-get update -qq
                $SUDO apt-get install -y -qq "$@" ;;
            fedora|rhel|centos)
                $SUDO dnf install -y -q "$@" ;;
            arch|manjaro)
                $SUDO pacman -Sy --noconfirm "$@" ;;
            *) echo "unknown distro $DISTRO — install '$cmd' manually"; exit 1 ;;
        esac
    fi
}

if ! command -v python3 >/dev/null 2>&1; then
    if [[ "$PLATFORM" == "mac" ]]; then
        ensure_cmd python3 python@3.12
    else
        case "$DISTRO" in
            ubuntu|debian)
                ensure_cmd python3 python3.12 python3.12-venv python3-pip ;;
            fedora|rhel|centos)
                ensure_cmd python3 python3.12 python3-pip ;;
            arch|manjaro)
                ensure_cmd python3 python ;;
        esac
    fi
fi
ensure_cmd git git
ensure_cmd jq jq
ensure_cmd openssl openssl

# --- Step 2: install Ollama -----------------------------------------
echo "[pluginfer] step 2/8: Ollama"
if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Configure Ollama on a non-default port so it doesn't collide with
# Pluginfer's devserver (port 11434). Mac uses launchd; Linux uses
# systemd (or just env-var for non-systemd setups).
if [[ "$PLATFORM" == "mac" ]]; then
    # Mac Ollama reads OLLAMA_HOST from launchctl env.
    launchctl setenv OLLAMA_HOST "0.0.0.0:$OLLAMA_PORT" 2>/dev/null || true
    if pgrep -x ollama >/dev/null; then
        pkill -x ollama || true
        sleep 1
    fi
    OLLAMA_HOST="0.0.0.0:$OLLAMA_PORT" nohup ollama serve >/tmp/ollama.log 2>&1 &
    sleep 2
elif systemctl --version >/dev/null 2>&1 && systemctl list-unit-files | grep -q ollama; then
    $SUDO mkdir -p /etc/systemd/system/ollama.service.d
    echo -e "[Service]\nEnvironment=\"OLLAMA_HOST=0.0.0.0:$OLLAMA_PORT\"" \
        | $SUDO tee /etc/systemd/system/ollama.service.d/port.conf >/dev/null
    $SUDO systemctl daemon-reload
    $SUDO systemctl restart ollama
else
    # Fallback: run Ollama in the background.
    OLLAMA_HOST="0.0.0.0:$OLLAMA_PORT" nohup ollama serve >/tmp/ollama.log 2>&1 &
    sleep 2
fi

# Wait for Ollama API to come up.
for i in {1..30}; do
    if curl -fs "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

# --- Step 3: pull the model -----------------------------------------
echo "[pluginfer] step 3/8: pull $MODEL_ID (may take a few minutes)"
OLLAMA_HOST="http://127.0.0.1:$OLLAMA_PORT" ollama pull "$MODEL_ID"

# --- Step 4: clone or download Pluginfer ----------------------------
echo "[pluginfer] step 4/8: Pluginfer source"
mkdir -p "$INSTALL_DIR"
if [[ -n "$PLUGINFER_RELEASE_URL" ]]; then
    curl -fsSL "$PLUGINFER_RELEASE_URL" -o "$INSTALL_DIR/release.tar.gz"
    tar -xzf "$INSTALL_DIR/release.tar.gz" -C "$INSTALL_DIR" --strip-components=1
elif [[ ! -d "$INSTALL_DIR/.git" ]]; then
    git clone "$PLUGINFER_REPO" "$INSTALL_DIR/repo"
    INSTALL_DIR="$INSTALL_DIR/repo"
else
    INSTALL_DIR="$INSTALL_DIR/repo"
fi
cd "$INSTALL_DIR"
git fetch --tags 2>/dev/null || true
git checkout "$PLUGINFER_VERSION" 2>/dev/null || true

# --- Step 5: venv + Python deps -------------------------------------
echo "[pluginfer] step 5/8: Python virtualenv"
if [[ ! -d "$INSTALL_DIR/.venv" ]]; then
    python3 -m venv "$INSTALL_DIR/.venv"
fi
"$INSTALL_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$INSTALL_DIR/.venv/bin/pip" install --quiet \
    -r "$INSTALL_DIR/v2/api/requirements-devserver.txt" \
    cryptography

# --- Step 6: wallet -------------------------------------------------
echo "[pluginfer] step 6/8: wallet"
DATA_DIR="$HOME/.pluginfer"
mkdir -p "$DATA_DIR"
WALLET_PATH="$DATA_DIR/auto_mesh_wallet.pem"
PASSPHRASE_FILE="$DATA_DIR/wallet.passphrase"
if [[ ! -f "$PASSPHRASE_FILE" ]]; then
    openssl rand -hex 32 > "$PASSPHRASE_FILE"
    chmod 600 "$PASSPHRASE_FILE"
fi

# --- Step 7: write env + boot ---------------------------------------
echo "[pluginfer] step 7/8: configure"
cat > "$DATA_DIR/auto_mesh.env" <<EOF
PLUGINFER_SEED_HOST=$SEED_HOST
PLUGINFER_SEED_PORT=$SEED_PORT
PLUGINFER_NODE_PORT=$NODE_PORT
PLUGINFER_NODE_ID=$(hostname)-$(openssl rand -hex 4)
OLLAMA_HOST=http://127.0.0.1:$OLLAMA_PORT
PLUGINFER_ALPHA_MODEL_ID=$MODEL_ID
PLUGINFER_WALLET_PASSPHRASE=$(cat "$PASSPHRASE_FILE")
PLUGINFER_JOBS_DB=$DATA_DIR/jobs.db
EOF
chmod 600 "$DATA_DIR/auto_mesh.env"

# --- Step 8: boot the node ------------------------------------------
echo "[pluginfer] step 8/8: starting auto_mesh"
# Source the env into THIS shell.
set -a; . "$DATA_DIR/auto_mesh.env"; set +a

cd "$INSTALL_DIR/v2"

# Boot in the background briefly to verify the runtime resolved to
# Ollama. If the user wants long-lived foreground, we then start it
# again under the supervisor wrapper.
"$INSTALL_DIR/.venv/bin/python" -m tools.auto_mesh \
    --seed-host "$SEED_HOST" --seed-port "$SEED_PORT" \
    --node-port "$NODE_PORT" --wallet-path "$WALLET_PATH" \
    > /tmp/pluginfer_node.log 2>&1 &
NODE_PID=$!

# Wait for /healthz.
for i in {1..30}; do
    if curl -fs "http://localhost:$NODE_PORT/healthz" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Verify the real adapter resolved.
HW="$(curl -fs http://localhost:$NODE_PORT/v1/hardware || echo '{}')"
RUNTIME="$(echo "$HW" | jq -r '.runtime.name // "missing"')"
IS_ECHO="$(echo "$HW" | jq -r '.runtime.is_echo // true')"

if [[ "$RUNTIME" != "ollama" ]] || [[ "$IS_ECHO" == "true" ]]; then
    kill $NODE_PID 2>/dev/null || true
    cat <<EOF >&2

[pluginfer] ERROR: real adapter did NOT resolve.
   Runtime: $RUNTIME
   is_echo: $IS_ECHO
   Tail of node log:
$(tail -20 /tmp/pluginfer_node.log)

Common fixes:
  * Confirm Ollama is running:    curl http://127.0.0.1:$OLLAMA_PORT/api/tags
  * Confirm the model is pulled:  OLLAMA_HOST=http://127.0.0.1:$OLLAMA_PORT ollama list
  * Re-run this installer with --model qwen2.5:1.5b (or another tag)

EOF
    exit 2
fi

echo
echo "[pluginfer] OK — node up on http://localhost:$NODE_PORT"
echo "[pluginfer]      runtime: $RUNTIME ($MODEL_ID)"
echo "[pluginfer]      seed:    $SEED_HOST:$SEED_PORT"
echo
echo "Try it:"
echo "  curl http://localhost:$NODE_PORT/peers | jq"
echo "  curl -X POST http://localhost:$NODE_PORT/v1/chat/completions \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":32,\"pluginfer_cost_ceiling_usd\":0.05}'"
echo
echo "Logs:   tail -f /tmp/pluginfer_node.log"
echo "Stop:   kill $NODE_PID"

if [[ "$RUN_FOREGROUND" == "1" ]]; then
    echo "[pluginfer] now switching to foreground supervisor (Ctrl+C to stop)"
    kill $NODE_PID 2>/dev/null || true
    sleep 1
    exec "$INSTALL_DIR/.venv/bin/python" -m tools.run_node \
        --seed-host "$SEED_HOST" --seed-port "$SEED_PORT" \
        --node-port "$NODE_PORT" --wallet-path "$WALLET_PATH"
fi
