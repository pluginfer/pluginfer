#!/usr/bin/env bash
# install_node.sh — one command to turn a fresh Ubuntu 22.04/24.04
# GPU VPS into a working Pluginfer compute node serving REAL model
# inference (Ollama-backed). Sister script to install_seed.sh.
#
# What it does:
#   1. Installs Python 3.12 + venv + git + Ollama.
#   2. Pulls a small permissively-licensed model (qwen2.5:1.5b).
#   3. Clones Pluginfer at the pinned version.
#   4. Generates a wallet for THIS node if absent.
#   5. Writes /etc/pluginfer/auto_mesh.env with the seed URL.
#   6. Drops the auto_mesh systemd unit, enables, starts.
#   7. Verifies the node came up + Ollama is serving + the runtime
#      adapter resolved to ollama, NOT echo.
#
# Usage (on each compute node):
#   curl -fsSL https://pluginfer.network/deploy/install_node.sh | \
#       sudo bash -s -- \
#         --seed-host seed-eu.pluginfer.network \
#         --seed-port 9000 \
#         --node-port 8101 \
#         --model qwen2.5:1.5b
#
# Verification — after the script finishes:
#   curl http://localhost:8101/peers
#   curl http://localhost:8101/v1/hardware
#   curl -X POST http://localhost:8101/v1/chat/completions \
#        -H 'Content-Type: application/json' \
#        -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
#
# The /v1/hardware response MUST show "runtime.name": "ollama" — if it
# shows "alpha-echo" then the model didn't pull or Ollama isn't running
# and the script will exit non-zero before completing.

set -euo pipefail

PLUGINFER_VERSION="${PLUGINFER_VERSION:-main}"
PLUGINFER_REPO="${PLUGINFER_REPO:-https://github.com/pluginfer/pluginfer.git}"
INSTALL_DIR="/opt/pluginfer"
DATA_DIR="/var/lib/pluginfer"
SEED_HOST="${PLUGINFER_SEED_HOST:-127.0.0.1}"
SEED_PORT="${PLUGINFER_SEED_PORT:-9000}"
NODE_PORT="${PLUGINFER_NODE_PORT:-8101}"
OLLAMA_PORT="11435"   # avoid colliding with Pluginfer devserver default 11434
MODEL_ID="${PLUGINFER_MODEL:-qwen2.5:1.5b}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pluginfer-version) PLUGINFER_VERSION="$2"; shift 2 ;;
        --seed-host) SEED_HOST="$2"; shift 2 ;;
        --seed-port) SEED_PORT="$2"; shift 2 ;;
        --node-port) NODE_PORT="$2"; shift 2 ;;
        --model) MODEL_ID="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

if [[ $EUID -ne 0 ]]; then
    echo "must run as root" >&2
    exit 1
fi

echo "[install_node] step 1/8: system deps"
apt-get update -qq
apt-get install -y -qq \
    python3.12 python3.12-venv python3-pip git curl ufw jq

echo "[install_node] step 2/8: ollama"
if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
# Start ollama on a non-default port so Pluginfer devserver can have 11434.
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/port.conf <<EOF
[Service]
Environment="OLLAMA_HOST=0.0.0.0:$OLLAMA_PORT"
EOF
systemctl daemon-reload
systemctl restart ollama
systemctl enable ollama

echo "[install_node] step 3/8: pulling $MODEL_ID (may take a few minutes)"
OLLAMA_HOST="http://127.0.0.1:$OLLAMA_PORT" ollama pull "$MODEL_ID"

echo "[install_node] step 4/8: pluginfer user + clone"
if ! id -u pluginfer >/dev/null 2>&1; then
    useradd --system --shell /bin/false --home-dir "$DATA_DIR" \
            --create-home pluginfer
fi
mkdir -p "$DATA_DIR"
chown -R pluginfer:pluginfer "$DATA_DIR"
if [[ ! -d "$INSTALL_DIR/.git" ]]; then
    git clone "$PLUGINFER_REPO" "$INSTALL_DIR"
fi
cd "$INSTALL_DIR"
git fetch --tags
git checkout "$PLUGINFER_VERSION"

echo "[install_node] step 5/8: venv + deps"
if [[ ! -d "$INSTALL_DIR/.venv" ]]; then
    python3.12 -m venv "$INSTALL_DIR/.venv"
fi
"$INSTALL_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$INSTALL_DIR/.venv/bin/pip" install --quiet \
    -r "$INSTALL_DIR/v2/api/requirements-devserver.txt" \
    cryptography

echo "[install_node] step 6/8: wallet"
WALLET_PATH="$DATA_DIR/auto_mesh_wallet.pem"
if [[ ! -f "$WALLET_PATH" ]]; then
    PASSPHRASE="$(openssl rand -hex 32)"
    echo "$PASSPHRASE" > "$DATA_DIR/wallet.passphrase"
    chmod 600 "$DATA_DIR/wallet.passphrase"
    chown pluginfer:pluginfer "$DATA_DIR/wallet.passphrase"
fi

echo "[install_node] step 7/8: env + systemd unit"
mkdir -p /etc/pluginfer
PUBLIC_IP="$(curl -s -4 ifconfig.me || echo 0.0.0.0)"
cat > /etc/pluginfer/auto_mesh.env <<EOF
PLUGINFER_SEED_HOST=$SEED_HOST
PLUGINFER_SEED_PORT=$SEED_PORT
PLUGINFER_NODE_PORT=$NODE_PORT
PLUGINFER_PUBLIC_IP=$PUBLIC_IP
PLUGINFER_NODE_ID=$(hostname)-$(openssl rand -hex 4)
OLLAMA_HOST=http://127.0.0.1:$OLLAMA_PORT
PLUGINFER_ALPHA_MODEL_ID=$MODEL_ID
PLUGINFER_WALLET_PASSPHRASE=$(cat $DATA_DIR/wallet.passphrase)
EOF
chmod 600 /etc/pluginfer/auto_mesh.env
cp "$INSTALL_DIR/v2/deploy/auto_mesh.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable auto_mesh.service
systemctl restart auto_mesh.service

ufw allow "$NODE_PORT/tcp" || true
ufw allow "$SEED_PORT/tcp" || true
ufw --force enable || true

echo "[install_node] step 8/8: verify"
# Wait for the node to come up.
for i in {1..30}; do
    if curl -fs "http://localhost:$NODE_PORT/healthz" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Check that the runtime resolved to ollama, NOT echo.
HW="$(curl -fs http://localhost:$NODE_PORT/v1/hardware || echo {})"
RUNTIME="$(echo "$HW" | jq -r '.runtime.name // "missing"')"
IS_ECHO="$(echo "$HW" | jq -r '.runtime.is_echo // true')"
echo
echo "[install_node] runtime: $RUNTIME (is_echo: $IS_ECHO)"
if [[ "$RUNTIME" != "ollama" ]] || [[ "$IS_ECHO" == "true" ]]; then
    echo "[install_node] ERROR: real adapter did NOT resolve. The node is"
    echo "                    running but will serve echo responses. Check:"
    echo "                       - systemctl status ollama"
    echo "                       - OLLAMA_HOST=$OLLAMA_PORT ollama list"
    echo "                       - journalctl -u auto_mesh -n 50"
    exit 2
fi
echo "[install_node] OK: node $PUBLIC_IP:$NODE_PORT serving $MODEL_ID via $RUNTIME"
echo
echo "next:"
echo "  curl http://$PUBLIC_IP:$NODE_PORT/peers"
echo "  curl -X POST http://$PUBLIC_IP:$NODE_PORT/v1/chat/completions \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":32}'"
