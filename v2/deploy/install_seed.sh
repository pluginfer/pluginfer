#!/usr/bin/env bash
# install_seed.sh — bring a fresh Ubuntu 22.04 / 24.04 LTS box up
# as a Pluginfer public seed in ~5 minutes. Run as root on a clean
# VPS (Hetzner CX22 / Linode Nanode / DO 1-shared works fine).
#
# What it does:
#   1. Creates `pluginfer` user + data dir.
#   2. Installs Python 3.12 + venv + pip.
#   3. Clones the Pluginfer repo at the pinned commit.
#   4. Generates a fresh wallet for THIS seed if none exists.
#   5. Drops the systemd unit, enables it, starts it.
#   6. Prints the seed's public key fingerprint so the operator
#      can publish it in the cross-region quorum-signed registry.
#
# Usage:
#   curl -fsSL https://pluginfer.network/deploy/install_seed.sh | \
#       sudo bash -s -- --pluginfer-version v0.1.0
#
# Idempotent — re-running upgrades to the new pinned commit.

set -euo pipefail

PLUGINFER_VERSION="${PLUGINFER_VERSION:-main}"
PLUGINFER_REPO="${PLUGINFER_REPO:-https://github.com/pluginfer/pluginfer.git}"
INSTALL_DIR="/opt/pluginfer"
DATA_DIR="/var/lib/pluginfer"
SEED_PORT="${PLUGINFER_SEED_PORT:-9000}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pluginfer-version) PLUGINFER_VERSION="$2"; shift 2 ;;
        --port) SEED_PORT="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

if [[ $EUID -ne 0 ]]; then
    echo "must run as root" >&2
    exit 1
fi

echo "[install_seed] installing Pluginfer seed at $INSTALL_DIR (version=$PLUGINFER_VERSION)"

# 1. System packages.
apt-get update -qq
apt-get install -y -qq \
    python3.12 python3.12-venv python3-pip git ufw

# 2. Pluginfer user.
if ! id -u pluginfer >/dev/null 2>&1; then
    useradd --system --shell /bin/false --home-dir "$DATA_DIR" \
            --create-home pluginfer
fi
mkdir -p "$DATA_DIR"
chown -R pluginfer:pluginfer "$DATA_DIR"

# 3. Repo checkout.
if [[ ! -d "$INSTALL_DIR/.git" ]]; then
    git clone "$PLUGINFER_REPO" "$INSTALL_DIR"
fi
cd "$INSTALL_DIR"
git fetch --tags
git checkout "$PLUGINFER_VERSION"

# 4. Virtualenv.
if [[ ! -d "$INSTALL_DIR/.venv" ]]; then
    python3.12 -m venv "$INSTALL_DIR/.venv"
fi
"$INSTALL_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$INSTALL_DIR/.venv/bin/pip" install --quiet \
    -r "$INSTALL_DIR/v2/api/requirements-devserver.txt" \
    cryptography pynacl

# 5. Generate seed wallet if absent.
WALLET_PATH="$DATA_DIR/seed_wallet.pem"
if [[ ! -f "$WALLET_PATH" ]]; then
    "$INSTALL_DIR/.venv/bin/python" -c "
import sys
sys.path.insert(0, '$INSTALL_DIR/v2')
from core.tokenomics import Wallet
w = Wallet()
w.save_to_file('$WALLET_PATH', passphrase=b'$(openssl rand -hex 32)')
print(w.public_key_pem)
" > /tmp/seed_pubkey.pem
    chown pluginfer:pluginfer "$WALLET_PATH"
    chmod 600 "$WALLET_PATH"
fi

# 6. systemd unit.
cp "$INSTALL_DIR/v2/deploy/seed_node.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable seed_node.service
systemctl restart seed_node.service

# 7. Firewall.
ufw allow "$SEED_PORT/tcp"
ufw --force enable || true

echo "[install_seed] seed running on tcp/$SEED_PORT"
echo "[install_seed] seed pubkey (PUBLISH this in seed_registry.json):"
cat /tmp/seed_pubkey.pem
echo
echo "[install_seed] next:"
echo "  - publish the pubkey in your quorum-signed seed_registry.json"
echo "  - confirm reachability:  nc -zv <this-host> $SEED_PORT"
echo "  - watch logs:  journalctl -u seed_node -f"
