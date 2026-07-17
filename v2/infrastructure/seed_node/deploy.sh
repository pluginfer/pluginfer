#!/usr/bin/env bash
# One-command Pluginfer seed-node deployment for fresh Ubuntu 22.04
# VPS instances. Tested against Hetzner / DigitalOcean / Vultr.
#
# Usage:    ./deploy.sh
# Cleanup:  ./deploy.sh down
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/pluginfer}"
PORT="${SEED_PORT:-9000}"
SERVICE="pluginfer-seed"
COMPOSE="docker compose"   # docker-compose v2

cmd_up() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "[deploy] installing docker..."
        sudo apt-get update -y
        sudo apt-get install -y docker.io docker-compose-plugin
    fi

    if [ ! -d "$REPO_DIR" ]; then
        echo "[deploy] cloning Pluginfer to $REPO_DIR"
        sudo git clone --depth 1 https://github.com/pluginfer/pluginfer "$REPO_DIR"
    else
        echo "[deploy] updating Pluginfer at $REPO_DIR"
        sudo git -C "$REPO_DIR" pull --ff-only
    fi

    sudo $COMPOSE \
        -f "$REPO_DIR/v2/infrastructure/seed_node/docker-compose.yml" \
        up -d --build

    sleep 3
    if sudo docker ps --format '{{.Names}}' | grep -q "^${SERVICE}$"; then
        echo "[deploy] OK seed running on port ${PORT}"
        sudo docker exec "$SERVICE" python -c \
            "import socket,json,sys
s=socket.socket();s.settimeout(2);s.connect(('127.0.0.1',9000))
s.sendall(b'{\"op\":\"PING\"}\n')
print(s.recv(4096).decode())"
    else
        echo "[deploy] ERROR container not running"
        sudo docker logs "$SERVICE" --tail 50 || true
        exit 1
    fi
}

cmd_down() {
    sudo $COMPOSE \
        -f "$REPO_DIR/v2/infrastructure/seed_node/docker-compose.yml" \
        down -v
}

case "${1:-up}" in
    up) cmd_up ;;
    down) cmd_down ;;
    *) echo "usage: $0 [up|down]"; exit 2 ;;
esac
