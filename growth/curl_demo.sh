#!/usr/bin/env bash
# Pluginfer 30-second demo.
#
# Boots the devserver image, fires one chat-completion through the
# OpenAI-compatible shim, fetches the signed PNIS receipt, verifies
# the signature. Drop the script in any DM / blog / README — anyone
# can run it in 30 seconds and see the audit-trail moat live.
#
# Requires:
#   * docker (or podman) on PATH
#   * curl
#   * (optional) python3 to verify the receipt signature locally

set -euo pipefail

PORT="${PORT:-11434}"
IMAGE="${PLUGINFER_DEVSERVER_IMAGE:-pluginfer/devserver:latest}"

color() { printf "\033[1;%dm%s\033[0m" "$1" "$2"; }
step()  { echo; color 36 "▎ $*"; echo; }

step "1/4  Booting devserver on :$PORT (image=$IMAGE)"
CID=$(docker run -d --rm -p "$PORT:11434" "$IMAGE")
trap 'docker rm -f "$CID" >/dev/null 2>&1 || true' EXIT

# Wait for /healthz.
for _ in $(seq 1 50); do
  if curl --fail --silent "http://127.0.0.1:$PORT/healthz" >/dev/null; then
    break
  fi
  sleep 0.2
done
color 32 "✓ devserver healthy"; echo

step "2/4  Calling /v1/chat/completions"
RESP_FILE=$(mktemp)
HEADER_FILE=$(mktemp)
trap 'rm -f "$RESP_FILE" "$HEADER_FILE"' EXIT

curl --fail --silent \
  -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "content-type: application/json" \
  -D "$HEADER_FILE" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role":"user","content":"hi from pluginfer"}],
    "max_tokens": 64
  }' \
  -o "$RESP_FILE"

echo "─ response body ─"
cat "$RESP_FILE"
echo
echo "─ response headers ─"
grep -i "^x-pluginfer" "$HEADER_FILE" | sed 's/^/  /'

JOB_ID=$(grep -i "^x-pluginfer-job-id:" "$HEADER_FILE" | awk '{print $2}' | tr -d '\r\n')

step "3/4  Fetching the signed PNIS receipt"
curl --fail --silent "http://127.0.0.1:$PORT/v1/receipts/$JOB_ID" \
  | python3 -m json.tool 2>/dev/null \
  || curl --fail --silent "http://127.0.0.1:$PORT/v1/receipts/$JOB_ID"

step "4/4  Verify (optional — requires the pluginfer Python SDK)"
if command -v python3 >/dev/null && python3 -c "import pluginfer" 2>/dev/null; then
  python3 <<'PY'
import json, os, urllib.request
job_id = os.environ.get("JOB_ID", "")
host = os.environ.get("HOST", "http://127.0.0.1:11434")
with urllib.request.urlopen(f"{host}/v1/receipts/{job_id}") as r:
    receipt = json.loads(r.read().decode("utf-8"))
from pluginfer.receipt import AIReceipt
ok = AIReceipt.from_dict(receipt).verify()
print(f"signature verified: {ok}")
PY
else
  color 33 "  pluginfer SDK not installed; skipping local verify"
  echo
  echo "  pip install pluginfer    # then re-run; receipt.verify() returns True"
fi

echo
color 32 "✓ done — every request leaves a tamper-evident audit trail."
