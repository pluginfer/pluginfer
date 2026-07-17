#!/bin/bash
# ============================================================================
#   Pluginfer launcher -- macOS double-click target.
#   Runs the §H3 first-run orchestrator: auto_setup if needed, then GUI.
#   On second-and-later launches the auto_setup step is a no-op, so opening
#   is fast.
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."
cd "$REPO_ROOT/v2"

if ! command -v python3 >/dev/null 2>&1; then
    osascript -e 'display dialog "Python 3 not found. Install via https://python.org or `brew install python3`, then run installer/Pluginfer-Setup.sh once."' \
        2>/dev/null || true
    exit 1
fi

# first_run is idempotent: if everything is already set up, it just opens the GUI.
nohup python3 -m ai.filum.first_run >/dev/null 2>&1 &
disown 2>/dev/null || true
exit 0
