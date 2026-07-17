"""Supervisor wrapper — auto-restart for `tools.auto_mesh`.

The auto_mesh node is the long-running production process. If
uvicorn crashes (OOM, segfault from a misbehaving native lib, a
peer that triggers a Python exception bypassed by the discovery
loop), we want it to come back up automatically. systemd / launchd
handle this in deployed setups, but operators on a laptop / Docker /
Hetzner without systemd need a portable supervisor.

Behaviour:
  * spawn `python -m tools.auto_mesh <args>` as a subprocess,
  * stream its stdout/stderr to ours,
  * on exit, if exit code != 0 AND it ran for > `restart_min_uptime_s`,
    restart with exponential backoff up to `max_restart_delay_s`,
  * SIGINT / SIGTERM propagate to the child cleanly.

Crash loops (child dies in under `restart_min_uptime_s`) trigger a
short-cool-off so a misconfigured node doesn't burn CPU. Operators
get a clear log line on each restart with reason + delay.

CLI:

    python -m tools.run_node \
        --seed-host seed-eu.pluginfer.network \
        --seed-port 9000 \
        --node-port 8101

All flags after `python -m tools.run_node` pass through to
`tools.auto_mesh` unchanged.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

logger = logging.getLogger("pluginfer.run_node")

DEFAULT_INITIAL_DELAY_S = 1.0
DEFAULT_MAX_DELAY_S = 60.0
DEFAULT_MIN_UPTIME_S = 10.0


def _spawn_child(child_args: list) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "tools.auto_mesh", *child_args]
    return subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=sys.stdout, stderr=sys.stderr,
    )


def supervise(child_args: list, *,
              initial_delay_s: float = DEFAULT_INITIAL_DELAY_S,
              max_delay_s: float = DEFAULT_MAX_DELAY_S,
              min_uptime_s: float = DEFAULT_MIN_UPTIME_S) -> int:
    """Main supervisor loop. Returns the exit code of the LAST child
    process when we stop trying (SIGINT or clean exit)."""
    delay = initial_delay_s
    while True:
        t0 = time.monotonic()
        child = _spawn_child(child_args)
        logger.info("supervisor: started auto_mesh pid=%s", child.pid)

        def _forward(sig, _frame):
            logger.info("supervisor: forwarding signal %s to child", sig)
            try:
                child.send_signal(sig)
            except ProcessLookupError:
                pass

        prev_int = signal.signal(signal.SIGINT, _forward)
        prev_term = signal.signal(signal.SIGTERM, _forward)
        try:
            ret = child.wait()
        finally:
            signal.signal(signal.SIGINT, prev_int)
            signal.signal(signal.SIGTERM, prev_term)
        uptime = time.monotonic() - t0
        if ret == 0:
            logger.info("supervisor: child exited cleanly; not restarting")
            return 0
        logger.warning(
            "supervisor: child exited code=%s after %.1fs",
            ret, uptime,
        )
        if uptime < min_uptime_s:
            # Crash loop: back off to avoid wasting CPU.
            delay = min(max_delay_s, delay * 2.0)
        else:
            # Long-lived crash: a single restart is usually right.
            delay = initial_delay_s
        logger.warning("supervisor: restart in %.1fs", delay)
        time.sleep(delay)


def main() -> int:
    # The supervisor is the root of the whole node process tree, so
    # installing here puts the auto_mesh child (and ITS children) under
    # one job-object memory cap + below-normal priority. Kill-on-close
    # also guarantees no orphaned node survives a dead supervisor.
    import host_guard
    host_guard.install("run_node")
    logging.basicConfig(
        level=os.environ.get("PLUGINFER_LOG_LEVEL", "INFO"),
        format="[run_node] %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Pluginfer auto_mesh supervisor — restart on crash."
    )
    parser.add_argument(
        "--restart-initial-delay", type=float,
        default=DEFAULT_INITIAL_DELAY_S,
    )
    parser.add_argument(
        "--restart-max-delay", type=float, default=DEFAULT_MAX_DELAY_S,
    )
    parser.add_argument(
        "--restart-min-uptime", type=float, default=DEFAULT_MIN_UPTIME_S,
    )
    # Everything after `--` (or unrecognised by us) is passed to
    # auto_mesh as-is.
    args, child_args = parser.parse_known_args()
    return supervise(
        list(child_args),
        initial_delay_s=args.restart_initial_delay,
        max_delay_s=args.restart_max_delay,
        min_uptime_s=args.restart_min_uptime,
    )


if __name__ == "__main__":
    sys.exit(main())
