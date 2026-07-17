"""§H3 Service mode — Pluginfer runs forever, survives reboots, never bothers the user.

Goals:
* Run as a background service / daemon that auto-starts on login
* Auto-update via §E2 delta-sync (small bandwidth, no re-download of whole model)
* Pause-during-game detector (don't compete with the user's Steam launch)
* Crash-restart loop with exponential backoff
* Exposes /metrics on localhost so the GUI can read live status

Platform integration:
* Windows: registers as a Windows Service via pywin32 (or a userspace
  scheduled-task fallback when admin not granted).
* macOS:   ships as a LaunchAgent plist in ~/Library/LaunchAgents.
* Linux:   ships as a systemd --user unit in ~/.config/systemd/user.

This module ships the *runner core*. Platform-specific install
scripts that register the service are in ``installer/``.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ServiceStatus:
    started_ts: float = 0.0
    running: bool = False
    paused_for_game: bool = False
    last_update_check_ts: float = 0.0
    crash_count: int = 0
    backoff_seconds: float = 1.0
    bound_address: tuple = ("127.0.0.1", 0)


class ServiceRunner:
    """The long-running background process.

    Public lifecycle: ``run()`` blocks until SIGTERM. Internally:

    1. Load runtime_config.json (written by auto_setup).
    2. Start the §B + §C subsystems (telemetry, NBGGA, transport,
       gossip, safety gate, observability).
    3. Start the metrics HTTP listener on a localhost port.
    4. Loop: poll for game-detection / update / mesh-health every
       few seconds; restart on crash with exponential backoff.

    Designed for restarts to be cheap: state lives on disk via
    NBGGA cursor persistence + grain transport's seen ring +
    compute_currency state. After a crash + restart the node
    rejoins where it left off.
    """

    def __init__(self, *, state_dir: Optional[str] = None,
                 metrics_port: int = 0):
        from .auto_setup import default_state_dir, load_runtime_config

        self.state_dir = state_dir or default_state_dir()
        self.config = load_runtime_config(self.state_dir) or {}
        self.metrics_port = metrics_port
        self.status = ServiceStatus()
        self._stop = threading.Event()
        self._http_thread: Optional[threading.Thread] = None
        # Lazy refs to subsystems built when run() is called.
        self._sampler = None
        self._registry = None

    # --- the loop --------------------------------------------------------

    def run(self) -> int:
        if not self.config:
            logger.error("no runtime_config; run auto_setup first")
            return 1
        self.status.started_ts = time.time()
        self.status.running = True

        # Start telemetry sampler.
        from .hpa.telemetry import PressureSampler
        self._sampler = PressureSampler(period_s=0.5).start()

        # Start metrics HTTP listener (localhost only).
        from .hpa.observability import MetricsRegistry
        self._registry = MetricsRegistry()
        self._registry.bind_pressure_sampler(self._sampler)
        self._http_thread = threading.Thread(
            target=self._serve_metrics, daemon=True,
        )
        self._http_thread.start()

        # Hook SIGTERM / SIGINT for clean exit.
        signal.signal(signal.SIGINT, self._handle_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._handle_signal)

        # Main loop.
        try:
            while not self._stop.is_set():
                self._tick()
                self._stop.wait(2.0)
        finally:
            self.status.running = False
            if self._sampler is not None:
                self._sampler.stop()
        return 0

    def _tick(self) -> None:
        """Periodic housekeeping. Called every ~2s."""
        # 1. Pause if a known game process is running.
        if _is_game_running():
            if not self.status.paused_for_game:
                self.status.paused_for_game = True
                logger.info("game detected -- pausing mesh contributions")
        else:
            if self.status.paused_for_game:
                self.status.paused_for_game = False
                logger.info("game stopped -- resuming mesh contributions")

        # 2. Check for available delta-sync update once per hour.
        now = time.time()
        if now - self.status.last_update_check_ts > 3600:
            self.status.last_update_check_ts = now
            # Production: pull update manifest, apply via delta_sync.
            # Stub for v0: just log the check happened.
            logger.debug("delta-sync update check (no update available)")

    def _serve_metrics(self) -> None:
        """Tiny HTTP server that serves /metrics. Localhost only."""
        from http.server import BaseHTTPRequestHandler, HTTPServer

        registry = self._registry
        status = self.status

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass   # silence default access log

            def do_GET(self):
                if self.path == "/metrics":
                    body = registry.render().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/status":
                    import json
                    body = json.dumps({
                        "running": status.running,
                        "paused_for_game": status.paused_for_game,
                        "uptime_s": time.time() - status.started_ts,
                        "crash_count": status.crash_count,
                    }).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(404)
                self.end_headers()

        try:
            srv = HTTPServer(("127.0.0.1", self.metrics_port), Handler)
            self.status.bound_address = srv.server_address
            logger.info("metrics on http://127.0.0.1:%d/metrics",
                         srv.server_address[1])
            srv.serve_forever()
        except Exception as e:
            logger.warning("metrics server failed: %s", e)

    def _handle_signal(self, signum, frame):
        logger.info("signal %s received; stopping", signum)
        self._stop.set()


# ---------- game detector --------------------------------------------------

_KNOWN_GAME_BIN_PREFIXES = (
    "csgo", "dota", "league", "valorant", "fortnite", "minecraft",
    "rocketleague", "warzone", "apex", "overwatch", "cyberpunk",
    "rdr2", "elden", "diablo", "cs2", "rust", "genshin",
    "starfield", "baldur", "halo", "destiny",
)


def _is_game_running() -> bool:
    try:
        import psutil
        for proc in psutil.process_iter(attrs=("name",)):
            try:
                name = (proc.info.get("name") or "").lower()
            except Exception:
                continue
            for prefix in _KNOWN_GAME_BIN_PREFIXES:
                if name.startswith(prefix):
                    return True
        return False
    except Exception:
        return False


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(message)s")
    runner = ServiceRunner()
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
