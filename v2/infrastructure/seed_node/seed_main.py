"""Combined entrypoint for the Pluginfer seed node.

Runs the TCP REGISTER/PEERS server (`seed_server.py`) AND the UDP
hole-punch + TURN relay server (`punch_server.py`) on the same port.
Both are required for a useful seed -- TCP for first-boot peer
discovery, UDP for the symmetric-NAT survival path.

Production usage (the deploy.sh wraps this):

    PLUGINFER_SEED_PORT=9000 python -m infrastructure.seed_node.seed_main
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal

from .punch_server import run_punch_server
from .seed_server import run_server as run_tcp_server

logger = logging.getLogger("seed_main")


async def _run(host: str, port: int) -> None:
    logging.basicConfig(
        level=os.environ.get("PLUGINFER_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info(
        "Starting Pluginfer seed: TCP+UDP on %s:%d (TCP for REGISTER/PEERS, "
        "UDP for hole-punch + TURN relay).",
        host, port,
    )
    tcp = asyncio.create_task(run_tcp_server(host=host, port=port),
                              name="seed-tcp")
    udp = asyncio.create_task(run_punch_server(host=host, port=port),
                              name="seed-udp")
    stop = asyncio.Event()

    def _handle_stop() -> None:
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in ("SIGINT", "SIGTERM"):
        if hasattr(signal, sig):
            try:
                loop.add_signal_handler(getattr(signal, sig), _handle_stop)
            except NotImplementedError:
                # Windows: signal handlers via add_signal_handler aren't
                # available; KeyboardInterrupt covers SIGINT.
                pass

    try:
        await stop.wait()
    finally:
        for t in (tcp, udp):
            t.cancel()
        for t in (tcp, udp):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Pluginfer seed (TCP+UDP).")
    parser.add_argument("--host",
                        default=os.environ.get("PLUGINFER_SEED_HOST", "0.0.0.0"))
    parser.add_argument("--port",
                        default=int(os.environ.get("PLUGINFER_SEED_PORT", "9000")),
                        type=int)
    args = parser.parse_args()
    try:
        asyncio.run(_run(host=args.host, port=args.port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
