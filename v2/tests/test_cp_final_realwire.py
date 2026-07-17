"""CP-FINAL-REAL — two real OS processes mesh over real TCP and
complete one job end-to-end. This is the test that ends the
"but does it ACTUALLY work in the wild?" objection.

Three processes:
  * Gateway (uvicorn on a free localhost port).
  * Provider (polls /open_jobs, executes the prompt, delivers).
  * Buyer (this test's parent process — POSTs /v1/chat/completions,
    verifies the signed receipt).

The test soft-skips on environments without `uvicorn` available so
CI matrices without a working uvicorn install (rare) don't go red.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


V2 = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for_health(url: str, deadline_s: float) -> bool:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if r.status == 200:
                    return True
        except urllib.error.URLError:
            pass
        time.sleep(0.2)
    return False


def _have_uvicorn() -> bool:
    try:
        import uvicorn  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _have_uvicorn(), reason="uvicorn not installed")
def test_cp_final_realwire_two_processes_complete_one_job():
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(V2) + os.pathsep + env.get("PYTHONPATH", "")
    # Quiet down noisy logs in CI.
    env["PYTHONUNBUFFERED"] = "1"

    gateway_proc = subprocess.Popen(
        [sys.executable, "-m", "tools.run_realwire_demo",
         "--role", "gateway", "--host", "127.0.0.1",
         "--port", str(port)],
        cwd=str(V2),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        assert _wait_for_health(f"{base}/healthz", 15.0), \
            "gateway never reported healthz"

        provider_proc = subprocess.Popen(
            [sys.executable, "-m", "tools.run_realwire_demo",
             "--role", "provider", "--gateway", base, "--run-for", "25"],
            cwd=str(V2),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            # Brief settle so the provider has registered before we
            # submit the chat. The provider's first /register hits
            # within a few hundred ms.
            time.sleep(1.0)

            buyer_proc = subprocess.run(
                [sys.executable, "-m", "tools.run_realwire_demo",
                 "--role", "buyer", "--gateway", base],
                cwd=str(V2),
                env=env,
                capture_output=True,
                text=True,
                timeout=40,
            )
            assert buyer_proc.returncode == 0, (
                f"buyer exited {buyer_proc.returncode}\n"
                f"stdout: {buyer_proc.stdout}\n"
                f"stderr: {buyer_proc.stderr}\n"
            )
            # The buyer's last JSON line summarises the round trip.
            tail_lines = [
                ln for ln in buyer_proc.stdout.splitlines()
                if ln.startswith("{")
            ]
            assert tail_lines, f"no JSON summary in {buyer_proc.stdout}"
            summary = json.loads(tail_lines[-1])
            assert summary["ok"] is True
            assert summary["receipt_signed"] is True
            assert summary["round_trip_ms"] > 0
            assert summary["job_id"]
            assert summary["provider_id"]
        finally:
            provider_proc.terminate()
            try:
                provider_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                provider_proc.kill()
    finally:
        gateway_proc.terminate()
        try:
            gateway_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            gateway_proc.kill()
