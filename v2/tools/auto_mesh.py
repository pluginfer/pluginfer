"""Pluginfer auto-mesh node — two strangers, zero config, one binary.

This script is **the product test** from TODO §1.3:

  Two people who don't know each other each install Pluginfer and run
  this one command. Within ~30 seconds, both nodes have:

    1. Generated their own wallet (encrypted at rest, W31 path).
    2. Discovered their public IP (RFC-5737-safe, no DNS leak).
    3. Registered with a published seed node (TCP, ECDSA-signed
       REGISTER per `infrastructure/seed_node/`).
    4. Pulled the live peer list from the seed.
    5. Registered as a *provider* with every discovered peer's
       gateway (`/v1/providers/register`), so their compute is
       offered to the other end's auction.
    6. Hosted a local Pluginfer gateway (the §A21 OpenAI/Anthropic
       shim) so they can both submit AND serve work.
    7. Started polling every peer's `/v1/providers/open_jobs` for
       jobs they're qualified to run.

  Neither user types an IP. Neither user knows the other exists.
  The application takes care of the rest.

  When user-A submits `OPENAI_BASE_URL=http://localhost:<a>/v1`
  curl through the gateway, the auction routes the job to whichever
  registered provider (user-A itself OR user-B across the wire) bids
  best on the Pareto-scored cost/latency/quality/privacy axis. The
  result lands as a signed §A1 PNIS receipt.

Usage:

    # Run the seed (you, or anyone — once it's up, anyone can join):
    python -m infrastructure.seed_node.seed_main --port 9000

    # Each stranger runs this on their own machine, pointing at the seed:
    python -m tools.auto_mesh \\
        --seed-host seed.pluginfer.network \\
        --seed-port 9000 \\
        --node-port 8101

The defaults (`--seed-host 127.0.0.1 --seed-port 9000 --node-port 0`)
make a local two-process demo trivial.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# MODULE-level on purpose: this file uses `from __future__ import
# annotations`, so FastAPI resolves endpoint annotations against module
# globals. A function-local `from fastapi import Request` silently
# degrades every `request: Request` endpoint into a required ?request=
# query param (422s in production). Keep these here.
from fastapi import Request
from fastapi.responses import JSONResponse

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

logger = logging.getLogger("auto_mesh")

NODE_VERSION = "1.0.0"
SEED_REGISTER_INTERVAL_S = 60.0     # seed entries TTL at 600s; refresh 10x faster
PEER_POLL_INTERVAL_S = 12.0
PROVIDER_REGISTER_INTERVAL_S = 30.0
PROVIDER_JOB_POLL_INTERVAL_S = 1.0
GOSSIP_TICK_INTERVAL_S = 6.0        # gossip layer firing cadence
HEARTBEAT_TICK_INTERVAL_S = 10.0    # cross-node liveness probe cadence
EXECUTE_RUNNER = "flagship-alpha"   # we serve the alpha-tier flagship locally


# ---------------------------------------------------------------------------
# Local provider — runs jobs in-process when peers submit them to us.
# Wrapped as a §G4 flagship so PNIS receipts stamp the upstream model id.
#
# Resolution order at boot:
#   1. core.runtime_adapters.autodetect_runner() — picks the first
#      adapter whose deps are satisfied (ollama / llama-cpp /
#      transformers). When this resolves, the node serves REAL
#      inference and the alpha tag flips to the upstream model id.
#   2. Echo fallback — deterministic, clearly tagged, never lies
#      about being a real model. The receipt's `runtime` field
#      surfaces which path executed so audit reviewers cannot
#      mistake an echo for a Qwen forward pass.
# ---------------------------------------------------------------------------

# Mutable holder so build_node_app can publish the chosen runtime
# at /v1/hardware without re-resolving each call.
_RUNTIME_STATE: Dict[str, Any] = {"name": "alpha-echo", "model_id": "echo"}


def _echo_runner(prompt: str, payload: Dict[str, Any]) -> bytes:
    """Deterministic echo. Used when no real adapter is reachable.
    Clearly tagged as echo so receipts never mis-advertise capability."""
    text = (
        f"pluginfer-alpha: {prompt[:200]}\n"
        f"served-by-node: {os.environ.get('PLUGINFER_NODE_ID', 'unset')}\n"
    )
    return text.encode("utf-8")


def _resolve_alpha_runner() -> tuple[Any, str, str]:
    """Return (runner_fn, adapter_name, model_id). On any failure
    (no Ollama running, no GGUF dropped, no torch installed) we
    return the echo and tag it honestly. Never raises."""
    preferred_model = os.environ.get(
        "PLUGINFER_ALPHA_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct",
    )
    # Operator can force the echo path (e.g. for the hermetic
    # two-strangers test) via PLUGINFER_FORCE_ECHO=1.
    if os.environ.get("PLUGINFER_FORCE_ECHO") == "1":
        return _echo_runner, "alpha-echo", "echo"
    try:
        from core.runtime_adapters import autodetect_runner
        from core.runtime_adapters.base import _REGISTRY
        runner = autodetect_runner(model_id=preferred_model)
        # Recover which adapter actually resolved by re-probing
        # (cheap; probe paths are no-ops).
        chosen = "real-adapter"
        for name, factory in _REGISTRY:
            try:
                factory(model_id=preferred_model, _probe=True)
                chosen = name
                break
            except Exception:
                continue
        # Adapters may remap to what the runtime actually has (e.g.
        # Ollama negotiating to a pulled tag); receipts must stamp
        # THAT model, so trust the runner over the request.
        served = getattr(runner, "served_model_id", preferred_model)
        return runner, chosen, served
    except Exception as e:
        logger.info(
            "no real runtime adapter available (%s) — running echo. "
            "Install ollama or llama-cpp-python for real inference.",
            type(e).__name__,
        )
        return _echo_runner, "alpha-echo", "echo"


# ---------------------------------------------------------------------------
# Wallet at rest (W31 encryption)
# ---------------------------------------------------------------------------

def _load_or_create_wallet(wallet_path: Path, passphrase: bytes,
                           synthesized: bool = False):
    """Returns a `core.tokenomics.Wallet` instance, loaded from disk if
    present and encrypted with the supplied passphrase; brand new
    otherwise. Generation is encrypted-on-write per W31 — refusing to
    write plaintext keys.

    Recovery policy on an unloadable wallet file:
    * operator-supplied passphrase -> hard error. The operator owns a
      real passphrase; we never destroy a wallet they might recover.
    * synthesized (zero-config) passphrase -> the file is a demo wallet
      encrypted under a passphrase that can no longer be reproduced
      (pre-fix random node_id). Unrecoverable by construction, so the
      node must NOT die: move it aside with a timestamped `.orphaned-`
      suffix, log loudly, start fresh.
    """
    from core.tokenomics import Wallet
    wallet_path.parent.mkdir(parents=True, exist_ok=True)
    if wallet_path.exists():
        w = Wallet.load_from_file(str(wallet_path), passphrase=passphrase)
        if w is not None:
            return w
        if not synthesized:
            raise RuntimeError(
                f"wallet at {wallet_path} exists but could not be loaded "
                "(wrong passphrase or corrupt file). Check "
                "PLUGINFER_WALLET_PASSPHRASE, or move the file aside to "
                "generate a fresh wallet."
            )
        orphan = wallet_path.with_name(
            wallet_path.name + f".orphaned-{int(time.time())}")
        wallet_path.rename(orphan)
        logger.warning(
            "wallet at %s could not be decrypted with the synthesized "
            "passphrase (legacy random-node-id encryption). Moved it to "
            "%s and generating a fresh wallet.", wallet_path, orphan.name,
        )
    w = Wallet()
    if hasattr(w, "save_to_file"):
        ok = w.save_to_file(str(wallet_path), passphrase=passphrase)
        if not ok:
            logger.warning(
                "wallet save_to_file declined — running with in-memory key only "
                "(passphrase missing?). Set PLUGINFER_WALLET_PASSPHRASE."
            )
    return w


def _persistent_node_id(wallet_path: Path) -> str:
    """Node identity must survive restarts: the wallet passphrase is
    derived from it, and a provider's reputation/earnings hang off the
    same identity. Stored beside the wallet; created once."""
    id_path = wallet_path.parent / "node_id"
    try:
        existing = id_path.read_text("utf-8").strip()
        if existing:
            return existing
    except FileNotFoundError:
        pass
    except Exception:
        pass
    fresh = secrets.token_hex(4)
    try:
        id_path.parent.mkdir(parents=True, exist_ok=True)
        id_path.write_text(fresh, "utf-8")
    except Exception:
        logger.warning("could not persist node_id at %s — identity will "
                       "rotate on restart", id_path)
    return fresh


def _passphrase_from_env() -> Tuple[bytes, bool]:
    """Returns (passphrase, synthesized). `synthesized` distinguishes
    the zero-config demo passphrase from an operator-supplied one —
    recovery policy differs (see _load_or_create_wallet)."""
    env = os.environ.get("PLUGINFER_WALLET_PASSPHRASE", "")
    if env:
        return env.encode("utf-8"), False
    # Zero-config path: synthesise a stable per-machine passphrase from
    # the user's home dir + the PERSISTED node id. The node id used to
    # be random per run, which encrypted every wallet under a
    # passphrase that could never be reproduced — every restart lost
    # the earning identity. The persisted id closes that class.
    seed = (str(Path.home()) + ":" + os.environ.get("PLUGINFER_NODE_ID", "default"))
    return hashlib.sha256(seed.encode("utf-8")).digest(), True


# ---------------------------------------------------------------------------
# Local IP discovery (real-world IP) — wraps the seed_client utility
# but lets the caller override via env (e.g. inside Docker, or behind
# a corporate proxy).
# ---------------------------------------------------------------------------

def _local_ip(override: Optional[str] = None) -> str:
    if override:
        return override
    env = os.environ.get("PLUGINFER_PUBLIC_IP", "")
    if env:
        return env
    try:
        from infrastructure.seed_node.seed_client import discover_local_ip
        return discover_local_ip()
    except Exception:
        return "127.0.0.1"


def _should_adopt_observed(current_ip: str, observed_ip: str,
                           pinned: bool) -> bool:
    """Adopt the seed-observed public IP as our advertised address?

    `discover_local_ip()` returns the LAN address; behind NAT that is
    unroutable for WAN peers, so the mesh silently degrades to
    same-WiFi-only. The seed's REGISTER response tells us the source
    IP it actually saw (free STUN). Adopt it iff:
      * the operator did NOT pin an address (--bind-ip /
        PLUGINFER_PUBLIC_IP — explicit config always wins),
      * it differs from what we're advertising,
      * it is a real global address (never adopt loopback/RFC1918 —
        keeps localhost test meshes and LAN-seed meshes unchanged),
      * our current address is NOT itself global (a node already on a
        public IP shouldn't be re-pointed by a middlebox's view).
    """
    if pinned or not observed_ip or observed_ip == current_ip:
        return False
    import ipaddress
    try:
        observed = ipaddress.ip_address(observed_ip)
        current = ipaddress.ip_address(current_ip)
    except ValueError:
        return False
    return observed.is_global and not current.is_global


# ---------------------------------------------------------------------------
# Cross-node provider (B's gateway is a provider in A's auction).
#
# When A's auction picks this provider, execute() forwards the job to
# the remote node's `/v1/providers/open_jobs` poll path. Practically,
# both ends already use the §G6 browser-tab protocol; we just call it
# from a node-side client instead of a tab.
# ---------------------------------------------------------------------------

class _CrossNodeProvider:
    """Provider-shaped wrapper that forwards execute() to a remote
    Pluginfer node via HTTP. Quacks like a `core.providers.Provider`.

    The remote node, when running auto_mesh, has the
    `/v1/providers/register` + open_jobs + deliver endpoints from G6
    serving the same long-poll protocol the browser-tab provider
    uses."""

    # Backoff window after a connection failure. The cross-node abstains
    # from bidding while the timer is hot so the auction routes around
    # an unreachable peer instead of wedging the job. The next gossip
    # round (or a successful relay) lifts the suspension.
    UNREACHABLE_COOLDOWN_S: float = 30.0

    def __init__(self, *, peer_url: str, peer_pubkey: str,
                 my_pubkey: str, my_wallet,
                 relay_pool_getter: Optional[Any] = None,
                 punch_rpc_getter: Optional[Any] = None,
                 bandwidth_profile: Optional[Any] = None,
                 market_observer: Optional[Any] = None):
        self.peer_url = peer_url.rstrip("/")
        self.peer_pubkey = peer_pubkey
        # §HG6 — lambda returning the node's PunchRPC (or None). Third
        # rung of the reachability ladder: UDP hole-punch / TURN via
        # the seed, for peers unreachable by direct HTTP AND by every
        # HTTP relay candidate (symmetric NAT both sides).
        self._punch_rpc_getter = punch_rpc_getter
        self.provider_id = "cross-" + hashlib.sha256(
            (peer_url + peer_pubkey).encode("utf-8")
        ).hexdigest()[:16]
        self.privacy_grade = "public"
        self.hardware_class = "remote-mesh"
        self.tier = "staked"            # cross-node nodes are not browser tabs
        self.max_job_cost_usd = 0.0     # 0 = uncapped (handled by our auction)
        self.base_quality = 0.7
        self._my_pubkey = my_pubkey
        self._my_wallet = my_wallet
        # Bandwidth: when the operator sets PLUGINFER_DEFAULT_EGRESS_USD_PER_GB
        # the bid template adds expected egress to base price so the
        # provider isn't bidding below their wire cost.
        from core.bandwidth_pricing import BandwidthProfile
        self._bandwidth_profile = bandwidth_profile or BandwidthProfile()
        # Market observer: when present, the provider blends its
        # static-template bid with the observed clearing-price median
        # so fresh nodes track market without operator intervention.
        self._market_observer = market_observer
        # Filled lazily on first bid by GET /v1/hardware on the peer.
        # NVIDIA -> 50x score, AMD ROCm -> 40x, Apple MPS -> 5x,
        # Intel DirectML -> 3x, CPU -> 1x. Higher score = better
        # expected_quality, lower price-per-token, smaller eta_ms.
        self._peer_hw: Optional[Dict[str, Any]] = None
        self._peer_score: float = 1.0
        # Reachability tracking — direct HTTP works most of the time,
        # but on a symmetric-NAT WAN we may need to relay through a
        # third peer. Failure marks the cross-node "unreachable
        # direct" for UNREACHABLE_COOLDOWN_S; successful relay still
        # delivers the job. `last_path_used` is "direct" | "relay" |
        # "" — surfaced in evidence for audit.
        self._last_failure_unix: float = 0.0
        self._last_success_unix: float = 0.0
        self.last_path_used: str = ""
        # Lambda returning [(relay_url, relay_pubkey)] of OTHER peers
        # we know about. The cross-node uses these as relay candidates
        # when its direct path fails. Defaults to none — auction-only
        # fallback unless the discovery loop wires this.
        self._relay_pool_getter = relay_pool_getter

    def _fetch_peer_hw_once(self) -> None:
        if self._peer_hw is not None:
            return
        import urllib.error
        import urllib.request
        try:
            with urllib.request.urlopen(
                f"{self.peer_url}/v1/hardware", timeout=2.0,
            ) as r:
                self._peer_hw = json.loads(r.read().decode("utf-8"))
                self._peer_score = float(
                    self._peer_hw.get("performance_score", 1.0)
                )
                best = self._peer_hw.get("best_device") or {}
                # Map detected device class -> Pluginfer hardware_class
                # so PNIS receipts + leaderboard see the real tier.
                dtype = best.get("type", "unknown")
                self.hardware_class = {
                    "cuda": "consumer-gpu-high",
                    "rocm": "consumer-gpu-high",
                    "mps":  "consumer-gpu-mid",
                    "xpu":  "consumer-gpu-mid",
                    "directml": "consumer-gpu-low",
                    "cpu":  "consumer-cpu",
                }.get(dtype, "remote-mesh")
                # Quality + price scale with the peer's vendor tier.
                # 50x cuda → 0.85 quality; 1x cpu → 0.30 quality.
                # Cap [0.30, 0.95] so the auction never trusts a
                # cross-node infinitely.
                self.base_quality = max(0.30, min(0.95, 0.30 + 0.013 * self._peer_score))
        except (urllib.error.URLError, OSError, ValueError):
            # Peer doesn't run auto_mesh (browser tab, generic
            # provider) — keep the conservative template.
            self._peer_hw = {"best_device": {"type": "unknown"}}
            self._peer_score = 1.0

    def _is_in_cooldown(self) -> bool:
        """True when our last attempt failed and we're still inside the
        cooldown window. Bid returns None during this period so the
        auction routes around us."""
        if self._last_failure_unix == 0.0:
            return False
        # A subsequent success clears cooldown immediately.
        if self._last_success_unix >= self._last_failure_unix:
            return False
        return (time.time() - self._last_failure_unix) < self.UNREACHABLE_COOLDOWN_S

    def heartbeat_probe(self) -> bool:
        """Quick GET /healthz against the peer. Updates the success/
        failure timestamps as a side-effect so subsequent bids see the
        latest reachability state. Returns True iff the peer answered
        200 in under 2 seconds. Called by the discovery loop's
        liveness tick so dead peers fall out of the auction within
        ~one cycle instead of waiting for a job to fail first."""
        import urllib.error
        import urllib.request
        try:
            with urllib.request.urlopen(
                f"{self.peer_url}/healthz", timeout=2.0,
            ) as r:
                if r.status == 200:
                    self._last_success_unix = time.time()
                    return True
        except (urllib.error.URLError, OSError):
            pass
        self._last_failure_unix = time.time()
        return False

    # -- Provider interface ----------------------------------------------
    def bid(self, job) -> Optional[Any]:
        from core.bandwidth_pricing import (
            bandwidth_adjusted_price, estimate_egress_bytes,
        )
        from core.market_observer import blended_bid_price
        from core.providers import Bid
        if self._is_in_cooldown():
            return None
        self._fetch_peer_hw_once()
        approx_tokens = float((job.payload or {}).get("max_tokens", 200))
        # Price scales INVERSELY with the peer's score: a CUDA peer
        # bids 5x cheaper than a CPU peer because it can complete the
        # work in ~5x less wall-clock time at the same electricity
        # cost. Floors at $0.00005 / 1k tokens.
        per_1k_usd = max(0.00005, 0.0010 / max(1.0, self._peer_score / 10.0))
        static_price = per_1k_usd * (approx_tokens / 1000.0)
        # Market discovery: blend the static template with the rolling
        # clearing-price median for our bucket (hardware_class × kind).
        # Fresh nodes track market without operator config.
        market_median = None
        if self._market_observer is not None:
            try:
                market_median = self._market_observer.clearing_price(
                    self.hardware_class, getattr(job, "kind", "compute"),
                )
            except Exception:
                market_median = None
        price = blended_bid_price(
            static_template_price=static_price, market_price=market_median,
        )
        # Bandwidth: provider on a metered uplink bakes the expected
        # egress into the bid so they don't lose money on the wire.
        est_egress = estimate_egress_bytes(
            job.payload or {}, job_kind=getattr(job, "kind", ""),
        )
        price = bandwidth_adjusted_price(
            base_price_usd=price, profile=self._bandwidth_profile,
            est_egress_bytes=est_egress,
        )
        # ETA also scales inversely with score.
        eta_ms = max(200, int(15000.0 / max(1.0, self._peer_score / 10.0)))
        return Bid(
            provider_id=self.provider_id,
            price_usd=price,
            eta_ms=eta_ms,
            expected_quality=self.base_quality,
            privacy_grade=self.privacy_grade,
            evidence={
                "src": "cross-node",
                "peer_url": self.peer_url,
                "peer_score": self._peer_score,
                "peer_device": (self._peer_hw or {}).get(
                    "best_device", {}
                ).get("type", "unknown"),
                "hardware_class": self.hardware_class,
            },
        )

    def _post_chat_completions(
        self, url: str, body_bytes: bytes, timeout_s: float = 30.0,
    ) -> tuple[Optional[dict], Optional[dict], Optional[str]]:
        """Try `POST url` with the given JSON body. Returns
        (payload_dict, headers_dict, error_str_or_None)."""
        import urllib.error
        import urllib.request
        req = urllib.request.Request(
            url, data=body_bytes,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                payload = json.loads(r.read().decode("utf-8"))
                headers = dict(r.headers.items())
            return payload, headers, None
        except (urllib.error.URLError, OSError, ValueError) as e:
            return None, None, str(e)

    def execute(self, job, bid, *, on_delta=None) -> dict:
        """Submit the job to the peer's chat endpoint, wait, return the
        result in the same shape MeshGPUProvider / FlagshipProvider would.
        Synchronous because JobsService dispatches us via run_in_executor.

        Reachability strategy:
          1. Try the peer directly. Most LAN + public-IP cases work here.
          2. On connection failure, try each relay candidate from the
             pool getter (other reachable peers). Any one of them
             forwards to the target via its own /relay/{pubkey_hash}
             endpoint. Two symmetric-NAT-bound machines can swap work
             this way as long as at least one third peer is reachable
             from both.
          3. §HG6: UDP hole-punch / TURN relay via the seed
             (PunchRPC). Covers two machines behind symmetric NAT
             with NO mutually-reachable third peer — the seed brokers
             either a punched direct path or relays the datagrams.
          4. If every path fails, mark unreachable + start the
             cooldown. The auction will route around us until the
             next probe succeeds.
        """
        body_obj = {
            "model": "pluginfer-alpha",
            "messages": [{"role": "user", "content": (job.payload or {}).get("prompt", "")}],
            "max_tokens": int((job.payload or {}).get("max_tokens", 200)),
        }
        body = json.dumps(body_obj).encode("utf-8")

        # Direct.
        payload, headers, err = self._post_chat_completions(
            f"{self.peer_url}/v1/chat/completions", body,
        )
        path_used = "direct"
        if payload is None:
            # Relay fallback.
            relay_pool = (
                self._relay_pool_getter() if self._relay_pool_getter else []
            )
            peer_hash = hashlib.sha256(self.peer_pubkey.encode("utf-8")).hexdigest()
            for relay_url, _relay_pubkey in relay_pool:
                if relay_url.rstrip("/") == self.peer_url:
                    continue       # don't relay through the target
                payload, headers, err = self._post_chat_completions(
                    f"{relay_url.rstrip('/')}/relay/{peer_hash}/v1/chat/completions",
                    body, timeout_s=45.0,
                )
                if payload is not None:
                    path_used = "relay"
                    break

        if payload is None:
            # §HG6 — NAT-traversal rung: punch a UDP hole (or TURN-
            # relay through the seed) and run the same chat request
            # over PunchRPC. call_sync is safe here: execute() runs in
            # an executor thread, never on the event loop.
            rpc = (self._punch_rpc_getter()
                   if self._punch_rpc_getter else None)
            if rpc is not None:
                try:
                    status, hdrs, resp_body = rpc.call_sync(
                        self.peer_pubkey, body_obj, timeout_s=60.0,
                    )
                    if status == 200 and isinstance(resp_body, dict):
                        payload, headers = resp_body, hdrs
                        path_used = "punch"
                    else:
                        err = f"{err}; punch status={status}"
                except Exception as e:
                    err = f"{err}; punch: {e}"

        if payload is None:
            # Direct, relay AND punch failed. Mark unreachable + return
            # a structured failure for the auction.
            self._last_failure_unix = time.time()
            self.last_path_used = ""
            return {
                "status": "failed",
                "job_id": getattr(job, "job_id", ""),
                "reason": f"cross_node_unreachable: {err}",
            }

        self._last_success_unix = time.time()
        self.last_path_used = path_used
        text = ""
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            text = json.dumps(payload)
        out_bytes = text.encode("utf-8")
        h = hashlib.sha256(out_bytes).hexdigest()
        return {
            "status": "executed",
            "job_id": getattr(job, "job_id", ""),
            "result_bytes": base64.b64encode(out_bytes).decode("ascii"),
            "result_hash": h,
            "provider_sig": headers.get("X-Pluginfer-Provider", ""),
            "provider_pubkey_pem": self.peer_pubkey,
            "execution_ms": int(headers.get("X-Pluginfer-Execution-MS", "0") or 0),
            "model_id": "pluginfer-alpha",
            "cross_node_receipt_id": headers.get("X-Pluginfer-Receipt-ID", ""),
            "cross_node_path": path_used,
        }


# ---------------------------------------------------------------------------
# The node app — wraps the existing devserver + adds discovery endpoints
# ---------------------------------------------------------------------------

def build_node_app(*, my_pubkey: str, my_wallet, node_id: str):
    from api.devserver import build_devserver_app
    from api.jobs_service import JobsService
    from core.flagship import (
        register_alpha_flagship,
        spec_for_runtime,
    )
    from core.gossip_discovery import MembershipView
    from core.providers import Auction

    auction = Auction()
    svc = JobsService(auction=auction)
    # §RFC-3 — Budget-as-Contract on every node. With no envelopes
    # configured this is a pure pass-through (plus attribution
    # journalling); the moment an operator POSTs an envelope, the cap
    # binds fail-closed at submit. PLUGINFER_BUDGET_DIR persists
    # state + the chargeback journal across restarts.
    from governance.budget_ledger import BudgetLedger
    svc.budget = BudgetLedger(
        os.environ.get("PLUGINFER_BUDGET_DIR") or None)
    # The MONEY ledger — escrow, provider payouts, and the Pluginfer
    # commission (PLUGINFER_COMMISSION_RATE, default 10%) — was
    # previously never attached on this path, so a real node recorded
    # no commissions at all. Attached with persistence: every balance
    # and commission entry survives restarts under the node's state
    # dir (PLUGINFER_LEDGER_DIR overrides).
    from core.buyer_ledger import BuyerLedger
    _ledger_dir = (os.environ.get("PLUGINFER_LEDGER_DIR")
                   or os.environ.get("PLUGINFER_BUDGET_DIR")
                   or os.path.join(os.path.expanduser("~"), ".pluginfer",
                                   "ledger"))
    svc.ledger = BuyerLedger(_ledger_dir)
    # §HG16 — cash rails. A real gateway only when real creds exist
    # (PLUGINFER_STRIPE_SECRET_KEY); otherwise deposits refuse honestly
    # and withdrawals run the operator-payout queue (funds held with
    # correct accounting, closed only against a real payout reference).
    from core.payment_flows import PaymentFlows
    _pay_gateway = None
    try:
        from core.payments import StripeGateway
        _pay_gateway = StripeGateway()
    except Exception:
        _pay_gateway = None
    payment_flows = PaymentFlows(svc.ledger, gateway=_pay_gateway,
                                 state_dir=_ledger_dir)

    # Pick the best available runtime adapter. Falls back to honest
    # echo when no real backend is reachable — never mis-advertises.
    runner_fn, runtime_name, runtime_model_id = _resolve_alpha_runner()
    _RUNTIME_STATE["name"] = runtime_name
    _RUNTIME_STATE["model_id"] = runtime_model_id

    # Pre-warm the real runtime in the background so the cold model
    # load (minutes on consumer GPUs) happens at boot — never on the
    # first user request. Echo needs no warmup.
    if runtime_name not in ("echo", "alpha-echo"):
        import threading

        def _prewarm() -> None:
            try:
                t0 = time.monotonic()
                runner_fn("hi", {"max_tokens": 1})
                logger.info("runtime pre-warm complete in %.1fs "
                            "(first user request will be fast)",
                            time.monotonic() - t0)
            except Exception as e:
                logger.warning("runtime pre-warm failed (%s) — first "
                               "request will absorb the cold load", e)

        threading.Thread(target=_prewarm, daemon=True,
                         name="runtime-prewarm").start()

    # Our own compute serves as the local provider — peers can route to
    # us. The spec is derived from the RESOLVED runtime so receipts
    # stamp the model that actually answered, never a catalogue default.
    # PLUGINFER_BUYER_ONLY=1 makes this a pure CLIENT node: it registers
    # no local compute and only consumes mesh peers' compute — the
    # honest shape for a laptop/app that buys inference but never sells
    # it. (Also makes cross-node routing deterministic: with no local
    # bidder, a submitted job MUST run on a peer.)
    if os.environ.get("PLUGINFER_BUYER_ONLY", "0") != "1":
        register_alpha_flagship(
            jobs_service=svc,
            spec=spec_for_runtime(runtime_model_id, runtime_name),
            runner_fn=runner_fn,
            wallet=my_wallet,
        )
    else:
        logger.info("buyer_only mode: no local compute registered")

    # If a mesh-llm node (github.com/Mesh-LLM/mesh-llm) is serving on
    # this machine, its ENTIRE mesh becomes one more bidder in our
    # auction — we supply the economics + receipts, it supplies pooled
    # transport/inference. Opt-out: PLUGINFER_DISABLE_MESHLLM=1.
    try:
        from core.meshllm_provider import autodetect_meshllm
        meshllm = autodetect_meshllm(wallet=my_wallet)
        if meshllm is not None:
            auction.register(meshllm)
            _RUNTIME_STATE["meshllm"] = {
                "url": meshllm.base_url, "models": meshllm.models,
            }
    except Exception as e:
        logger.warning("mesh-llm autodetect skipped: %s", e)

    app = build_devserver_app(
        jobs_service=svc, title=f"Pluginfer Auto-Mesh Node {node_id}",
    )
    app.state.my_pubkey = my_pubkey
    app.state.my_node_id = node_id
    app.state.discovered_peers = []
    app.state.peer_providers: Dict[str, _CrossNodeProvider] = {}
    # Shared membership view drives both seed-bootstrap AND gossip
    # propagation. The /peers endpoint reads from it directly so a
    # peer learned via gossip (no seed contact) still shows up to
    # downstream gossip queries — that's how "find one, find all"
    # actually works in practice.
    app.state.view = MembershipView(own_pubkey=my_pubkey)
    app.state.runtime_name = runtime_name
    app.state.runtime_model_id = runtime_model_id

    # G-Heterogeneous: surface our hardware profile so peers can
    # bind their cross-node provider with REAL capability info,
    # not a flat template. HardwareDetector probes CUDA / MPS /
    # ROCm / Intel-XPU / DirectML / CPU on every supported OS.
    app.state._hw_profile_cache: Optional[Dict[str, Any]] = None

    from fastapi import Request
    from fastapi.responses import JSONResponse

    @app.post("/v1/quote")
    async def v1_quote(request: Request):
        """§RFC-3 quote-before-run: the auction prices the job without
        executing it. The sync auction rides the executor to keep the
        loop responsive, same as submit()."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        payload = body.get("payload") or {
            "prompt": body.get("prompt", ""),
            "model": body.get("model", "pluginfer-alpha"),
            "max_tokens": int(body.get("max_tokens", 200)),
        }
        quote = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: svc.quote(
                kind=str(body.get("kind", "inference")),
                payload=payload,
                cost_ceiling_usd=float(body.get("cost_ceiling_usd", 0.10)),
                latency_ceiling_ms=int(body.get("latency_ceiling_ms",
                                                30_000)),
                privacy_class=str(body.get("privacy_class", "public")),
                quality_floor=float(body.get("quality_floor", 0.7)),
            ),
        )
        return quote

    @app.get("/v1/budget/report")
    async def v1_budget_report(prefix: str = "", since_unix: float = 0.0):
        if svc.budget is None:
            return {"error": "budget ledger not attached"}
        return svc.budget.report(prefix=prefix, since_unix=since_unix)

    @app.post("/v1/budget/envelopes")
    async def v1_budget_set_envelope(request: Request):
        if svc.budget is None:
            return JSONResponse(status_code=503, content={
                "error": "budget ledger not attached"})
        try:
            body = await request.json()
            env = svc.budget.set_envelope(
                str(body["path"]), float(body["cap_usd"]),
                str(body.get("period", "month")))
        except (KeyError, ValueError, TypeError) as e:
            return JSONResponse(status_code=400,
                                content={"error": str(e)})
        return env.to_dict()

    @app.get("/v1/ledger/treasury")
    async def v1_ledger_treasury(limit: int = 100):
        """The commission book — how much Pluginfer has earned, from
        which jobs and buyers. This is the operator revenue view the
        audit found missing."""
        if svc.ledger is None:
            return JSONResponse(status_code=503, content={
                "error": "money ledger not attached"})
        return _stamp_economics(svc.ledger.treasury_report(limit=limit))

    @app.get("/v1/ledger/verify")
    async def v1_ledger_verify():
        """Anyone may audit this node's money ledger: recomputes every
        balance from its full entry history and reports the snapshot
        integrity status. ok=false blocks withdrawals automatically."""
        if svc.ledger is None:
            return JSONResponse(status_code=503, content={
                "error": "money ledger not attached"})
        return _stamp_economics(svc.ledger.verify_balances())

    @app.get("/v1/ledger/wallets/{wallet_id}")
    async def v1_ledger_wallet(wallet_id: str):
        if svc.ledger is None:
            return JSONResponse(status_code=503, content={
                "error": "money ledger not attached"})
        w = svc.ledger.get_wallet(wallet_id)
        if w is None:
            return JSONResponse(status_code=404, content={
                "error": f"no wallet {wallet_id!r}"})
        return _stamp_economics(w.to_public())

    # §HG16 — cash rails. Mutating money endpoints are gated by
    # PLUGINFER_NODE_ADMIN_KEY when it is set (recommended for any
    # non-local deployment); with no key set, the node is presumed a
    # local dev instance.
    def _money_denied(request: "Request"):
        want = os.environ.get("PLUGINFER_NODE_ADMIN_KEY", "")
        if want and request.headers.get("x-admin-key") != want:
            return JSONResponse(status_code=401, content={
                "error": "X-Admin-Key required for money operations"})
        return None

    # Economics mode — TESTNET by default, and every money surface says
    # so. The promise this encodes (stated up-front so it can never be
    # walked back quietly): testnet balances are real, persistent
    # accounting but are NOT redeemable for cash; they are preserved,
    # and any recognition at mainnet will be announced BEFORE it
    # happens. Payouts only ever come from real buyer payments —
    # never from a treasury subsidy. Flipping to mainnet is an
    # explicit operator act (PLUGINFER_ECONOMICS_MODE=mainnet) that
    # requires a configured real payment gateway.
    def _economics_mode() -> str:
        mode = os.environ.get("PLUGINFER_ECONOMICS_MODE",
                              "testnet").lower()
        return mode if mode in ("testnet", "mainnet") else "testnet"

    _TESTNET_NOTICE = (
        "TESTNET economics: balances are real, persistent accounting "
        "but are NOT redeemable for cash. They are preserved; any "
        "recognition at mainnet will be announced before it happens. "
        "Payouts come only from real buyer payments — never treasury "
        "subsidies.")

    def _stamp_economics(payload: dict) -> dict:
        payload["economics_mode"] = _economics_mode()
        if _economics_mode() == "testnet":
            payload["notice"] = _TESTNET_NOTICE
        return payload

    def _cash_denied():
        """Real cash movement is refused outright while in testnet —
        a mis-set Stripe key must not be able to charge anyone. And
        mainnet REFUSES to move money with no admin key configured:
        an open, unauthenticated cash endpoint is not a dev
        convenience once the money is real."""
        if _economics_mode() == "mainnet" and not os.environ.get(
                "PLUGINFER_NODE_ADMIN_KEY"):
            return JSONResponse(status_code=403, content=_stamp_economics({
                "error": "economics_mode=mainnet requires "
                         "PLUGINFER_NODE_ADMIN_KEY to be set — refusing "
                         "to expose unauthenticated money operations"}))
        if _economics_mode() == "testnet":
            return JSONResponse(status_code=403, content=_stamp_economics({
                "error": "economics_mode=testnet — real deposits and "
                         "withdrawal payouts are disabled. Set "
                         "PLUGINFER_ECONOMICS_MODE=mainnet (with a "
                         "configured payment gateway) to enable."}))
        return None

    @app.post("/v1/testnet/faucet")
    async def v1_testnet_faucet(request: Request):
        """Self-serve starter credit so anyone who joins can try the
        mesh as a buyer — TESTNET ONLY, once per wallet. Deliberately
        NOT behind the admin key: automation for newcomers is the
        point, idempotency is the abuse guard, and in mainnet mode this
        endpoint refuses outright (operator-minted mainnet balance
        would be a treasury subsidy, which this project never does)."""
        if _economics_mode() != "testnet":
            return JSONResponse(status_code=403, content=_stamp_economics({
                "error": "faucet exists only under testnet economics"}))
        if svc.ledger is None:
            return JSONResponse(status_code=503, content={
                "error": "money ledger not attached"})
        from decimal import Decimal as _D
        from core.buyer_ledger import FaucetAlreadyGranted
        try:
            body = await request.json()
            wallet_id = str(body["wallet_id"]).strip()
            if not wallet_id:
                raise ValueError("wallet_id required")
            amount = _D(os.environ.get(
                "PLUGINFER_TESTNET_FAUCET_USD", "25"))
            w = svc.ledger.faucet_grant(wallet_id, amount)
        except FaucetAlreadyGranted as e:
            return JSONResponse(status_code=409,
                                content=_stamp_economics({"error": str(e)}))
        except (KeyError, ValueError, ArithmeticError) as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        return _stamp_economics({
            "wallet_id": wallet_id,
            "granted_usd": str(amount),
            "available_usd": str(w.available_usd),
        })

    @app.post("/v1/payments/deposit")
    async def v1_payments_deposit(request: Request):
        deny = _money_denied(request) or _cash_denied()
        if deny is not None:
            return deny
        from decimal import Decimal as _D
        from core.payment_flows import PaymentsNotConfigured
        try:
            body = await request.json()
            rec = payment_flows.deposit(
                wallet_id=str(body["wallet_id"]),
                amount_usd=_D(str(body["amount_usd"])),
                customer_id=str(body["customer_id"]),
                idempotency_key=body.get("idempotency_key"),
            )
        except PaymentsNotConfigured as e:
            return JSONResponse(status_code=503, content={"error": str(e)})
        except (KeyError, ValueError, ArithmeticError) as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except RuntimeError as e:
            return JSONResponse(status_code=402, content={"error": str(e)})
        return rec.to_public()

    @app.post("/v1/payments/withdraw")
    async def v1_payments_withdraw(request: Request):
        deny = _money_denied(request) or _cash_denied()
        if deny is not None:
            return deny
        from decimal import Decimal as _D
        from core.buyer_ledger import InsufficientFunds
        try:
            body = await request.json()
            rec = payment_flows.request_withdrawal(
                wallet_id=str(body["wallet_id"]),
                amount_usd=_D(str(body["amount_usd"])),
                destination=str(body["destination"]),
            )
        except InsufficientFunds as e:
            return JSONResponse(status_code=402, content={"error": str(e)})
        except (KeyError, ValueError, ArithmeticError) as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        return rec.to_public()

    @app.get("/v1/payments/withdrawals")
    async def v1_payments_withdrawals(wallet_id: str = ""):
        return _stamp_economics({
            "withdrawals":
                payment_flows.withdrawals(wallet_id=wallet_id or None)})

    @app.post("/v1/payments/withdrawals/{withdrawal_id}/complete")
    async def v1_payments_withdraw_complete(withdrawal_id: str,
                                            request: Request):
        deny = _money_denied(request)
        if deny is not None:
            return deny
        from core.payment_flows import UnknownWithdrawal
        try:
            body = await request.json()
            rec = payment_flows.complete_withdrawal(
                withdrawal_id,
                payout_reference=str(body["payout_reference"]))
        except UnknownWithdrawal:
            return JSONResponse(status_code=404, content={
                "error": f"no withdrawal {withdrawal_id!r}"})
        except (KeyError, ValueError) as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except RuntimeError as e:
            return JSONResponse(status_code=409, content={"error": str(e)})
        return rec.to_public()

    @app.post("/v1/payments/withdrawals/{withdrawal_id}/cancel")
    async def v1_payments_withdraw_cancel(withdrawal_id: str,
                                          request: Request):
        deny = _money_denied(request)
        if deny is not None:
            return deny
        from core.payment_flows import UnknownWithdrawal
        try:
            rec = payment_flows.cancel_withdrawal(withdrawal_id)
        except UnknownWithdrawal:
            return JSONResponse(status_code=404, content={
                "error": f"no withdrawal {withdrawal_id!r}"})
        except RuntimeError as e:
            return JSONResponse(status_code=409, content={"error": str(e)})
        return rec.to_public()

    @app.get("/peers")
    def peers_endpoint(
        from_pubkey: str = "", from_ip: str = "",
        from_port: int = 0, from_version: str = "1.0.0",
    ) -> Dict[str, Any]:
        # Union of seed-fetched + gossip-learned peers, deduped by
        # pubkey via the shared MembershipView. Gossip can now
        # propagate THIS node's knowledge of peer X to peer Y who
        # never asked the seed about X.
        #
        # When the caller carries from_pubkey/from_ip/from_port,
        # we fold them into our view + bind them as a CrossNodeProvider.
        # This closes the bidirectional-discovery loop: a node that
        # bootstraps via us announces itself in the same round-trip
        # it uses to pull our view. No separate /announce endpoint.
        if from_pubkey and from_ip and from_port and from_pubkey != my_pubkey:
            from core.gossip_discovery import PeerEntry
            entry = PeerEntry(
                pubkey_pem=from_pubkey, ip=from_ip,
                port=int(from_port), node_version=from_version,
            )
            if app.state.view.add_or_update(entry):
                _bind_peer_if_new(
                    app, svc, my_pubkey=my_pubkey, my_wallet=my_wallet,
                    pubkey_pem=from_pubkey, ip=from_ip, port=int(from_port),
                )
        view_wire = app.state.view.to_wire_list()
        return {
            "me": {"node_id": node_id, "pubkey": my_pubkey},
            "discovered_peers": view_wire or app.state.discovered_peers,
            "view_size": len(app.state.view),
            "auction_size": len(svc.auction.providers),
            "registered_cross_nodes": list(app.state.peer_providers.keys()),
            "runtime": {
                "name": runtime_name,
                "model_id": runtime_model_id,
                "is_echo": runtime_name == "alpha-echo",
            },
        }

    # Mesh-native relay: any auto_mesh node can forward a chat
    # completion to another known peer on behalf of a third node that
    # can't reach the target directly. This is what makes the mesh
    # actually work across symmetric NAT: A→C→B becomes a usable path
    # whenever there exists a C reachable from both A and B. No
    # specialized relay infrastructure required — every node is a
    # candidate. Innovation lead: §A24 "Permissionless HTTP relay
    # over an authenticated mesh."
    from fastapi import HTTPException, Request as _FastReq
    @app.post("/relay/{peer_hash}/v1/chat/completions")
    async def relay_chat(peer_hash: str, request: _FastReq) -> Any:
        import urllib.error
        import urllib.request
        # Look up the peer by SHA-256(pubkey_pem) — keeps URLs
        # short + opaque without exposing keys in path params.
        target_url = None
        for pk, prov in app.state.peer_providers.items():
            if hashlib.sha256(pk.encode("utf-8")).hexdigest() == peer_hash:
                target_url = prov.peer_url
                break
        if target_url is None:
            raise HTTPException(404, "peer not in our membership view")
        body = await request.body()
        req = urllib.request.Request(
            f"{target_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=45.0) as r:
                payload_bytes = r.read()
                upstream_headers = dict(r.headers.items())
        except (urllib.error.URLError, OSError) as e:
            raise HTTPException(502, f"relay_upstream_unreachable: {e}")
        from fastapi.responses import Response
        forward_headers = {
            k: v for k, v in upstream_headers.items()
            if k.lower().startswith("x-pluginfer-")
        }
        forward_headers["X-Pluginfer-Relay"] = my_pubkey.splitlines()[1][:32]
        return Response(
            content=payload_bytes,
            media_type="application/json",
            headers=forward_headers,
        )

    @app.get("/v1/hardware")
    def hardware_endpoint() -> Dict[str, Any]:
        """Each node advertises its compute profile — vendor, score,
        device list — so peers bind a hardware-aware Provider rather
        than a flat 'remote-mesh' template."""
        if app.state._hw_profile_cache is None:
            try:
                from core.hardware_detector import HardwareDetector
                d = HardwareDetector()
                devices = d.detect_all_devices()
                best = d.get_best_device()
                app.state._hw_profile_cache = {
                    "node_id": node_id,
                    "pubkey": my_pubkey,
                    "best_device": best,
                    "devices": devices,
                    "performance_score": d.get_performance_score(),
                    "runtime": {
                        "name": runtime_name,
                        "model_id": runtime_model_id,
                        "is_echo": runtime_name == "alpha-echo",
                    },
                }
            except Exception as e:
                app.state._hw_profile_cache = {
                    "node_id": node_id, "pubkey": my_pubkey,
                    "best_device": {"type": "unknown"},
                    "devices": [],
                    "performance_score": 1.0,
                    "error": str(e),
                    "runtime": {
                        "name": runtime_name,
                        "model_id": runtime_model_id,
                        "is_echo": runtime_name == "alpha-echo",
                    },
                }
        # host_guard is live state (pressure/accepting flip at runtime),
        # so it rides OUTSIDE the cached profile: peers and audits see
        # whether this node is currently shedding work and under what
        # memory cap it runs.
        out = dict(app.state._hw_profile_cache)
        try:
            import host_guard
            out["host_guard"] = host_guard.status()
        except ImportError:
            pass
        # §HG6 — advertise NAT-traversal capability so peers and
        # audits can see whether this node is punch-reachable.
        pc = getattr(app.state, "punch_client", None)
        out["nat_traversal"] = {
            "punch_active": pc is not None,
            "external_udp_addr": list(pc.external_addr) if (
                pc is not None and pc.external_addr) else None,
        }
        return out

    return app, svc


# ---------------------------------------------------------------------------
# Discovery loop — runs alongside uvicorn
# ---------------------------------------------------------------------------

def _bind_peer_if_new(app, svc, *, my_pubkey: str, my_wallet,
                      pubkey_pem: str, ip: str, port: int) -> bool:
    """Idempotent bind: register a CrossNodeProvider for this peer
    on our auction if we haven't seen them before. Returns True iff
    this was a first bind. Shared between seed-poll AND gossip
    on_new_peer so the two paths converge on identical auction
    state regardless of which one saw the peer first."""
    if not pubkey_pem or pubkey_pem == my_pubkey:
        return False
    if pubkey_pem in app.state.peer_providers:
        return False
    from core.gossip_discovery import peer_base_url
    peer_url = peer_base_url(ip, port)

    # Relay-pool getter: returns every OTHER known peer's (url, pubkey).
    # The cross-node uses this list as relay candidates when its
    # direct path to the target fails. Closes the symmetric-NAT
    # "two strangers behind firewalls can't reach each other" gap
    # without any specialised relay infrastructure.
    target_pubkey = pubkey_pem
    def _relay_pool_for_this_target() -> List[tuple]:
        out: List[tuple] = []
        for pk, prov in app.state.peer_providers.items():
            if pk == target_pubkey:
                continue
            if prov._is_in_cooldown():
                continue
            out.append((prov.peer_url, pk))
        return out

    prov = _CrossNodeProvider(
        peer_url=peer_url, peer_pubkey=pubkey_pem,
        my_pubkey=my_pubkey, my_wallet=my_wallet,
        relay_pool_getter=_relay_pool_for_this_target,
        punch_rpc_getter=lambda: getattr(app.state, "punch_rpc", None),
    )
    app.state.peer_providers[pubkey_pem] = prov
    svc.auction.register(prov)
    logger.info("cross_node_provider_added: %s", peer_url)
    return True


async def discovery_loop(
    app, svc,
    *,
    seed_host: str, seed_port: int,
    my_pubkey: str, my_wallet,
    my_ip: str, my_port: int,
    bootstrap_peers: Optional[List[str]] = None,
    ip_pinned: bool = False,
):
    from core.gossip_discovery import (
        PeerEntry,
        _http_get_json,
        gossip_round,
    )
    from infrastructure.seed_node.seed_client import (
        SeedAddress,
        fetch_peers_async,
        register_async,
    )
    seed = SeedAddress(host=seed_host, port=seed_port)
    sign_fn = lambda msg: my_wallet.sign(msg)
    next_seed_reg = 0.0
    next_peer_poll = 0.0
    next_gossip = time.monotonic() + GOSSIP_TICK_INTERVAL_S
    next_heartbeat = time.monotonic() + HEARTBEAT_TICK_INTERVAL_S

    # Bootstrap via known-good peer URLs when the seed is unreachable
    # or the operator wants a closed mesh. For each bootstrap address
    # we hit /peers WITH our own identity attached so the bootstrap
    # peer folds us into its view too. After this, normal gossip
    # rounds carry the membership both ways.
    from core.gossip_discovery import _announce_query_string
    for bp in bootstrap_peers or []:
        try:
            host_port = bp.split(":")
            if len(host_port) != 2:
                continue
            qs = _announce_query_string(
                my_pubkey, my_ip, my_port, NODE_VERSION,
            )
            from core.gossip_discovery import peer_base_url
            bp_url = (peer_base_url(host_port[0], int(host_port[1]))
                      + f"/peers{qs}")
            payload = await asyncio.get_running_loop().run_in_executor(
                None, lambda u=bp_url: _http_get_json(u, timeout=3.0),
            )
            if not payload:
                logger.warning("gossip_bootstrap unreachable: %s", bp)
                continue
            me_block = payload.get("me") or {}
            if me_block.get("pubkey"):
                entry = PeerEntry(
                    pubkey_pem=str(me_block["pubkey"]),
                    ip=host_port[0], port=int(host_port[1]),
                )
                app.state.view.add_or_update(entry)
                _bind_peer_if_new(
                    app, svc, my_pubkey=my_pubkey, my_wallet=my_wallet,
                    pubkey_pem=entry.pubkey_pem,
                    ip=entry.ip, port=entry.port,
                )
                logger.info("gossip_bootstrap: bound %s", bp)
        except Exception as e:
            logger.warning("gossip_bootstrap failed for %s: %s", bp, e)

    def _on_gossip_new(entry):
        # Each newly-discovered peer (via ANY path) gets bound to
        # our auction. This is the "find one, find all" payoff:
        # the moment we learn about a transitive peer through
        # someone else's /peers, we can route work to them.
        _bind_peer_if_new(
            app, svc,
            my_pubkey=my_pubkey, my_wallet=my_wallet,
            pubkey_pem=entry.pubkey_pem, ip=entry.ip, port=entry.port,
        )

    while True:
        now = time.monotonic()
        if now >= next_seed_reg:
            try:
                resp = await register_async(
                    seed,
                    pubkey_pem=my_pubkey,
                    sign_fn=sign_fn,
                    ip=my_ip, port=my_port,
                    node_version=NODE_VERSION,
                )
                if resp:
                    logger.info("seed_register: %s", resp.get("status"))
                    # WAN self-correction: behind NAT we self-reported
                    # a LAN address no remote peer can dial. The seed
                    # tells us the source IP it actually saw; adopt it
                    # and re-register (freshly signed) so the peer
                    # table carries an address that works across the
                    # web, not just inside this WiFi.
                    observed = str(resp.get("observed_ip") or "")
                    if _should_adopt_observed(my_ip, observed, ip_pinned):
                        logger.warning(
                            "NAT detected: seed observed us at %s while "
                            "we advertised %s — adopting the public "
                            "address and re-registering.",
                            observed, my_ip,
                        )
                        my_ip = observed
                        await register_async(
                            seed,
                            pubkey_pem=my_pubkey,
                            sign_fn=sign_fn,
                            ip=my_ip, port=my_port,
                            node_version=NODE_VERSION,
                        )
            except Exception as e:
                logger.warning("seed_register_failed: %s", e)
            next_seed_reg = now + SEED_REGISTER_INTERVAL_S
        if now >= next_peer_poll:
            try:
                peers = await fetch_peers_async(seed)
            except Exception as e:
                logger.warning("fetch_peers_failed: %s", e)
                peers = []
            fresh = [
                p for p in peers
                if p.get("pubkey_pem") and p["pubkey_pem"] != my_pubkey
            ]
            app.state.discovered_peers = fresh
            # Seed-fetched peers ALSO get folded into the gossip
            # membership view so the next gossip round can
            # propagate them to peers who never asked the seed.
            for peer in fresh:
                try:
                    entry = PeerEntry(
                        pubkey_pem=peer["pubkey_pem"],
                        ip=peer["ip"], port=int(peer["port"]),
                        node_version=peer.get("node_version") or "1.0.0",
                    )
                    if app.state.view.add_or_update(entry):
                        _bind_peer_if_new(
                            app, svc, my_pubkey=my_pubkey,
                            my_wallet=my_wallet,
                            pubkey_pem=entry.pubkey_pem,
                            ip=entry.ip, port=entry.port,
                        )
                except (KeyError, ValueError, TypeError):
                    continue
            next_peer_poll = now + PEER_POLL_INTERVAL_S
        if now >= next_gossip:
            try:
                await gossip_round(
                    app.state.view, on_new_peer=_on_gossip_new,
                    own_ip=my_ip, own_port=my_port,
                    own_node_version=NODE_VERSION,
                )
            except Exception as e:
                logger.warning("gossip_round_failed: %s", e)
            next_gossip = now + GOSSIP_TICK_INTERVAL_S
        if now >= next_heartbeat:
            # Liveness sweep — probe every registered cross-node peer
            # in parallel. Dead peers' next bid abstains automatically
            # via _is_in_cooldown; live peers' success timestamps
            # update so they keep bidding. Runs the probes in the
            # default thread pool so a slow peer doesn't stall the
            # discovery loop.
            providers_snapshot = list(app.state.peer_providers.values())
            if providers_snapshot:
                loop = asyncio.get_running_loop()
                await asyncio.gather(*[
                    loop.run_in_executor(None, prov.heartbeat_probe)
                    for prov in providers_snapshot
                ], return_exceptions=True)
            next_heartbeat = now + HEARTBEAT_TICK_INTERVAL_S
        await asyncio.sleep(0.5)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _run(args) -> None:
    import uvicorn
    wallet_path = Path(args.wallet_path).expanduser()
    # Identity FIRST, then passphrase: the synthesized passphrase is
    # derived from the node id, so the id must be stable across runs
    # (persisted beside the wallet) or the wallet can never be reloaded.
    node_id = (args.node_id or os.environ.get("PLUGINFER_NODE_ID")
               or _persistent_node_id(wallet_path))
    os.environ["PLUGINFER_NODE_ID"] = node_id

    passphrase, synthesized = _passphrase_from_env()
    wallet = _load_or_create_wallet(wallet_path, passphrase, synthesized)
    my_pubkey = wallet.public_key_pem

    my_port = int(args.node_port) if int(args.node_port) > 0 else _free_port()
    my_ip = _local_ip(args.bind_ip)
    # Advertised port MAY differ from the bound port: behind a tunnel
    # (ngrok/cloudflared TCP) or a port-forward, peers must reach us at
    # the PUBLIC port while uvicorn binds the LOCAL one. Defaults to the
    # bound port, so nothing changes for the common same-port case.
    advertised_port = int(os.environ.get("PLUGINFER_PUBLIC_PORT") or my_port)

    app, svc = build_node_app(
        my_pubkey=my_pubkey, my_wallet=wallet, node_id=node_id,
    )

    logger.info(
        "auto-mesh booting node_id=%s on %s:%d (seed=%s:%d, pubkey=%s)",
        node_id, my_ip, my_port, args.seed_host, args.seed_port,
        my_pubkey.splitlines()[1][:32] + "..",
    )
    config = uvicorn.Config(app, host="0.0.0.0", port=my_port, log_level="warning")
    server = uvicorn.Server(config)

    bootstrap_peers = list(args.gossip_bootstrap or [])
    env_bootstrap = os.environ.get("PLUGINFER_GOSSIP_BOOTSTRAP_PEER", "").strip()
    if env_bootstrap:
        bootstrap_peers.extend(p.strip() for p in env_bootstrap.split(",") if p.strip())

    # §HG6 — NAT traversal bring-up. One punched UDP socket per node,
    # registered with the seed's UDP punch server (seed_main serves
    # TCP registry + UDP punch on the SAME port). Inbound punched jobs
    # loop back into our own local gateway so they run the exact same
    # auction/receipt pipeline as HTTP jobs. Non-fatal by design: a
    # TCP-only seed simply never answers REGISTER_UDP and the
    # reachability ladder stops at HTTP relay.
    app.state.punch_rpc = None
    if os.environ.get("PLUGINFER_ENABLE_PUNCH", "1") != "0":
        try:
            from core.peer_connect import (
                PeerConnectClient,
                SeedAddress as _PunchSeed,
            )
            from core.punch_rpc import PunchRPC

            punch_client = await PeerConnectClient.start(
                seeds=[_PunchSeed(host=args.seed_host,
                                  port=args.seed_port)],
                local_pubkey_pem=my_pubkey,
                sign=lambda m: wallet.sign(m),
            )

            async def _serve_punched_chat(body: Dict[str, Any]):
                def _post():
                    import urllib.request
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{my_port}/v1/chat/completions",
                        data=json.dumps(body).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=90.0) as r:
                        return (r.status, dict(r.headers.items()),
                                json.loads(r.read().decode("utf-8")))
                return await asyncio.get_running_loop().run_in_executor(
                    None, _post)

            app.state.punch_rpc = PunchRPC(
                punch_client, _serve_punched_chat,
                my_pubkey_pem=my_pubkey,
            )
            app.state.punch_client = punch_client
            logger.info(
                "nat_traversal_up: punch socket registered with "
                "%s:%d (udp)", args.seed_host, args.seed_port,
            )
        except Exception as e:
            logger.warning(
                "nat_traversal_unavailable (%s: %s) — peers behind "
                "symmetric NAT reach this node only via HTTP relay.",
                type(e).__name__, e,
            )

    disc_task = asyncio.create_task(discovery_loop(
        app, svc,
        seed_host=args.seed_host, seed_port=args.seed_port,
        my_pubkey=my_pubkey, my_wallet=wallet,
        my_ip=my_ip, my_port=advertised_port,
        bootstrap_peers=bootstrap_peers,
        # Explicit operator address always wins over NAT self-correction.
        ip_pinned=bool(
            args.bind_ip or os.environ.get("PLUGINFER_PUBLIC_IP", "")
            or os.environ.get("PLUGINFER_PUBLIC_PORT", "")),
    ))
    try:
        await server.serve()
    finally:
        disc_task.cancel()
        try:
            await disc_task
        except (asyncio.CancelledError, Exception):
            pass
        pc = getattr(app.state, "punch_client", None)
        if pc is not None:
            try:
                pc.close()
            except Exception:
                pass


NODE_BANNER = r"""
  ____  _             _        __
 |  _ \| |_   _  __ _(_)_ __  / _| ___ _ __
 | |_) | | | | |/ _` | | '_ \| |_ / _ \ '__|
 |  __/| | |_| | (_| | | | | |  _|  __/ |
 |_|   |_|\__,_|\__, |_|_| |_|_|  \___|_|
                |___/
 Pluginfer node - auction | receipts | mesh
"""


def main() -> None:
    print(NODE_BANNER)
    # Host protection before anything heavy loads (torch/BLAS import
    # inside the runtime adapters): job-object memory cap, below-normal
    # priority, thread caps, memory watchdog. A node must never hang
    # its operator's machine.
    import host_guard
    host_guard.install("auto_mesh")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-host", default=os.environ.get(
        "PLUGINFER_SEED_HOST", "127.0.0.1"))
    ap.add_argument("--seed-port", type=int, default=int(os.environ.get(
        "PLUGINFER_SEED_PORT", "9000")))
    ap.add_argument("--node-port", type=int, default=int(os.environ.get(
        "PLUGINFER_NODE_PORT", "0")))   # 0 -> auto-pick free port
    ap.add_argument("--bind-ip", default="",
                    help="Override the IP we register with the seed.")
    ap.add_argument("--node-id", default="")
    ap.add_argument("--wallet-path",
                    default=str(Path.home() / ".pluginfer" / "auto_mesh_wallet.pem"))
    ap.add_argument(
        "--gossip-bootstrap", action="append", default=[],
        help="HOST:PORT of a known peer to bootstrap from when the seed "
             "is unreachable. Repeatable. Can also be set via "
             "PLUGINFER_GOSSIP_BOOTSTRAP_PEER (comma-separated).",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=os.environ.get("PLUGINFER_LOG_LEVEL", "INFO"),
        format="[auto_mesh] %(levelname)s %(message)s",
    )
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
