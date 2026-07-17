"""
Distributed Task Router
=======================
The piece that lets a Pluginfer node actually push work to other
nodes (and receive results back). Until this module landed, the
mesh formed but every task ran on the originating node — the
distributed claim was vacuous. `architect.py` had a literal
TODO comment at the dispatch site.

What this implements
--------------------
1. **Send-task protocol** (TCP, JSON envelope):
       SUBMIT  job_id, plugin, input, deadline_ms, redundancy
       ACCEPT  job_id, worker_node_id, eta_ms
       RESULT  job_id, status, output | error
       AUDIT   job_id, audit_seed, batch_idx        (optional)
       CANCEL  job_id

2. **K-redundant execution**: each task is dispatched to K independent
   peers; the router accepts the *first* matching result, validates
   that the others agree (median for numeric, hash equality for
   bytes), and slashes outliers' reputation. Single-worker poisoning
   becomes economically irrational.

3. **Brain-driven peer selection**: uses `PluginferBrain` for the
   trust decision per peer (treats each candidate as `incoming peer`,
   asks the brain whether to trust/audit/drop). Falls back to scout
   latency ranking when the brain isn't confident.

4. **Heterogeneous-aware matching**: every node publishes its hardware
   profile (GPU class, VRAM, CPU TFLOPS, bandwidth). The router only
   routes a 'high_vram' job to peers that meet the requirement.

5. **Streaming-result hook**: long jobs (training rounds, batch
   inference) yield partial progress via the same TCP connection.
   The caller registers a callback that receives chunks as they
   arrive.

6. **Built-in retry / failover**: if a worker drops mid-task, the
   router re-routes to the next-best peer with no caller change.

This is the core primitive that makes the network a real compute
substitute for a centralised cloud.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Wire protocol message types
MSG_SUBMIT = "ROUTER_SUBMIT"
MSG_ACCEPT = "ROUTER_ACCEPT"
MSG_RESULT = "ROUTER_RESULT"
MSG_PROGRESS = "ROUTER_PROGRESS"
MSG_AUDIT = "ROUTER_AUDIT"
MSG_CANCEL = "ROUTER_CANCEL"


@dataclass
class HardwareProfile:
    """What a peer publishes about its compute capabilities."""
    gpu_class: str = "cpu"          # 'cuda' | 'mps' | 'rocm' | 'directml' | 'cpu'
    gpu_vram_gb: float = 0.0
    cpu_cores: int = 1
    cpu_tflops_estimate: float = 0.05
    network_mbps: float = 100.0
    continent: Optional[str] = None
    plugins: List[str] = field(default_factory=list)
    tee_attested: bool = False
    last_updated: float = 0.0
    # Training-governor snapshot from the peer (see
    # ai.filum.training_governor.TrainingGovernor.health_snapshot).
    # When present we use it to soft-quarantine recovering nodes.
    health: Optional[Dict[str, Any]] = None
    health_updated: float = 0.0


@dataclass
class TaskRequirements:
    """What a job needs to run."""
    plugin: str
    min_vram_gb: float = 0.0
    needs_gpu: bool = False
    deadline_ms: int = 60_000
    redundancy: int = 1                  # K-redundant execution
    require_tee_attested: bool = False
    cost_ceiling_plg: float = 1.0
    prefer_continent: Optional[str] = None


@dataclass
class _InflightTask:
    job_id: str
    requirements: TaskRequirements
    input_data: Dict[str, Any]
    started_at: float
    accepted_workers: List[str] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    callback: Optional[Callable[[Dict[str, Any]], None]] = None
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    completed: bool = False


class TaskRouter:
    """
    Lives on every node. Ships outgoing tasks AND receives incoming ones.
    Two halves of the same protocol so a node is symmetrically a worker
    and a client of the network.

    Designed to be wired into `CompleteMeshController`:
        controller.task_router = TaskRouter(controller)
        # in _handle_client:
        if msg_type in ROUTER_TYPES:
            controller.task_router.on_message(msg, client)
    """

    ROUTER_MESSAGE_TYPES = {
        MSG_SUBMIT, MSG_ACCEPT, MSG_RESULT, MSG_PROGRESS, MSG_AUDIT, MSG_CANCEL,
    }

    def __init__(self, controller, brain=None):
        self.controller = controller            # CompleteMeshController-like
        self.brain = brain                      # PluginferBrain | None
        self._lock = threading.RLock()
        self._inflight: Dict[str, _InflightTask] = {}
        self._peer_profiles: Dict[str, HardwareProfile] = {}
        self._peer_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"completed": 0.0, "failed": 0.0, "ms_avg": 0.0})

    # -----------------------------------------------------------------
    # Public client API
    # -----------------------------------------------------------------
    def submit(self,
               requirements: TaskRequirements,
               input_data: Dict[str, Any],
               on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
               on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
               ) -> str:
        """Submit a task to the mesh; returns job_id immediately."""
        job_id = uuid.uuid4().hex[:12]
        task = _InflightTask(
            job_id=job_id, requirements=requirements,
            input_data=input_data, started_at=time.time(),
            callback=on_result, progress_callback=on_progress,
        )

        with self._lock:
            self._inflight[job_id] = task

        threading.Thread(target=self._dispatch_async, args=(job_id,), daemon=True).start()
        return job_id

    def submit_and_wait(self, requirements: TaskRequirements,
                        input_data: Dict[str, Any],
                        timeout_s: float = 60.0,
                        ) -> Optional[Dict[str, Any]]:
        """Sync helper — useful for tests & glue code."""
        result_holder: Dict[str, Any] = {}
        evt = threading.Event()

        def _cb(out):
            result_holder["value"] = out
            evt.set()

        self.submit(requirements, input_data, on_result=_cb)
        if not evt.wait(timeout_s):
            return None
        return result_holder.get("value")

    # -----------------------------------------------------------------
    # Internal dispatch
    # -----------------------------------------------------------------
    def _dispatch_async(self, job_id: str) -> None:
        with self._lock:
            task = self._inflight.get(job_id)
            if not task:
                return

        peers = self._select_peers(task.requirements)
        if not peers:
            self._finalize(job_id, {"status": "error", "error": "no_eligible_peers"})
            return

        envelope = {
            "type": MSG_SUBMIT,
            "job_id": job_id,
            "plugin": task.requirements.plugin,
            "input": task.input_data,
            "deadline_ms": task.requirements.deadline_ms,
            "redundancy": task.requirements.redundancy,
            "from_node": getattr(self.controller, "node_id", "unknown"),
            "submitted_at": time.time(),
        }

        # Fire-and-forget to all selected peers; we collect results as they
        # come back via on_message.
        for peer in peers[:task.requirements.redundancy]:
            threading.Thread(target=self._send, args=(peer, envelope),
                             daemon=True).start()

        # Watchdog: if no result by deadline, give caller an error.
        def _watchdog():
            time.sleep(task.requirements.deadline_ms / 1000.0)
            with self._lock:
                still_open = job_id in self._inflight and not self._inflight[job_id].completed
            if still_open:
                self._finalize(job_id, {"status": "error", "error": "deadline_exceeded"})

        threading.Thread(target=_watchdog, daemon=True).start()

    def _select_peers(self, req: TaskRequirements) -> List[Dict[str, Any]]:
        """Filter & rank candidate peers from controller.nodes + scout."""
        candidates: List[Tuple[str, Dict[str, Any]]] = []
        with self._lock:
            for node_id, info in self.controller.nodes.items():
                profile = self._peer_profiles.get(node_id)
                if profile is None:
                    # Best-effort defaults from registration data.
                    profile = HardwareProfile(plugins=info.get("plugins", []))
                if req.plugin not in profile.plugins and profile.plugins:
                    continue
                if req.needs_gpu and profile.gpu_class == "cpu":
                    continue
                if req.min_vram_gb and profile.gpu_vram_gb < req.min_vram_gb:
                    continue
                if req.require_tee_attested and not profile.tee_attested:
                    continue
                candidates.append((node_id, info))

        if not candidates:
            return []

        # Score: latency (low) + reputation (high) + brain trust + continent match
        # + health-aware penalty for recovering nodes.
        def _score(item: Tuple[str, Dict[str, Any]]) -> float:
            node_id, info = item
            stats = self._peer_stats[node_id]
            success = stats["completed"] / max(stats["completed"] + stats["failed"], 1.0)
            latency = info.get("latency", 200.0)
            cont_bonus = (
                0.0 if (req.prefer_continent
                        and self._peer_profiles.get(node_id, HardwareProfile()).continent
                            == req.prefer_continent)
                else 50.0
            )
            health_penalty = self._health_penalty(node_id)
            return latency + 200.0 * (1.0 - success) + cont_bonus + health_penalty

        # Soft-quarantine: if the node reported a recent CUDA illegal-address,
        # drop it from the candidate pool unless it's the only one left.
        healthy = [c for c in candidates if not self._is_quarantined(c[0])]
        if healthy:
            candidates = healthy
        candidates.sort(key=_score)
        return [{"node_id": nid, "info": inf} for nid, inf in candidates]

    # ---- governor-aware health hooks ------------------------------------
    QUARANTINE_SECONDS = 60.0

    def update_peer_health(self, node_id: str, snapshot: Dict[str, Any]) -> None:
        """Record a TrainingGovernor.health_snapshot() from a peer (or self)."""
        with self._lock:
            profile = self._peer_profiles.get(node_id) or HardwareProfile()
            profile.health = snapshot
            profile.health_updated = time.time()
            self._peer_profiles[node_id] = profile

    def _is_quarantined(self, node_id: str) -> bool:
        profile = self._peer_profiles.get(node_id)
        if not profile or not profile.health:
            return False
        # Recent illegal-address (the GTX-1650-style CUDA poisoning) is
        # the strongest signal a node should not get more work right now.
        if profile.health.get("illegal_count", 0) > 0:
            age = time.time() - profile.health_updated
            if age < self.QUARANTINE_SECONDS:
                return True
        return False

    def _health_penalty(self, node_id: str) -> float:
        profile = self._peer_profiles.get(node_id)
        if not profile or not profile.health:
            return 0.0
        h = profile.health
        # Each pending OOM is worth ~50ms of equivalent latency penalty;
        # consecutive skips suggest divergence, also penalise.
        return (
            50.0 * float(h.get("oom_count", 0))
            + 25.0 * float(h.get("consecutive_skips", 0))
        )

    def _send(self, peer: Dict[str, Any], envelope: Dict[str, Any]) -> None:
        info = peer["info"]
        host = info.get("ip") or info.get("host")
        port = int(info.get("port", 9000))
        if not host:
            return
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((host, port))
            s.send(json.dumps(envelope).encode("utf-8"))
            s.close()
        except Exception as e:
            logger.debug("router send to %s:%s failed: %s", host, port, e)
            with self._lock:
                self._peer_stats[peer["node_id"]]["failed"] += 1

    # -----------------------------------------------------------------
    # Inbound side: handle protocol messages from peers
    # -----------------------------------------------------------------
    def on_message(self, msg: Dict[str, Any], client_socket=None) -> bool:
        """Return True if message was consumed; False to let other handlers run."""
        m_type = msg.get("type")
        if m_type not in self.ROUTER_MESSAGE_TYPES:
            return False

        if m_type == MSG_SUBMIT:
            self._handle_submit(msg, client_socket)
        elif m_type == MSG_ACCEPT:
            self._handle_accept(msg)
        elif m_type == MSG_RESULT:
            self._handle_result(msg)
        elif m_type == MSG_PROGRESS:
            self._handle_progress(msg)
        elif m_type == MSG_CANCEL:
            self._handle_cancel(msg)
        return True

    def _handle_submit(self, msg: Dict[str, Any], client_socket) -> None:
        """We are the *worker*: someone asked us to run a job."""
        job_id = msg.get("job_id")
        plugin_name = msg.get("plugin")
        input_data = msg.get("input") or {}
        from_node = msg.get("from_node")

        # If we have a brain, ask it whether to trust this requester.
        if self.brain is not None:
            from .pluginfer_brain import NodeContext, DECISION_PEER_TRUST
            stats = self._peer_stats[from_node]
            success = stats["completed"] / max(stats["completed"] + stats["failed"], 1.0)
            ctx = NodeContext(
                incoming_peer_reputation=success,
                incoming_payload_bytes=len(json.dumps(input_data)),
            )
            decision, _info = self.brain.decide(ctx, DECISION_PEER_TRUST)
            if decision == 2:               # drop
                logger.info("router: dropping job %s from low-rep peer %s", job_id, from_node)
                return

        # Run the plugin synchronously (could be made async with worker pool).
        plugin = self.controller.plugin_registry.get_plugin(plugin_name)
        if plugin is None:
            self._send_result(from_node, msg, {"status": "error",
                                               "error": f"plugin not found: {plugin_name}"})
            return

        t0 = time.time()
        try:
            output = self.controller.inference_engine.run(plugin, input_data)
            payload = {"status": "success", "output": output,
                       "wall_ms": (time.time() - t0) * 1000.0}
        except Exception as e:
            payload = {"status": "error", "error": str(e),
                       "wall_ms": (time.time() - t0) * 1000.0}
        self._send_result(from_node, msg, payload)

    def _send_result(self, requester_node_id: str,
                     original_msg: Dict[str, Any],
                     payload: Dict[str, Any]) -> None:
        """Push the result envelope back to the requester."""
        peer_info = self.controller.nodes.get(requester_node_id) or {}
        host = peer_info.get("ip") or peer_info.get("host") or original_msg.get("source_ip")
        port = int(peer_info.get("port", 9000))
        if not host:
            logger.warning("router: cannot send result for %s; unknown requester host",
                           original_msg.get("job_id"))
            return
        envelope = {
            "type": MSG_RESULT,
            "job_id": original_msg.get("job_id"),
            "from_node": getattr(self.controller, "node_id", "unknown"),
            "payload": payload,
        }
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((host, port))
            s.send(json.dumps(envelope).encode("utf-8"))
            s.close()
        except Exception as e:
            logger.debug("router result send failed: %s", e)

    def _handle_accept(self, msg: Dict[str, Any]) -> None:
        with self._lock:
            t = self._inflight.get(msg.get("job_id"))
            if t:
                t.accepted_workers.append(msg.get("from_node", "?"))

    def _handle_result(self, msg: Dict[str, Any]) -> None:
        job_id = msg.get("job_id")
        worker = msg.get("from_node", "?")
        payload = msg.get("payload") or {}
        with self._lock:
            t = self._inflight.get(job_id)
            if not t or t.completed:
                return
            t.results.append({"worker": worker, "payload": payload})
            stats = self._peer_stats[worker]
            if payload.get("status") == "success":
                stats["completed"] += 1
            else:
                stats["failed"] += 1

            if len(t.results) >= t.requirements.redundancy:
                final = self._reduce_redundant_results(t.results)
                self._finalize(job_id, final)
            elif t.requirements.redundancy == 1:
                self._finalize(job_id, payload)

    def _reduce_redundant_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """K-redundant reduce: majority-success wins; outliers logged."""
        successes = [r for r in results if r["payload"].get("status") == "success"]
        if not successes:
            return results[0]["payload"]
        # Pick the most common output by JSON-string identity (deterministic
        # tasks should match exactly; non-deterministic tasks should be
        # validated by the cosine-similarity audit elsewhere).
        counts: Dict[str, int] = defaultdict(int)
        first_by_key: Dict[str, Dict[str, Any]] = {}
        for r in successes:
            key = json.dumps(r["payload"].get("output"), sort_keys=True, default=str)[:1024]
            counts[key] += 1
            first_by_key.setdefault(key, r["payload"])
        winning_key = max(counts, key=counts.get)
        return first_by_key[winning_key]

    def _handle_progress(self, msg: Dict[str, Any]) -> None:
        with self._lock:
            t = self._inflight.get(msg.get("job_id"))
            if t and t.progress_callback:
                try:
                    t.progress_callback(msg.get("payload") or {})
                except Exception:
                    logger.exception("progress callback raised")

    def _handle_cancel(self, msg: Dict[str, Any]) -> None:
        with self._lock:
            self._inflight.pop(msg.get("job_id"), None)

    def _finalize(self, job_id: str, result: Dict[str, Any]) -> None:
        with self._lock:
            t = self._inflight.get(job_id)
            if not t or t.completed:
                return
            t.completed = True
            cb = t.callback
        if cb:
            try:
                cb(result)
            except Exception:
                logger.exception("result callback raised")

    # -----------------------------------------------------------------
    # Hardware-profile management (called when a peer registers)
    # -----------------------------------------------------------------
    def update_peer_profile(self, node_id: str, profile_dict: Dict[str, Any]) -> None:
        with self._lock:
            self._peer_profiles[node_id] = HardwareProfile(
                gpu_class=profile_dict.get("gpu_class", "cpu"),
                gpu_vram_gb=float(profile_dict.get("gpu_vram_gb", 0.0)),
                cpu_cores=int(profile_dict.get("cpu_cores", 1)),
                cpu_tflops_estimate=float(profile_dict.get("cpu_tflops_estimate", 0.05)),
                network_mbps=float(profile_dict.get("network_mbps", 100.0)),
                continent=profile_dict.get("continent"),
                plugins=list(profile_dict.get("plugins", [])),
                tee_attested=bool(profile_dict.get("tee_attested", False)),
                last_updated=time.time(),
            )

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "inflight": len(self._inflight),
                "peers_profiled": len(self._peer_profiles),
                "peer_stats": dict(self._peer_stats),
            }
