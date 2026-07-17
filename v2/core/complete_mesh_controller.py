"""
Complete Mesh Controller - Actually Executes Plugins Remotely
Integrated with Tokenomics (Wallet, Mining, Ledger) and Swarm Security.
"""
import socket
import json
import threading
import time
import hashlib
import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from queue import Queue, Empty
from decimal import Decimal

# Core Components
from .plugin_registry import PluginRegistry
from .networking import NetworkManager
from .inference_engine import InferenceEngine
from .security_manager import SecurityManager
from .hardware_detector import HardwareDetector
from .dependency_manager import DependencyManager
from .advanced_mesh_features import AdvancedMeshFeatures
from .ai_sentinel import AISentinel
from .updater import AutoUpdater
from .self_learning import SelfLearningOptimizer
from .gossip import GossipProtocol
from .staking import StakingContract
from .broker import EconomicBroker
from .scout import NetworkScout
from .architect import TaskArchitect
from .smart_contracts import SmartContractVM
from .auditor import SystemAuditor
from .oracle import PricingOracle

# [V3 UPGRADE] Security & Verification
from .secure_sandbox import SecureSandbox
from .arbiter import Arbiter

# Tokenomics & Ledger
from .compute_ledger import ComputeLedger
from .tokenomics import Wallet, TokenMinter, Transaction, BURN_ADDRESS
from .marketplace import IPMarketplace
from .logger_setup import setup_logger
from .election import LeaderElection
from .reputation import ReputationManager
from .interop import BridgeManager

# Configure Rotating Logger
logger = setup_logger("MeshController")

# Startup Check (Moved to __init__ to prevent import crashes)
# try:
#     DependencyManager.ensure_accelerators()
# except Exception as e:
#     logger.warning(f"Optimization auto-setup failed: {e}")

@dataclass
class MeshTask:
    """Represents a task to be distributed"""
    task_id: str
    plugin_name: str
    input_data: Dict[str, Any]
    priority: int = 5
    timeout: int = 300
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

# ---------------------------------------------------------------------------
# CP-2 Task 2.2: Layer-3 hardcoded bootstrap seed nodes.
#
# These addresses + pubkeys are pinned at build time. The seeds are operated
# by the Pluginfer core team / community; they are a CONVENIENCE for fast
# discovery, NOT a control plane (a node that prefers DHT-only bootstrap is
# treated identically). Multi-region for redundancy: a single seed going
# down doesn't break new-node onboarding.
#
# To replace these on a fork: deploy your own seeds via
# v2/infrastructure/seed_node/deploy.sh, generate the seed wallet, paste
# the resulting pubkey here. NEVER ship a wallet PEM in this list -- only
# the public key.
#
# The pubkey field is currently advisory (Task 2.2 ships the address +
# self-registration; pinned-key seed-response signature verification is
# the next CP-2 hardening pass before production launch).
BOOTSTRAP_SEEDS: list[dict] = [
    # {"host": "seed-eu.pluginfer.network", "port": 9000,
    #  "pubkey_pem": "-----BEGIN PUBLIC KEY-----\n<seed wallet pubkey>\n-----END PUBLIC KEY-----",
    #  "region": "eu"},
    # {"host": "seed-us.pluginfer.network", "port": 9000,
    #  "pubkey_pem": "-----BEGIN PUBLIC KEY-----\n<seed wallet pubkey>\n-----END PUBLIC KEY-----",
    #  "region": "us-east"},
]

PLUGINFER_NODE_VERSION: str = "1.0.0"


class CompleteMeshController:
    """
    Complete Mesh Network Controller with ACTUAL plugin execution.
    Features:
    - Token Mining & Wallet Integration
    - Swarm ID Security Enforcement
    - Full Blockchain Ledger
    """
    
    def __init__(self, host: str = '0.0.0.0', port: int = 9999, mode: str = 'coordinator'):
        self.host = host
        # Resolve Wildcard Host without phoning home.
        # The previous version did `s.connect(("8.8.8.8", 80))` which
        # leaked every node's startup to Google DNS *and* silently fell
        # through to host='0.0.0.0' in air-gapped / offline-first
        # environments. Now we use a link-local address that doesn't
        # leave the LAN; the OS still fills in the local IP it would
        # have used to route there.
        if self.host == '0.0.0.0':
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    # 198.51.100.0/24 is RFC 5737 documentation range —
                    # never routable, never sent to anyone. The OS
                    # picks a local interface anyway.
                    s.connect(("198.51.100.1", 1))
                    self.host = s.getsockname()[0]
                except OSError:
                    # Fall back to gethostbyname; last resort '127.0.0.1'.
                    try:
                        self.host = socket.gethostbyname(socket.gethostname())
                    except OSError:
                        self.host = '127.0.0.1'
                finally:
                    s.close()
            except OSError:
                self.host = '127.0.0.1'
                
        self.port = port
        self.mode = mode
        self.node_id = self._generate_node_id()
        
        # Load Config (Swarm ID)
        self.swarm_id = 'public'
        try:
            if os.path.exists("config.json"):
                with open("config.json", 'r') as f:
                    cfg = json.load(f)
                    self.swarm_id = cfg.get('swarm_id', 'public')
        except: pass
        logger.info(f"Identity Initialized: {self.node_id} (Swarm: {self.swarm_id})")

        # Network & Tools
        self.net_manager = NetworkManager(port)
        
        # [FIX] Bind broadcast for Election module
        # The Election module expects net_manager.broadcast_message(msg)
        # We route this through our internal peer broadcast
        self.net_manager.broadcast_message = self._broadcast_election_msg
        
        self.nodes: Dict[str, Dict] = {}
        self.task_queue = Queue()
        self.results: Dict[str, Any] = {}
        self.running = False
        self.paused = False
        self.server_socket = None
        self.threads = []
        
        # Task Tracking
        self.task_states: Dict[str, str] = {} 
        self.task_progress: Dict[str, Dict] = {} 
        self.task_events: Dict[str, threading.Event] = {} 
        
        # Plugins & Engine
        self.plugin_registry = PluginRegistry()
        self.inference_engine = InferenceEngine()
        self._plugins_loaded = False
        
        # Hardware & Security
        self.hardware_detector = HardwareDetector()
        self.local_hardware = self.hardware_detector.get_best_device()
        self.local_score = self.hardware_detector.get_performance_score()
        self.security_manager = SecurityManager()
        
        # TOKENOMICS INTEGRATION
        # 1. Wallet
        loaded_wallet = Wallet.load_from_file()
        if loaded_wallet:
             self.wallet = loaded_wallet
             logger.info(f"[WALLET LOADED]: {self.wallet.address}")
        else:
             self.wallet = Wallet()
             self.wallet.save_to_file() # Save new wallet immediately
             logger.info(f"[WALLET NEW]: {self.wallet.address}")
        
        # 2. Ledger & Minter
        self.ledger = ComputeLedger(self.node_id)
        self.ledger.load_chain() # Try to load existing chain
        self.minter = TokenMinter()
        
        # Advanced Modules
        self.advanced = AdvancedMeshFeatures()
        self.sentinel = AISentinel(sensitivity=3.0)
        self.updater = AutoUpdater()
        self.brain = SelfLearningOptimizer()
        
        # Discovery
        self.discovery = None
        
        # [PERSISTENCE] Load Peers
        self.known_peers = []
        self._load_peers()
        
        # [PHASE 3] Gossip Protocol for Global Consensus
        self.gossip = GossipProtocol(self.node_id, self._broadcast_to_peers)
        
        # [PHASE 3] DeFi Staking
        self.staking = StakingContract(self.ledger)
        
        # [PHASE 3.5] The Broker (Economic Agent)
        self.broker = EconomicBroker(self.wallet, self.ledger, self.staking)
        self.broker_last_run = 0
        
        # [PHASE 3.5] The Scout (Network Agent)
        self.scout = NetworkScout(self.node_id, self.known_peers)
        self.scout.start_scouting()
        
        # [PHASE 3.5] The Architect (Task Agent)
        self.architect = TaskArchitect(self.scout)
        
        # [PHASE 4] Smart Contracts (Mini-VM)
        self.vm = SmartContractVM(self.ledger)
        
        # [PHASE 3.5 Extension] The Auditor (Compliance Agent)
        self.auditor = SystemAuditor(self.ledger, core_path="./core")
        self.auditor.start_background_audit()
        
        # [PHASE 5] Pricing Oracle
        self.oracle = PricingOracle()
        
        # [PHASE 6] IP Marketplace (Royalties)
        # [PHASE 6] IP Marketplace (Royalties)
        self.marketplace = IPMarketplace(self.ledger)
        
        # [V3 UPGRADE] The Arbiter (Trustless Verification)
        self.arbiter = Arbiter(self.ledger)
        logger.info("[V3] Arbiter Module Online (Probabilistic Audit Enabled)")

        # [REPUTATION] — chain-derived (W28). Score is computed from
        # ledger-observable events; node_id is the wallet address.
        self.reputation = ReputationManager(
            ledger=self.ledger,
            address=getattr(self, "wallet_address", None) or self.node_id,
        )
        
        # [PHASE 8] Leader Election (Auto-Failover)
        self.election = LeaderElection(
            node_id=self.node_id,
            ledger=self.ledger,
            net_manager=self.net_manager,
            reputation_manager=self.reputation,
            on_promote_callback=self._promote_to_coordinator,
            peer_count_fn=lambda: len(self.nodes)
        )

    def _promote_to_coordinator(self):
        """Callback: Election Won via Consensus"""
        if self.mode == 'coordinator': return
        
        logger.warning(f"👑 ELECTION WON! Promoting Self to COORDINATOR")
        self.mode = 'coordinator'

    def _broadcast_election_msg(self, msg: Dict[str, Any]):
        """Bridge: Election -> Mesh Broadcast"""
        # Wrap in GOSSIP envelope or just send raw if peers expect it
        # Election messages are usually raw JSON
        # We recycle _broadcast_to_peers
        envelope = {
            'type': 'ELECTION',
            'origin': self.node_id,
            'payload': msg
        }
        self._broadcast_to_peers(envelope)

    def _broadcast_to_peers(self, envelope: Dict[str, Any], exclude_sender: bool = False):
        """
        Send Gossip Envelope to all known peers.
        Uses threading to avoid blocking the main loop.
        """
        origin = envelope.get('origin')
        
        def _send(peer):
            try:
                ip = peer.get('ip')
                port = peer.get('port')
                # Don't send back to origin (flooding optimization)
                if ip == self.host and int(port) == self.port: return
                
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2) # Fast timeout for gossip
                s.connect((ip, int(port)))
                s.send(json.dumps(envelope).encode())
                s.close()
            except: pass

        # Broadcast loop
        for peer in self.known_peers:
            threading.Thread(target=_send, args=(peer,), daemon=True).start()

    def _handle_gossip_payload(self, payload: Dict[str, Any]):
        """Reaction to new Gossip Message (Block/TX)"""
        p_type = payload.get('type') # 'BLOCK' or 'TX' (payload was passed as dict)
        # Note: payload is the 'data' part, envelope has GOSSIP wrapper
        
        # Actually in gossip.py logic: payload IS the inner data.
        # But we need to check if payload has a 'hash'? 
        # Wait, if I broadcast new_block.to_dict(), then payload IS the block dict.
        
        # The block dict has no 'type' field by default using to_dict().
        # We need to distinguish Block vs TX vs Other.
        # But gossip protocol separates 'gossip_type' in envelope.
        # Ah, handle_gossip returns (is_new, payload). It does NOT return gossip_type.
        # I need gossip_type from the envelope in the handler?
        # Let's check gossip.py logic again.
        
        # gossip.py: return True, envelope.get('payload')
        # It loses the 'gossip_type' context unless payload has it.
        # I should probably wrap payload in broadcast call? 
        # i.e. self.gossip.broadcast('BLOCK', block_dict)
        # Recipient sees envelope.gossip_type = 'BLOCK'.
        # handle_gossip processes envelope. 
        # It returns payload.
        
        # Fix: separate handler for envelope vs payload
        pass

    def _load_peers(self):
        """Load known peers from disk (History)"""
        self.known_peers = []
        try:
            if os.path.exists("peers.json"):
                with open("peers.json", 'r') as f:
                    self.known_peers = json.load(f)
                logger.info(f"Loaded {len(self.known_peers)} known peers from history")
        except Exception as e:
            logger.error(f"Failed to load peers: {e}")

    def _save_peers(self):
        """Save known peers to disk"""
        try:
            # simple dedupe
            seen = set()
            unique_peers = []
            for p in self.known_peers:
                key = f"{p.get('ip')}:{p.get('port')}"
                if key not in seen:
                    seen.add(key)
                    unique_peers.append(p)
            
            with open("peers.json", 'w') as f:
                json.dump(unique_peers, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save peers: {e}")

    def _bootstrap_connectivity(self):
        """The 'Cannot Fail' Connection Logic"""
        # 1. UPnP (Open Firewall)
        threading.Thread(target=self.net_manager.enable_upnp, daemon=True).start()
        
        # 2. Connect to Seeds  (3-layer fallback chain)
        seeds = self.known_peers.copy() # Layer 1: peers.json history

        # Layer 2: explicit local config
        try:
            if os.path.exists("config.json"):
                with open("config.json", 'r') as f:
                    cfg = json.load(f)
                    config_seeds = cfg.get('seeds', [])
                    seeds.extend(config_seeds)
        except Exception:
            pass

        # Layer 3: hardcoded community seed nodes (CP-2 Task 2.2).
        # Each entry pins the seed's wallet pubkey at build time, so a
        # DNS-spoofing attacker who captures the IP cannot impersonate a
        # legitimate seed (the peer-list response is implicit-trust here;
        # the registration ACK we get back is incidental data, no
        # signature gate yet -- DHT-bootstrap path in W19-deep is the
        # tamper-evident successor).
        seeds_from_bootstrap = self._bootstrap_from_seeds()
        seeds.extend(seeds_from_bootstrap)

        if not seeds:
            logger.info(
                "No seeds found across history, config, or BOOTSTRAP_SEEDS. "
                "Waiting for incoming connections. Operator can also pass "
                "--add-peer <ip:port> on the next launch."
            )
            return

        logger.info(f"Bootstrapping: Attempting to connect to {len(seeds)} peers...")
        for peer in seeds:
            if isinstance(peer, dict) and 'ip' in peer:
                self.connect_to_peer(peer['ip'], peer.get('port', 9000))

    def _bootstrap_from_seeds(self) -> list:
        """Try each entry in BOOTSTRAP_SEEDS, register self, fetch peer list.

        Returns a list of {ip, port, ...} dicts compatible with the
        existing seeds-list shape. Entries already known via Layer 1/2
        are deduplicated by the caller. Errors per-seed are swallowed
        and logged; total failure means we return [] and the controller
        continues with whatever Layer 1 / Layer 2 provided.
        """
        out: list = []
        if not BOOTSTRAP_SEEDS:
            return out
        from infrastructure.seed_node import seed_client as _sc

        local_ip = _sc.discover_local_ip()
        for entry in BOOTSTRAP_SEEDS:
            seed = _sc.SeedAddress(host=entry['host'], port=entry.get('port', 9000))
            try:
                # Register self so other newly-arriving nodes can find us
                _sc.register_sync(
                    seed,
                    pubkey_pem=self.wallet.public_key_pem,
                    sign_fn=self.wallet.sign,
                    ip=local_ip,
                    port=self.port,
                    node_version=PLUGINFER_NODE_VERSION,
                )
                # Pull a sample of live peers
                peers = _sc.fetch_peers_sync(seed, max_n=50)
            except Exception as e:
                logger.warning(
                    "[BOOTSTRAP] seed %s:%s unreachable: %s",
                    seed.host, seed.port, e,
                )
                continue
            if not peers:
                logger.info(
                    "[BOOTSTRAP] seed %s:%s reachable but empty",
                    seed.host, seed.port,
                )
                continue
            for p in peers:
                # Skip self (same pubkey).
                if p.get('pubkey_pem') == self.wallet.public_key_pem:
                    continue
                out.append({
                    'ip': p['ip'],
                    'port': p['port'],
                    'pubkey_pem': p.get('pubkey_pem'),
                    'node_version': p.get('node_version'),
                })
            logger.info(
                "[BOOTSTRAP] seed %s:%s returned %d peers",
                seed.host, seed.port, len(peers),
            )
        # Persist the freshly-discovered seeds so a restart can reach
        # the network even if every seed VPS goes down.
        if out:
            try:
                self._persist_peers(out)
            except Exception as e:
                logger.debug("persist_peers failed: %s", e)
        return out

    def _persist_peers(self, new_peers: list) -> None:
        """Append to peers.json so a restart doesn't always need a seed."""
        path = "peers.json"
        existing: list = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    existing = json.load(f)
            except Exception:
                existing = []
        seen = {(p.get('ip'), p.get('port')) for p in existing}
        for p in new_peers:
            key = (p.get('ip'), p.get('port'))
            if key not in seen:
                existing.append(p)
                seen.add(key)
        with open(path, 'w') as f:
            json.dump(existing, f, indent=2)

    def connect_to_peer(self, ip: str, port: int):
        """Active TCP Connection to a Peer"""
        if ip == self.host and int(port) == self.port: return # Don't connect to self
        
        def _connect():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5)
                s.connect((ip, int(port)))
                
                # Handshake
                handshake = {
                    'type': 'register',
                    'swarm_id': self.swarm_id,
                    'node_info': {
                        'node_id': self.node_id,
                        'ip': self.host, 
                        'port': self.port,
                        'role': self.mode
                    }
                }
                s.send(json.dumps(handshake).encode())
                
                # [PEX] Ask for their peers
                pex_req = {'type': 'get_peers', 'swarm_id': self.swarm_id}
                s.send(json.dumps(pex_req).encode())
                
                # Receive PEX Response
                try:
                    data = s.recv(16384) # Large buffer for peer list
                    if data:
                        msg = json.loads(data.decode())
                        if msg.get('type') == 'peer_list':
                            new_peers = msg.get('peers', [])
                            for p in new_peers:
                                if isinstance(p, dict) and 'ip' in p and 'port' in p:
                                    self.known_peers.append(p)
                            self._save_peers()
                            # logger.info(f"[PEX] Synced {len(new_peers)} peers")
                except: pass
                
                s.close()
            except Exception as e:
                # logger.debug(f"Failed to connect to {ip}:{port}")
                pass

        threading.Thread(target=_connect, daemon=True).start()
    
    def _generate_node_id(self) -> str:
        unique_str = f"{socket.gethostname()}-{self.host}-{self.port}-{time.time()}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:16]

    def _ensure_plugins_loaded(self):
        if not self._plugins_loaded:
            count = self.plugin_registry.discover_plugins()
            self._plugins_loaded = True
            logger.info(f"Loaded {count} plugins")
            
    def start(self):
        self.running = True
        self._ensure_plugins_loaded()
        
        # TCP Server
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        
        threading.Thread(target=self._listen_for_connections, daemon=True).start()
        
        if self.mode in ['worker', 'hybrid']:
            threading.Thread(target=self._process_tasks, daemon=True).start()
        
        # Discovery with Swarm ID
        from .discovery import MeshDiscovery
        caps = {'plugins': self.plugin_registry.list_plugins()}
        self.discovery = MeshDiscovery(self.node_id, self.port, caps, swarm_id=self.swarm_id)
        self.discovery.start(role=self.mode)
        
        # [BOOTSTRAP] Start Smart Connectivity
        self._bootstrap_connectivity()
        
        logger.info(f"Node Started. Listening on {self.port}")

    def stop(self):
        self.running = False
        if self.discovery: self.discovery.stop()
        if self.server_socket: self.server_socket.close()
        # [PERSISTENCE] Save state on shutdown
        try:
            self.ledger.save_chain()
        except: pass

    def pause(self):
        self.paused = True
        logger.info("[PAUSED] Node Paused")

    def resume(self):
        self.paused = False
        logger.info("[RESUMED] Node Resumed")

    def _listen_for_connections(self):
        """Accept TCP connections with SWARM ID Handshake"""
        while self.running:
            try:
                client, addr = self.server_socket.accept()
                threading.Thread(target=self._handle_client, args=(client, addr), daemon=True).start()
            except:
                if self.running: time.sleep(1)

    def _handle_client(self, client, addr):
        try:
            # Simple Handshake
            data = client.recv(4096).decode('utf-8')
            msg = json.loads(data)
            
            # [SECURITY] SWARM CHECK
            remote_swarm = msg.get('swarm_id', 'public')
            if remote_swarm != self.swarm_id:
                logger.warning(f"[REJECTED] connection from {addr}: Wrong Swarm ID ({remote_swarm})")
                client.close()
                return

            # Process Message
            msg_type = msg.get('type')
            
            # [PHASE 8] Election Messages
            if "ELECTION" in str(msg_type) or "HEARTBEAT" in str(msg_type):
                self.election.handle_message(msg)
                return
            
            if msg_type == 'task':
                # Worker receiving task
                task_data = msg.get('task')
                # Add to queue... (Implementation simplified)
                
            elif msg_type == 'register':
                # Coordinator receiving registration. Capture wallet_address
                # so payments go to a signable PLG address, not an IP string.
                node_info = msg.get('node_info') or {}
                # Backwards-compat: older nodes may not include wallet_address.
                # We accept registration but mark it unpayable until they upgrade.
                if 'wallet_address' not in node_info:
                    node_info['wallet_address'] = None
                    node_info['payable'] = False
                else:
                    node_info['payable'] = True
                self.nodes[node_info['node_id']] = node_info
                client.send(json.dumps({
                    'status': 'success',
                    'coordinator_wallet': self.wallet.address,
                }).encode())
                
            elif msg_type == 'get_peers':
                # [PEX] Respond with our known peers
                response = {
                    'type': 'peer_list',
                    'swarm_id': self.swarm_id,
                    'peers': self.known_peers
                }
                try:
                    client.send(json.dumps(response).encode())
                except: pass
            
            elif msg_type == 'GOSSIP':
                # [PHASE 3] Handle Gossip
                # capture type first
                g_type = msg.get('gossip_type')
                is_new, payload = self.gossip.handle_gossip(msg)
                if is_new and payload:
                   if g_type == 'BLOCK':
                       logger.info(f"[CONSENSUS] New Block Received via Gossip")
                       if self.ledger.receive_remote_block(payload):
                           self.ledger.save_chain() # Persist if accepted
                   elif g_type == 'TX':
                       logger.info(f"[CONSENSUS] New Transaction Received")
                   elif g_type == 'GOVERNOR_HEALTH':
                       # gap #9 — peer reports a health snapshot
                       # so the local task router can quarantine
                       # recovering nodes. payload = {sender_id, snapshot}.
                       self._handle_governor_health(payload)
                
            client.close()
        except Exception as e:
            logger.error(f"Client handler error: {e}")
            client.close()

    # ---- Governor health gossip (gap #9 fix) -----------------------------
    def gossip_governor_health(self, snapshot: Dict[str, Any]) -> None:
        """Wire endpoint for `TrainingGovernor.broadcast_fn`. Wraps the
        snapshot in a GOSSIP envelope and floods to known peers. Peers'
        `_handle_governor_health` updates their local TaskRouter so
        recovering nodes are quarantined within one tick.
        """
        payload = {"sender_id": self.node_id, "snapshot": snapshot}
        # Reuse the existing gossip protocol so dedupe / TTL / origin
        # tracking come for free.
        try:
            self.gossip.broadcast("GOVERNOR_HEALTH", payload)
        except Exception as e:
            # Falls through to direct broadcast if the gossip layer
            # isn't ready (e.g. controller starting up).
            logger.debug("gossip.broadcast failed, direct path: %s", e)
            envelope = {
                "type": "GOSSIP", "gossip_type": "GOVERNOR_HEALTH",
                "origin": self.node_id, "payload": payload,
            }
            self._broadcast_to_peers(envelope)

    def _handle_governor_health(self, payload: Dict[str, Any]) -> None:
        """Receive a peer's health snapshot. Updates the task_router so
        the next routing decision penalises / quarantines them."""
        sender_id = payload.get("sender_id")
        snapshot = payload.get("snapshot") or {}
        if not sender_id:
            return
        tr = getattr(self, "task_router", None)
        if tr is None:
            return
        try:
            tr.update_peer_health(sender_id, snapshot)
        except Exception as e:
            logger.debug("update_peer_health failed for %s: %s",
                         sender_id[:12], e)

    def submit_task(self, plugin_name: str, input_data: Dict[str, Any], priority: int = 5) -> str:
        task_id = hashlib.sha256(f"{plugin_name}-{time.time()}".encode()).hexdigest()[:16]
        task = MeshTask(task_id, plugin_name, input_data, priority)
        task = MeshTask(task_id, plugin_name, input_data, priority)
        
        # [V3 UPGRADE] Arbiter Spot Check Logic
        if self.arbiter.should_audit(task_id):
            logger.info(f"🎲 ARBITER: Task {task_id} selected for SPOT CHECK audit.")
            # In a full implementation, we would duplicate this task here.
            # For now, we flag it so the worker knows it's being watched.
            task.input_data['_audit_flag'] = True
            
        self.task_queue.put(task)
        return task_id

    def get_result(self, task_id: str, timeout: float = None) -> Optional[Dict]:
        start = time.time()
        while True:
            if task_id in self.results:
                return self.results[task_id]
            if timeout is not None and (time.time() - start) > timeout:
                return None
            time.sleep(0.1)

    def submit_complex_job_api(self, data: Dict[str, Any]) -> str:
        """API: Smart Job Submission"""
        return self.architect.submit_complex_job(data)

    def deploy_contract(self, code: str) -> str:
        """API: Deploy Smart Contract"""
        return self.vm.deploy_contract(self.wallet.address, code)
        
    def call_contract(self, address: str, func: str, args: list) -> Any:
        """API: Execute Smart Contract using V3 Secure Sandbox"""
        # [V3 UPGRADE] Use Secure Sandbox instead of raw VM execution
        # We fetch the code from VM but execute it safely
        contract = self.vm.contracts.get(address)
        if not contract:
             raise ValueError("Contract not found")
        
        code = contract['code']
        # Helper to inject args into simple script interaction if needed
        # For now, we assume the code is a script that uses 'args' global
        return SecureSandbox.run(code, args=args)

    def _process_tasks(self):
        """Worker Loop: Process Tasks & MINT TOKENS"""
        while self.running:
            if self.paused:
                time.sleep(1)
                continue
                
            # [PHASE 3.5] Broker Agent: Economic Optimization (Every 60s)
            # Checks if we have too much idle cash and auto-stakes it
            if time.time() - self.broker_last_run > 60:
                self.broker.evaluate_strategy()
                self.broker_last_run = time.time()

            # [PHASE 3.5] Healer Agent: Active Monitoring
            # Check every 10 iterations or so to be efficient, but for safety check often
            health_check = self.brain.monitor_health()
            if health_check.get('action') == 'THROTTLE' and not self.paused:
                logger.warning(f"[HEALER] Auto-Pausing Node: {health_check.get('reason')}")
                self.pause()
                time.sleep(5) # Cool down
                continue
            elif health_check.get('action') == 'EMERGENCY_RESTART':
                 logger.error(f"[HEALER] CRITICAL: System unstable. {health_check.get('reason')}. Exiting to trigger auto-restart wrapper.")
                 self.stop()
                 return
                
            try:
                task = self.task_queue.get(timeout=1)
                
                # Execute
                self.task_states[task.task_id] = 'RUNNING'
                result = self._execute_task(task)
                self.results[task.task_id] = result
                
                # Tokenomics Logic
                if result.get('status') == 'success':
                    self.task_states[task.task_id] = 'COMPLETED'
                    
                    # [SAFEGUARD] Minting should not block task completion
                    try:
                        # 1. Calculate Complexity
                        # (Simple proxy: execution time or file size)
                        complexity = 1.0
                        try:
                            sz = task.input_data.get('size_bytes', 0)
                            if sz > 1000000: complexity = 2.0 # >1MB = 2x reward
                        except: pass
                        
                        # 2. Mint Tokens
                        current_height = self.ledger.get_height()
                        tx = self.minter.mint_coinbase(
                            recipient_addr=self.wallet.address,
                            block_height=current_height, 
                            difficulty_factor=complexity
                        )
                        
                        if tx:
                            # 3. Add to Ledger
                            self.ledger.add_transaction(tx)
                            new_block = self.ledger.mine_block(self.wallet.address)
                            
                            # [REPUTATION]
                            self.reputation.record_block()
                            self.reputation.record_task()

                            if new_block:
                                self.ledger.save_chain() # [PERSISTENCE] Save after every block
                                logger.info(f"[MINTED] {tx.amount} PLG for Task {task.task_id}")
                                
                                # [PHASE 3] Broadcast Block
                                self.gossip.broadcast('BLOCK', new_block.to_dict())
                    except Exception as e:
                        logger.error(f"[MINT FAILURE] Token Minting Failed for Task {task.task_id}: {e}")
                        # Task is already marked COMPLETED, so we are safe.
                else:
                    self.task_states[task.task_id] = 'FAILED'
                    
            except Empty:
                pass
            except Exception as e:
                logger.error(f"Worker Loop Error: {e}")

    def _execute_task(self, task: MeshTask) -> Dict:
        try:
            plugin = self.plugin_registry.get_plugin(task.plugin_name)
            if not plugin:
                raise Exception(f"Plugin {task.plugin_name} not found")
                
            # Run
            result_data = self.inference_engine.run(plugin, task.input_data)
            
            # Strict Error Check
            if isinstance(result_data, dict) and 'error' in result_data:
                return {'status': 'error', 'error': result_data['error']}
                
            return {'status': 'success', 'result': result_data}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    # Additional helpers required by dashboard
    def get_mesh_stats(self):
        return {
            'total_nodes': len(self.nodes) + 1,
            'active_peers': len(self.scout.get_best_peers(100)), # Real active count
            'best_peer_latency': self.scout.get_best_peers(1)[0]['latency'] if self.scout.get_best_peers(1) else -1,
            'audit_status': 'ACTIVE',
            'tasks_completed': self.ledger.get_height(), # Use block height as proxy
            'revenue_earned': float(self.ledger.get_balance(self.wallet.address)), # Real Wallet Balance
            'node_id': self.node_id,
            'mode': self.mode,
            'queue_size': self.task_queue.qsize(),
            'is_paused': self.paused,
            'staked_amount': self.staking.get_staking_info(self.wallet.address).get('amount', 0) if self.staking.get_staking_info(self.wallet.address) else 0
        }

    def _pay_peer(self, peer_node_id: str, plugin_name: str = "generic"):
        """
        Pay a peer for completed work.

        IMPORTANT: this method now requires a *node_id*, not an IP address.
        The previous version sent funds to peer_ip, which is not a signable
        wallet address; those funds were unrecoverable. We resolve the
        peer's wallet via the registration table and refuse to pay if the
        peer hasn't published one.

        Split:
          70% Worker · 20% Treasury · 5% Burn · 5% Plugin Royalty
        """
        node_info = self.nodes.get(peer_node_id) or {}
        peer_wallet = node_info.get('wallet_address')
        if not peer_wallet or not node_info.get('payable'):
            logger.warning(
                "[PAY-PEER] Refusing to pay node %s: no wallet_address registered. "
                "Peer must upgrade to v3 client.", peer_node_id,
            )
            return False

        wallet = self.wallet
        TREASURY_ADDRESS = "PLG_FOUNDATION_TREASURY_MAINNET"

        total_amount = self.oracle.get_task_price(plugin_name)
        burn_amount = total_amount * Decimal("0.05")
        royalty_amount = total_amount * Decimal("0.05")
        platform_tax = total_amount * Decimal("0.20")
        worker_pay = total_amount - platform_tax - burn_amount - royalty_amount
        optimal_fee = self.broker.estimate_safe_fee()

        # A. Worker (70%) -- now to a real signable PLG address
        tx_worker = Transaction(
            sender=wallet.address,
            recipient=peer_wallet,
            amount=worker_pay,
            type='task_payment',
            sender_pub_key=wallet.public_key_pem,
        )
        tx_worker.fee = optimal_fee
        tx_worker.tx_id = tx_worker.calculate_hash()
        tx_worker.signature = wallet.sign(tx_worker.tx_id)
        self.ledger.add_transaction(tx_worker)
        logger.info(
            "PAID worker %s (node %s): %s PLG (70%%)",
            peer_wallet[:12], peer_node_id[:8], worker_pay,
        )
        
        # 5. B: Treasury (20%)
        if platform_tax > 0:
            tx_treasury = Transaction(
                sender=wallet.address,
                recipient=TREASURY_ADDRESS,
                amount=platform_tax,
                type='platform_fee',
                sender_pub_key=wallet.public_key_pem
            )
            tx_treasury.tx_id = tx_treasury.calculate_hash()
            tx_treasury.signature = wallet.sign(tx_treasury.tx_id)
            self.ledger.add_transaction(tx_treasury)
            logger.info(f"PAID Treasury {platform_tax} PLG (20%)")

        # 6. C: BURN (5%)
        if burn_amount > 0:
            tx_burn = Transaction(
                sender=wallet.address,
                recipient=BURN_ADDRESS,
                amount=burn_amount,
                type='burn', 
                sender_pub_key=wallet.public_key_pem
            )
            tx_burn.tx_id = tx_burn.calculate_hash()
            tx_burn.signature = wallet.sign(tx_burn.tx_id)
            self.ledger.add_transaction(tx_burn)
            logger.info(f"🔥 BURNED {burn_amount} PLG (5%)")
            
        # 7. D: ROYALTY (5%)
        # [NEW REAL WORLD USE CASE]
        owner = self.marketplace.get_owner(plugin_name)
        if owner and royalty_amount > 0:
            tx_royalty = Transaction(
                sender=wallet.address,
                recipient=owner,
                amount=royalty_amount,
                type='royalty',
                sender_pub_key=wallet.public_key_pem
            )
            tx_royalty.tx_id = tx_royalty.calculate_hash()
            tx_royalty.signature = wallet.sign(tx_royalty.tx_id)
            self.ledger.add_transaction(tx_royalty)
            logger.info(f"🎨 ROYALTY PAID {royalty_amount} PLG to {owner[:8]}")

    def stake_tokens(self, amount: float):
        """User Action: Stake Tokens"""
        try:
            # We construct a Transfer TX to the Pool
            tx = Transaction(
                sender=self.wallet.address,
                recipient="PLG_STAKING_POOL_V1",
                amount=amount,
                type='transfer',
                sender_pub_key=self.wallet.public_key_pem
            )
            # Sign
            tx.signature = self.wallet.sign(tx.tx_id)
            
            # Add to Ledger
            if self.ledger.add_transaction(tx):
                # Immediate mine for demo (in prod this waits for miner)
                new_block = self.ledger.mine_block(self.wallet.address)
                if new_block:
                    self.ledger.save_chain()
                    self.gossip.broadcast('BLOCK', new_block.to_dict())
                    
                logger.info(f"[STAKING] Staked {amount} PLG")
                return True
        except Exception as e:
            logger.error(f"Staking failed: {e}")
        return False
