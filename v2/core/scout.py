
"""
The Scout (Network Agent)
Active P2P Latency Monitoring & Routing Optimization.
- Periodically pings known peers to check health.
- Maintains a 'Routing Table' sorted by latency (Speed).
- Isolates slow/dead nodes.
"""
import time
import socket
import logging
import threading
from typing import List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class PeerStats:
    ip: str
    port: int
    latency_ms: float = 9999.0
    last_seen: float = 0.0
    success_rate: float = 1.0
    
class NetworkScout:
    def __init__(self, node_id: str, known_peers: List[Dict]):
        self.node_id = node_id
        self.peers: Dict[str, PeerStats] = {} # "ip:port" -> Stats
        self.lock = threading.Lock()
        
        # Init with known peers
        for p in known_peers:
            key = f"{p.get('ip')}:{p.get('port')}"
            self.peers[key] = PeerStats(p.get('ip'), p.get('port'))
            
    def get_best_peers(self, count: int = 5) -> List[Dict]:
        """
        [GOLD STANDARD] Dynamic Routing
        Returns the top N fastest peers.
        """
        with self.lock:
            # Sort by latency (lowest first)
            sorted_peers = sorted(
                self.peers.values(), 
                key=lambda x: x.latency_ms
            )
            
            # Convert back to dict format for controller
            results = []
            for p in sorted_peers[:count]:
                if p.latency_ms < 1000: # Only return usable peers (<1s latency)
                    results.append({'ip': p.ip, 'port': p.port, 'latency': p.latency_ms})
            return results

    def start_scouting(self):
        """Start background auditing"""
        threading.Thread(target=self._scout_loop, daemon=True).start()
        
    def _scout_loop(self):
        """
        Periodically ping peers to update stats.
        """
        while True:
            try:
                # 1. Snapshot peers to check
                with self.lock:
                    targets = list(self.peers.values())
                    
                for peer in targets:
                    latency = self._ping_peer(peer.ip, peer.port)
                    
                    with self.lock:
                        if latency is not None:
                            peer.latency_ms = latency
                            peer.last_seen = time.time()
                        else:
                            # Penalty for timeout
                            peer.latency_ms = 9999.0 
                            
                time.sleep(30) # Scout every 30 seconds
            except Exception as e:
                logger.error(f"[SCOUT] Loop error: {e}")
                time.sleep(30)

    def _ping_peer(self, ip: str, port: int) -> float:
        """
        Measure TCP Connect time (Ping equivalent for app layer)
        """
        try:
            start = time.time()
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0) # Strict timeout
            s.connect((ip, int(port)))
            s.close()
            return (time.time() - start) * 1000 # ms
        except:
            return None
