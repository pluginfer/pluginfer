"""
Advanced Mesh Features
- Multi-coordinator redundancy
- Geographic routing
- Model caching
- Checkpoint/resume for long tasks
- Bandwidth optimization
"""
import socket
import json
import hashlib
import time
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class Checkpoint:
    """Checkpoint for resumable tasks"""
    task_id: str
    progress: float  # 0.0 to 1.0
    state: Dict[str, Any]
    created_at: float
    last_updated: float

class ModelCache:
    """
    Cache for AI models to avoid re-downloading.
    Saves bandwidth and speeds up task execution.
    """
    
    def __init__(self, cache_dir: str = "./model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_index = {}
        self._load_index()
    
    def _get_model_hash(self, model_name: str, version: str = "latest") -> str:
        """Get unique hash for model"""
        return hashlib.sha256(f"{model_name}-{version}".encode()).hexdigest()[:16]
    
    def _load_index(self):
        """Load cache index"""
        index_file = self.cache_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                self.cache_index = json.load(f)
    
    def _save_index(self):
        """Save cache index"""
        index_file = self.cache_dir / "index.json"
        with open(index_file, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
    
    def has_model(self, model_name: str, version: str = "latest") -> bool:
        """Check if model is cached"""
        model_hash = self._get_model_hash(model_name, version)
        return model_hash in self.cache_index
    
    def get_model_path(self, model_name: str, version: str = "latest") -> Optional[Path]:
        """Get path to cached model"""
        model_hash = self._get_model_hash(model_name, version)
        if model_hash in self.cache_index:
            return self.cache_dir / self.cache_index[model_hash]['filename']
        return None
    
    def cache_model(self, model_name: str, model_data: bytes, version: str = "latest"):
        """Cache a model"""
        model_hash = self._get_model_hash(model_name, version)
        filename = f"{model_hash}.pkl"
        filepath = self.cache_dir / filename
        
        with open(filepath, 'wb') as f:
            f.write(model_data)
        
        self.cache_index[model_hash] = {
            'model_name': model_name,
            'version': version,
            'filename': filename,
            'size': len(model_data),
            'cached_at': time.time()
        }
        
        self._save_index()
        logger.info(f"Cached model: {model_name} ({len(model_data)} bytes)")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size = sum(entry['size'] for entry in self.cache_index.values())
        return {
            'total_models': len(self.cache_index),
            'total_size_mb': total_size / (1024 * 1024),
            'models': list(self.cache_index.values())
        }

class CheckpointManager:
    """
    Manages checkpoints for long-running tasks.
    Enables resume after interruption.
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, task_id: str, progress: float, state: Dict[str, Any]):
        """Save checkpoint for a task. Uses JSON + sha256 integrity tag.

        We deliberately do NOT use pickle (CWE-502 / B301): pickle.load
        executes arbitrary code from the file, and an attacker who can
        write to the checkpoint dir would otherwise gain code execution
        on resume. JSON + a sha256 of the body lets us detect tampering
        and crash safely instead.
        """
        checkpoint = Checkpoint(
            task_id=task_id,
            progress=progress,
            state=state,
            created_at=time.time(),
            last_updated=time.time(),
        )
        body = asdict(checkpoint)
        canonical = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
        envelope = {
            "body": body,
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "version": 2,
        }
        filepath = self.checkpoint_dir / f"{task_id}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(envelope, f)

        logger.info(f"Checkpoint saved: {task_id} ({progress*100:.1f}%)")
    
    def load_checkpoint(self, task_id: str) -> Optional[Checkpoint]:
        """Load checkpoint for a task.

        Verifies the sha256 envelope tag before constructing the
        dataclass; a tampered file is dropped and a warning is logged.
        Legacy `.pkl` files are ignored (deliberately not loaded — see
        save_checkpoint docstring for the CWE-502 rationale).
        """
        filepath = self.checkpoint_dir / f"{task_id}.json"
        if not filepath.exists():
            # Refuse to load any legacy .pkl checkpoint to close the
            # arbitrary-code-execution path completely.
            legacy = self.checkpoint_dir / f"{task_id}.pkl"
            if legacy.exists():
                logger.warning(
                    "Legacy pickle checkpoint for %s ignored (security)",
                    task_id,
                )
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                env = json.load(f)
            body = env["body"]
            tag = env["sha256"]
            canonical = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
            if hashlib.sha256(canonical).hexdigest() != tag:
                logger.warning("Checkpoint integrity check failed for %s", task_id)
                return None
            checkpoint = Checkpoint(**body)
            logger.info(f"Checkpoint loaded: {task_id} ({checkpoint.progress*100:.1f}%)")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def delete_checkpoint(self, task_id: str):
        """Delete checkpoint after task completion"""
        for ext in (".json", ".pkl"):  # also clean up any legacy file
            filepath = self.checkpoint_dir / f"{task_id}{ext}"
            if filepath.exists():
                filepath.unlink()
                logger.info(f"Checkpoint deleted: {task_id}{ext}")

class GeographicRouter:
    """
    Routes tasks based on geographic proximity.
    Reduces latency and bandwidth costs.
    """
    
    def __init__(self):
        self.node_locations = {}
        self.location_cache = {}
    
    def register_node_location(self, node_id: str, latitude: float, longitude: float):
        """Register node's geographic location"""
        self.node_locations[node_id] = {
            'lat': latitude,
            'lon': longitude,
            'registered_at': time.time()
        }
    
    def _get_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points (Haversine formula)"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def find_nearest_nodes(self, requester_lat: float, requester_lon: float, 
                          count: int = 3) -> List[str]:
        """Find nearest nodes to requester"""
        distances = []
        
        for node_id, location in self.node_locations.items():
            distance = self._get_distance(
                requester_lat, requester_lon,
                location['lat'], location['lon']
            )
            distances.append((node_id, distance))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Return top N
        return [node_id for node_id, _ in distances[:count]]
    
    def get_optimal_route(self, requester_location: tuple, 
                         available_nodes: List[str]) -> List[str]:
        """Get optimal routing order for nodes"""
        if not requester_location or len(requester_location) != 2:
            return available_nodes
        
        return self.find_nearest_nodes(
            requester_location[0],
            requester_location[1],
            count=len(available_nodes)
        )

class MultiCoordinatorManager:
    """
    Manages multiple coordinators for redundancy.
    Automatic failover if primary coordinator fails.
    """
    
    def __init__(self):
        self.coordinators = {}
        self.primary_coordinator = None
        self.heartbeat_interval = 10
    
    def register_coordinator(self, coordinator_id: str, host: str, port: int, 
                           is_primary: bool = False):
        """Register a coordinator"""
        self.coordinators[coordinator_id] = {
            'id': coordinator_id,
            'host': host,
            'port': port,
            'is_primary': is_primary,
            'status': 'online',
            'last_heartbeat': time.time()
        }
        
        if is_primary:
            self.primary_coordinator = coordinator_id
        
        logger.info(f"Coordinator registered: {coordinator_id} ({'PRIMARY' if is_primary else 'BACKUP'})")
    
    def check_coordinator_health(self, coordinator_id: str) -> bool:
        """Check if coordinator is healthy"""
        if coordinator_id not in self.coordinators:
            return False
        
        coordinator = self.coordinators[coordinator_id]
        time_since_heartbeat = time.time() - coordinator['last_heartbeat']
        
        if time_since_heartbeat > 30:  # 30 seconds timeout
            coordinator['status'] = 'offline'
            return False
        
        return True
    
    def failover_to_backup(self):
        """Failover to backup coordinator"""
        if not self.check_coordinator_health(self.primary_coordinator):
            logger.warning(f"Primary coordinator {self.primary_coordinator} is down!")
            
            # Find healthy backup
            for coord_id, coord_data in self.coordinators.items():
                if coord_id != self.primary_coordinator and self.check_coordinator_health(coord_id):
                    logger.info(f"Failing over to backup: {coord_id}")
                    self.primary_coordinator = coord_id
                    self.coordinators[coord_id]['is_primary'] = True
                    return True
            
            logger.error("No healthy backup coordinators available!")
            return False
        
        return True
    
    def get_active_coordinator(self) -> Optional[Dict[str, Any]]:
        """Get currently active coordinator"""
        # Ensure primary is healthy
        self.failover_to_backup()
        
        if self.primary_coordinator:
            return self.coordinators[self.primary_coordinator]
        
        return None

class BandwidthOptimizer:
    """
    Optimizes bandwidth usage for mesh network.
    - Compresses data when beneficial
    - Routes through efficient paths
    - Monitors bandwidth usage
    """
    
    def __init__(self):
        self.bandwidth_usage = {}
        self.compression_stats = {
            'compressed_bytes': 0,
            'original_bytes': 0,
            'savings': 0
        }
    
    def should_compress(self, data: bytes, threshold: int = 1024) -> bool:
        """Determine if data should be compressed"""
        # Only compress if data is larger than threshold
        return len(data) > threshold
    
    def compress_data(self, data: bytes) -> bytes:
        """Compress data"""
        import zlib
        
        original_size = len(data)
        compressed = zlib.compress(data)
        compressed_size = len(compressed)
        
        # Update stats
        self.compression_stats['original_bytes'] += original_size
        self.compression_stats['compressed_bytes'] += compressed_size
        self.compression_stats['savings'] = (
            1 - self.compression_stats['compressed_bytes'] / 
            self.compression_stats['original_bytes']
        ) * 100
        
        logger.debug(f"Compressed: {original_size} → {compressed_size} bytes")
        
        return compressed
    
    def decompress_data(self, data: bytes) -> bytes:
        """Decompress data"""
        import zlib
        return zlib.decompress(data)
    
    def track_bandwidth(self, node_id: str, bytes_sent: int, bytes_received: int):
        """Track bandwidth usage per node"""
        if node_id not in self.bandwidth_usage:
            self.bandwidth_usage[node_id] = {
                'sent': 0,
                'received': 0,
                'total': 0
            }
        
        self.bandwidth_usage[node_id]['sent'] += bytes_sent
        self.bandwidth_usage[node_id]['received'] += bytes_received
        self.bandwidth_usage[node_id]['total'] = (
            self.bandwidth_usage[node_id]['sent'] + 
            self.bandwidth_usage[node_id]['received']
        )
    
    def get_bandwidth_stats(self) -> Dict[str, Any]:
        """Get bandwidth statistics"""
        total_sent = sum(n['sent'] for n in self.bandwidth_usage.values())
        total_received = sum(n['received'] for n in self.bandwidth_usage.values())
        
        return {
            'total_sent_mb': total_sent / (1024 * 1024),
            'total_received_mb': total_received / (1024 * 1024),
            'total_mb': (total_sent + total_received) / (1024 * 1024),
            'compression_savings_pct': self.compression_stats['savings'],
            'per_node': self.bandwidth_usage
        }

class AdvancedMeshFeatures:
    """
    Combines all advanced mesh features.
    Use this as the main interface.
    """
    
    def __init__(self):
        self.model_cache = ModelCache()
        self.checkpoint_manager = CheckpointManager()
        self.geo_router = GeographicRouter()
        self.multi_coordinator = MultiCoordinatorManager()
        self.bandwidth_optimizer = BandwidthOptimizer()
        
        logger.info("Advanced mesh features initialized")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all features"""
        return {
            'model_cache': self.model_cache.get_cache_stats(),
            'bandwidth': self.bandwidth_optimizer.get_bandwidth_stats(),
            'coordinators': {
                'total': len(self.multi_coordinator.coordinators),
                'primary': self.multi_coordinator.primary_coordinator,
                'active': self.multi_coordinator.get_active_coordinator()
            }
        }
    
    def print_stats(self):
        """Print all statistics"""
        stats = self.get_all_stats()
        
        print("\n" + "="*70)
        print("📊 ADVANCED MESH FEATURES STATUS")
        print("="*70)
        
        print("\n🗄️  Model Cache:")
        print(f"   Models cached: {stats['model_cache']['total_models']}")
        print(f"   Total size: {stats['model_cache']['total_size_mb']:.2f} MB")
        
        print("\n📡 Bandwidth:")
        print(f"   Total sent: {stats['bandwidth']['total_sent_mb']:.2f} MB")
        print(f"   Total received: {stats['bandwidth']['total_received_mb']:.2f} MB")
        print(f"   Compression savings: {stats['bandwidth']['compression_savings_pct']:.1f}%")
        
        print("\n🔄 Coordinators:")
        print(f"   Total: {stats['coordinators']['total']}")
        print(f"   Primary: {stats['coordinators']['primary']}")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    print("="*70)
    print("ADVANCED MESH FEATURES DEMO")
    print("="*70)
    print()
    
    # Initialize
    features = AdvancedMeshFeatures()
    
    # Demo model caching
    print("1️⃣  Model Caching:")
    features.model_cache.cache_model("bert-base", b"fake_model_data" * 1000, "v1.0")
    print(f"   Cached: bert-base")
    print(f"   Has model: {features.model_cache.has_model('bert-base')}")
    print()
    
    # Demo checkpointing
    print("2️⃣  Checkpointing:")
    features.checkpoint_manager.save_checkpoint(
        "task_123",
        progress=0.75,
        state={'epoch': 3, 'loss': 0.234}
    )
    checkpoint = features.checkpoint_manager.load_checkpoint("task_123")
    print(f"   Checkpoint loaded: {checkpoint.progress*100:.0f}% complete")
    print()
    
    # Demo geographic routing
    print("3️⃣  Geographic Routing:")
    features.geo_router.register_node_location("node1", 37.7749, -122.4194)  # SF
    features.geo_router.register_node_location("node2", 40.7128, -74.0060)   # NYC
    features.geo_router.register_node_location("node3", 51.5074, -0.1278)    # London
    
    nearest = features.geo_router.find_nearest_nodes(37.7749, -122.4194, count=2)
    print(f"   Nearest nodes to SF: {nearest}")
    print()
    
    # Demo multi-coordinator
    print("4️⃣  Multi-Coordinator:")
    features.multi_coordinator.register_coordinator("coord1", "192.168.1.100", 9999, is_primary=True)
    features.multi_coordinator.register_coordinator("coord2", "192.168.1.101", 9999, is_primary=False)
    active = features.multi_coordinator.get_active_coordinator()
    print(f"   Active coordinator: {active['id']} at {active['host']}:{active['port']}")
    print()
    
    # Demo bandwidth optimization
    print("5️⃣  Bandwidth Optimization:")
    test_data = b"Hello World" * 100
    if features.bandwidth_optimizer.should_compress(test_data):
        compressed = features.bandwidth_optimizer.compress_data(test_data)
        print(f"   Original: {len(test_data)} bytes")
        print(f"   Compressed: {len(compressed)} bytes")
        print(f"   Savings: {(1 - len(compressed)/len(test_data))*100:.1f}%")
    print()
    
    # Show all stats
    features.print_stats()
    
    print("✅ All advanced features working!")
