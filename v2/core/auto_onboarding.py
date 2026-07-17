"""
Automatic Onboarding & Dynamic Pricing System
- Zero-config onboarding
- Dynamic pricing based on demand
- Revenue sharing (70% to users, 30% to platform)
- Peak hour pricing
- Automatic marketplace
"""
import os
import time
import json
import hashlib
import socket
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
# Lazy / safe imports. Previously these came from `core.mesh_controller`
# which is now archived under `core/_archive_v2/`. Replaced with the
# canonical v3 surfaces: HardwareDetector + MeshDiscovery.
try:
    from .discovery import MeshDiscovery as _MeshDiscovery
except ImportError:
    _MeshDiscovery = None  # auto_onboard still works in offline mode

try:
    from .hardware_detector import HardwareDetector as _HardwareDetector
except ImportError:
    _HardwareDetector = None

logger = logging.getLogger(__name__)


def _detect_capabilities() -> Dict[str, Any]:
    """
    Replacement for the archived `core.mesh_controller.get_system_capabilities`.
    Returns a flat dict with shape:
        {
          'has_gpu': bool,
          'gpu_type': 'cuda' | 'mps' | 'rocm' | 'cpu',
          'gpu_name': str,
          'cpu_name': str,
          'devices': [ raw HardwareDetector entries ],
        }
    """
    if _HardwareDetector is None:
        return {'has_gpu': False, 'gpu_type': 'cpu', 'gpu_name': '',
                'cpu_name': 'Unknown', 'devices': []}
    try:
        det = _HardwareDetector()
        devs = det.detect_all_devices() or []
    except Exception as e:
        logger.warning("hardware detection failed: %s", e)
        return {'has_gpu': False, 'gpu_type': 'cpu', 'gpu_name': '',
                'cpu_name': 'Unknown', 'devices': []}
    cpu = next((d for d in devs if d.get('type') == 'cpu'), {})
    gpu_priority = ('cuda', 'mps', 'rocm')
    gpu = next(
        (d for d in devs for t in gpu_priority if d.get('type') == t),
        None,
    )
    return {
        'has_gpu': gpu is not None,
        'gpu_type': gpu.get('type') if gpu else 'cpu',
        'gpu_name': gpu.get('name', '') if gpu else '',
        'cpu_name': cpu.get('name', 'Unknown'),
        'devices': devs,
    }


def _find_coordinator(timeout: float = 3.0,
                      swarm_id: str = "public") -> Optional[Dict[str, Any]]:
    """
    Replacement for the archived `core.mesh_controller.find_coordinator`.
    Uses MeshDiscovery.search_for_coordinator under the hood.
    """
    if _MeshDiscovery is None:
        return None
    try:
        node_id = hashlib.sha256(
            f"client-{socket.gethostname()}-{time.time()}".encode()
        ).hexdigest()[:16]
        # Capabilities aren't needed just to listen for coordinator beacons;
        # pass an empty dict.
        disc = _MeshDiscovery(node_id=node_id, port=0, capabilities={},
                              swarm_id=swarm_id)
        return disc.search_for_coordinator(timeout=int(timeout))
    except Exception as e:
        logger.warning("coordinator discovery failed: %s", e)
        return None

@dataclass
class UserProfile:
    """Automatically generated user profile"""
    user_id: str
    node_id: str
    hardware: Dict[str, Any]
    joined_at: float
    total_earned: float = 0.0
    tasks_completed: int = 0
    reputation_score: float = 100.0
    availability_hours: Optional[List[int]] = None  # Hours available (0-23)
    min_price_per_task: float = 0.01
    
class DynamicPricingEngine:
    """
    Calculates task prices based on:
    - Current demand
    - Node availability
    - Peak hours
    - Task complexity
    - Network congestion
    """
    
    BASE_PRICE = 0.01  # $0.01 base price per task
    PLATFORM_FEE = 0.30  # 30% platform fee (70% to user)
    
    PEAK_HOURS = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # 9 AM - 6 PM
    PEAK_MULTIPLIER = 1.5
    
    def __init__(self):
        self.demand_history = []
        self.current_demand = 0
        self.available_supply = 0
    
    def calculate_price(self, task_complexity: str = "medium",
                       is_urgent: bool = False,
                       current_hour: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate dynamic price for a task.
        
        Returns dict with:
        - total_price: What customer pays
        - user_earnings: What user gets (70%)
        - platform_fee: What platform gets (30%)
        """
        # Base price
        price = self.BASE_PRICE
        
        # Complexity multiplier
        complexity_multipliers = {
            'low': 0.5,
            'medium': 1.0,
            'high': 2.0,
            'extreme': 5.0
        }
        price *= complexity_multipliers.get(task_complexity, 1.0)
        
        # Peak hour multiplier
        hour = current_hour if current_hour is not None else datetime.now().hour
        if hour in self.PEAK_HOURS:
            price *= self.PEAK_MULTIPLIER
        
        # Demand multiplier
        if self.available_supply > 0:
            demand_ratio = self.current_demand / self.available_supply
            if demand_ratio > 1.5:  # High demand
                price *= 1.3
            elif demand_ratio > 2.0:  # Very high demand
                price *= 1.6
        
        # Urgency multiplier
        if is_urgent:
            price *= 1.5
        
        # Calculate split
        total_price = price
        platform_fee = price * self.PLATFORM_FEE
        user_earnings = price * (1 - self.PLATFORM_FEE)
        
        return {
            'total_price': round(total_price, 4),
            'user_earnings': round(user_earnings, 4),
            'platform_fee': round(platform_fee, 4),
            'breakdown': {
                'base': self.BASE_PRICE,
                'complexity': task_complexity,
                'is_peak_hour': hour in self.PEAK_HOURS,
                'demand_multiplier': demand_ratio if self.available_supply > 0 else 1.0,
                'is_urgent': is_urgent
            }
        }
    
    def update_demand(self, pending_tasks: int, available_nodes: int):
        """Update current demand metrics"""
        self.current_demand = pending_tasks
        self.available_supply = available_nodes
        
        self.demand_history.append({
            'timestamp': time.time(),
            'demand': pending_tasks,
            'supply': available_nodes
        })
        
        # Keep only last hour
        cutoff = time.time() - 3600
        self.demand_history = [
            h for h in self.demand_history 
            if h['timestamp'] > cutoff
        ]
    
    def get_current_pricing(self) -> Dict[str, Any]:
        """Get current pricing for all task types"""
        hour = datetime.now().hour
        
        return {
            'timestamp': datetime.now().isoformat(),
            'is_peak_hour': hour in self.PEAK_HOURS,
            'demand_level': 'high' if self.current_demand > self.available_supply * 1.5 else 'normal',
            'prices': {
                'low_complexity': self.calculate_price('low'),
                'medium_complexity': self.calculate_price('medium'),
                'high_complexity': self.calculate_price('high'),
                'urgent': self.calculate_price('medium', is_urgent=True)
            }
        }

class AutoOnboardingSystem:
    """
    Automatically onboards users with ZERO configuration.
    Detects hardware, creates profile, connects to mesh.
    """
    
    def __init__(self, data_dir: str = "./user_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.users = {}
        self._load_users()
    
    def _load_users(self):
        """Load existing users"""
        users_file = self.data_dir / "users.json"
        if users_file.exists():
            with open(users_file, 'r') as f:
                data = json.load(f)
                self.users = {
                    uid: UserProfile(**profile)
                    for uid, profile in data.items()
                }
    
    def _save_users(self):
        """Save users"""
        users_file = self.data_dir / "users.json"
        with open(users_file, 'w') as f:
            json.dump(
                {uid: asdict(profile) for uid, profile in self.users.items()},
                f,
                indent=2
            )
    
    def auto_onboard(self, capabilities: Optional[Dict[str, Any]] = None,
                     consent_network: bool = True,
                     consent_compute: bool = True,
                     verbose: bool = False) -> Optional[UserProfile]:
        """
        Non-interactive first-launch flow (W19 rewrite).

        What this DOES:
          1. Detects hardware (HardwareDetector).
          2. Derives a stable node_id from the hostname + a random
             salt (so reruns on the same machine give the same id).
          3. Persists a UserProfile to data_dir/users.json.

        What it does NOT do:
          * Print fabricated earnings projections. The previous version
            multiplied a hardcoded "10x for NVIDIA / 7x for Apple Silicon"
            by "100 tasks per hour" by "20 hours idle per day" by 30 —
            none of those numbers were measured on this machine; showing
            them as $/month put us in consumer-protection territory.
            Earnings can only be reported empirically, after the node has
            actually completed paid tasks.
          * Block on input(). Caller passes consent flags up front;
            "ZERO CONFIGURATION" must mean exactly that.
          * Promise an `earnings_url` that doesn't exist.

        Returns the persisted UserProfile, or None if consent missing.
        """
        if not consent_network or not consent_compute:
            logger.info("auto_onboard: consent not granted; skipped.")
            return None

        if not capabilities:
            capabilities = _detect_capabilities()

        # Stable node_id: hash(hostname + os user + salt-from-data-dir).
        # The salt is a one-time random value persisted to data_dir/.salt
        # so reruns on the same machine produce the same id, but two
        # parallel installs (different data_dir) get different ids.
        salt_path = self.data_dir / ".salt"
        if salt_path.exists():
            salt = salt_path.read_text(encoding="utf-8").strip()
        else:
            salt = hashlib.sha256(
                f"{socket.gethostname()}-{time.time()}-{os.urandom(16)}".encode()
            ).hexdigest()
            salt_path.write_text(salt, encoding="utf-8")

        identity_seed = f"{socket.gethostname()}|{os.environ.get('USER','')}|{salt}"
        identity_hash = hashlib.sha256(identity_seed.encode()).hexdigest()
        user_id = identity_hash[:12]
        node_id = identity_hash[:16]

        profile = self.users.get(user_id) or UserProfile(
            user_id=user_id, node_id=node_id, hardware=capabilities,
            joined_at=time.time(),
        )
        # Refresh the hardware snapshot on every onboard.
        profile.hardware = capabilities
        self.users[user_id] = profile
        self._save_users()

        if verbose:
            print("Pluginfer onboarding:")
            print(f"  user_id  : {user_id}")
            print(f"  node_id  : {node_id}")
            print(f"  hardware : {capabilities.get('gpu_type','cpu')} "
                  f"({capabilities.get('gpu_name') or capabilities.get('cpu_name')})")

        return profile

    def quick_start(self,
                    coordinator_host: str = "auto",
                    consent_network: bool = True,
                    consent_compute: bool = True,
                    verbose: bool = False) -> Dict[str, Any]:
        """
        One-call setup. Non-interactive.

        Returns a structured status dict — caller (the launcher /
        dashboard) decides how to render it. Never prints fabricated
        numbers.
        """
        profile = self.auto_onboard(
            consent_network=consent_network,
            consent_compute=consent_compute,
            verbose=verbose,
        )
        if profile is None:
            return {"status": "no_consent"}

        if coordinator_host == "auto":
            coordinator_host = self._discover_coordinator()

        return {
            "status": "online" if coordinator_host else "offline",
            "user_id": profile.user_id,
            "node_id": profile.node_id,
            "hardware": profile.hardware,
            "coordinator": coordinator_host,
            "joined_at": profile.joined_at,
            "notes": (
                "Earnings are not estimated. Once the node completes "
                "paid tasks, the dashboard will show empirical earnings "
                "from chain receipts."
            ),
        }
    
    def _discover_coordinator(self) -> Optional[str]:
        """
        Auto-discover a coordinator. Returns host string or None.
        Order:
          1. LAN UDP discovery via MeshDiscovery (3 s timeout)
          2. localhost (developer / single-machine path)
        Production DHT bootstrap (signed seed pubkeys) is W19 — until
        then this function returns None when LAN/localhost both fail.
        It does NOT silently fall back to a fictitious 'pluginfer.network'.
        """
        print("      Scanning local network for coordinator (UDP, 3 s)...")
        info = _find_coordinator(timeout=3)
        if info:
            host = info.get('ip') or info.get('host')
            if host:
                logger.info("Coordinator found via UDP at %s", host)
                return host

        # localhost fallback (single-machine / developer path)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            try:
                if sock.connect_ex(('127.0.0.1', 9999)) == 0:
                    logger.info("Coordinator found on localhost:9999")
                    return '127.0.0.1'
            finally:
                sock.close()
        except OSError:
            pass

        logger.info(
            "No coordinator found via LAN or localhost. "
            "DHT bootstrap (signed seed pubkeys) is tracked as W19; "
            "until then auto_onboard remains in offline mode."
        )
        return None

class MarketplaceSystem:
    """
    Automatic marketplace for buying/selling compute.
    Users can rent or provide compute with zero configuration.
    """
    
    def __init__(self):
        self.pricing = DynamicPricingEngine()
        self.onboarding = AutoOnboardingSystem()
        self.active_rentals = {}
    
    def rent_compute(self, num_gpus: int, hours: int,
                     complexity: str = "medium") -> Dict[str, Any]:
        """
        Rent compute power. NOT IMPLEMENTED in v3.0-alpha.

        The previous version returned a *mock* receipt with a fake
        rental_id, fake api_key (`pk_{rental_id}`) and fake
        access_url (`http://mesh.pluginfer.com/rental/{...}`) — all
        wired to nothing. A user who paid against this mock would have
        gotten back nothing.

        The real implementation is gated on:
          * core.broker.EconomicBroker matching the rental against
            available peer nodes (W15: provider auction);
          * an L1 lock-and-mint of the rental fee + on-chain
            settlement at expiry;
          * a dispatcher that routes the renter's jobs to those
            peers and aggregates results.

        Until that protocol lands we refuse to issue a receipt rather
        than ship one that doesn't redeem.
        """
        raise NotImplementedError(
            "MarketplaceSystem.rent_compute is not implemented in "
            "v3.0-alpha. The previous mock returned a fake receipt with "
            "no backing compute. See core/auto_onboarding.py docstring "
            "and TODO W15 (provider auction) + W26 for the production "
            "design."
        )

    def provide_compute(self,
                        consent_network: bool = True,
                        consent_compute: bool = True) -> Dict[str, Any]:
        """Start providing compute (non-interactive, no fabricated $)."""
        return self.onboarding.quick_start(
            consent_network=consent_network,
            consent_compute=consent_compute,
        )
    
    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get current marketplace statistics"""
        pricing = self.pricing.get_current_pricing()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'available_nodes': self.pricing.available_supply,
            'pending_tasks': self.pricing.current_demand,
            'current_prices': pricing['prices'],
            'is_peak_hour': pricing['is_peak_hour'],
            'active_rentals': len(self.active_rentals),
            'revenue_share': {
                'user': '70%',
                'platform': '30%'
            }
        }


if __name__ == "__main__":
    # Non-interactive demo. The previous version had `input()` calls
    # between sections that hung any non-TTY run.
    print("Pluginfer auto-onboarding demo (non-interactive)")
    print("-" * 60)
    onboarding = AutoOnboardingSystem()
    result = onboarding.quick_start(verbose=True)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("-" * 60)

    # Pricing engine: shows what a task WOULD cost under the
    # configured complexity/peak/demand assumptions. NOT a quote;
    # not a guarantee.
    pricing = DynamicPricingEngine()
    current = pricing.get_current_pricing()
    print(f"Pricing snapshot @ {current['timestamp']}:")
    for task_type, price_info in current["prices"].items():
        print(f"  {task_type}: total=${price_info['total_price']:.4f} "
              f"user=${price_info['user_earnings']:.4f}")
    print("-" * 60)
    print("Marketplace (rent_compute) is not implemented in v3.0-alpha; "
          "see W15/W26.")

# Removed: old-demo block that called rent_compute (now raises) and
# referenced fabricated "Marketplace Stats" with hardcoded
# "Revenue share: 70% to users" claims. Replaced by the
# non-interactive demo above (`if __name__ == "__main__"`).
