"""
System Doctor (Self-Healing Module)
Continuously monitors system health and automatically fixes issues.
"""
import os
import sys
import time
import socket
import logging
import threading
import shutil
import subprocess
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SystemDoctor:
    """
    The System Doctor ensures the node stays healthy and connected.
    It runs in the background and performs diagnostics and repairs.
    """
    
    def __init__(self, mesh_controller=None):
        self.mesh_controller = mesh_controller
        self.running = False
        self.health_status = {
            'network': 'unknown',
            'disk': 'unknown',
            'services': 'unknown',
            'last_check': 0,
            'issues': []
        }
        self.repair_history = []
        
        # Configuration
        self.CHECK_INTERVAL = 60  # Check every minute
        self.MIN_DISK_GB = 1.0    # Minimum free disk space
    
    def start(self):
        """Start the Doctor service"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("System Doctor started 🩺")
        
    def stop(self):
        """Stop the Doctor service"""
        self.running = False
        logger.info("System Doctor stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self.run_diagnostics()
                
                # If critical issues found, attempt repair
                if self.health_status['issues']:
                    self._attempt_repairs()
                    
            except Exception as e:
                logger.error(f"Doctor crashed during rounds: {e}")
                
            time.sleep(self.CHECK_INTERVAL)
            
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run full system diagnostics"""
        issues = []
        
        # 1. Network Check
        network_ok = self._check_internet()
        self.health_status['network'] = 'healthy' if network_ok else 'unhealthy'
        if not network_ok:
            issues.append('no_internet')
            
        # 2. Disk Check
        disk_ok = self._check_disk()
        self.health_status['disk'] = 'healthy' if disk_ok else 'critical'
        if not disk_ok:
            issues.append('low_disk_space')
            
        # 3. Service Check
        services_ok = self._check_services()
        self.health_status['services'] = 'running' if services_ok else 'stalled'
        if not services_ok:
            issues.append('service_stall')
            
        self.health_status['issues'] = issues
        self.health_status['last_check'] = time.time()
        
        return self.health_status
    
    def _check_internet(self, host="8.8.8.8", port=53, timeout=3):
        """Check internet connectivity"""
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except Exception:
            return False
            
    def check_python_version(self):
        """Ensure we are running on a supported Python version (3.8 - 3.12)"""
        import sys
        v = sys.version_info
        
        # WARN/FAIL for Python 3.14 (Verified Incompatible with Torch)
        if v.major == 3 and v.minor >= 14:
             logger.error("CRITICAL: Python 3.14 detected. PyTorch/DirectML does strictly NOT support 3.14 yet.")
             logger.error("Please install Python 3.10 or 3.11 and rebuild/restart.")
             return False
             
        # Recommend 3.10/3.11
        if v.minor not in [10, 11]:
            logger.warning(f"Running on Python {v.major}.{v.minor}. Recommended: 3.10 or 3.11 for best ML compatibility.")
            
        return True

    def _check_disk(self):
        """Check available disk space"""
        try:
            total, used, free = shutil.disk_usage("/")
            free_gb = free // (2**30)
            return free_gb >= self.MIN_DISK_GB
        except Exception:
            return True # Assume ok if check fails
            
    def _check_services(self):
        """Check if Mesh Controller is responsive"""
        if not self.mesh_controller:
            return True # Skip if not attached
            
        # Check if threads are alive
        if not self.mesh_controller.running:
            return False
            
        return True
        
    def _attempt_repairs(self):
        """Attempt to fix detected issues"""
        for issue in self.health_status['issues']:
            logger.warning(f"Doctor attempting to fix: {issue}")
            
            if issue == 'no_internet':
                self.fix_network()
            elif issue == 'low_disk_space':
                self.cleanup_disk()
            elif issue == 'service_stall':
                self.restart_services()
                
            self.repair_history.append({
                'timestamp': time.time(),
                'issue': issue,
                'action': 'attempted_repair'
            })
            
    # --- REPAIR PROCEDURES ---
    
    def fix_network(self):
        """Attempt to fix network issues"""
        logger.info("🩹 Doctor: Flushing DNS and checking connection...")
        try:
            # Platform specific dns flush
            if sys.platform == "win32":
                subprocess.run(["ipconfig", "/flushdns"], capture_output=True)
            elif sys.platform == "darwin":
                 subprocess.run(["dscacheutil", "-flushcache"], capture_output=True)
                 
            # Re-check
            if self._check_internet():
                logger.info("✅ Doctor: Internet restored!")
            else:
                logger.warning("❌ Doctor: Could not auto-fix internet.")
        except Exception as e:
            logger.error(f"Doctor network fix failed: {e}")
            
    def cleanup_disk(self):
        """Cleanup temporary files to free space"""
        logger.info("🩹 Doctor: Cleaning up temporary cache...")
        # Placeholder for actual cache cleanup
        # e.g. delete /tmp/pluginfer_cache/*
        pass
        
    def restart_services(self):
        """Restart stalled services"""
        logger.info("🩹 Doctor: Restarting mesh services...")
        if self.mesh_controller:
            # Soft restart logic would go here
            pass
            
    def get_report(self):
        """Get a human-readable report"""
        status_icon = "✅" if not self.health_status['issues'] else "⚠️"
        return {
            'status': status_icon,
            'summary': "System Healthy" if not self.health_status['issues'] else f"Issues: {', '.join(self.health_status['issues'])}",
            'details': self.health_status,
            'repairs_count': len(self.repair_history)
        }
