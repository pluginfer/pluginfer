"""
AI Security Sentinel
Statistical Anomaly Detection System to prevent hacking.
Optimized for LOW FALSE POSITIVES (never block a human).
"""
import time
import math
import logging
import ast
from typing import Dict, List, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ClientProfile:
    """Statistical profile of a client connection"""
    ip: str
    request_counts: List[float] = field(default_factory=list)
    payload_sizes: List[float] = field(default_factory=list)
    error_counts: int = 0
    start_time: float = 0.0
    threat_score: float = 0.0
    
    def __post_init__(self):
        self.start_time = time.time()

class AISentinel:
    """
    The 'Brain' that watches network traffic.
    Uses Z-Score analysis to find statistical outliers (hackers).
    """
    
    def __init__(self, sensitivity: float = 3.5):
        self.profiles: Dict[str, ClientProfile] = {}
        # Sensitivity (Standard Deviations). 
        # 3.5 = Very Conservative (Only blocks obvious attacks)
        # 2.0 = Aggressive (Might block humans)
        self.sensitivity = sensitivity 
        self.banned_ips = set()
        
        logger.info(f"AI Sentinel Active (Sensitivity: {self.sensitivity} sigma)")

    def scan_code(self, source_code: str) -> bool:
        """
        [GOLD STANDARD] Static Analysis (AST)
        Reads the code to find malicious intent BEFORE execution.
        Returns False if malicious code is detected.
        """
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                # 1. Detect Imports of Dangerous Libraries
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if self._is_banned_module(alias.name):
                            logger.error(f"[SENTINEL] Blocked import: {alias.name}")
                            return False
                if isinstance(node, ast.ImportFrom):
                    if self._is_banned_module(node.module):
                        logger.error(f"[SENTINEL] Blocked import from: {node.module}")
                        return False
                        
                # 2. Detect Function Calls (e.g., eval, exec, open)
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                         if node.func.id in ['eval', 'exec', 'compile', 'open']:
                             logger.error(f"[SENTINEL] Blocked dangerous function: {node.func.id}")
                             return False
                             
            return True
        except SyntaxError:
            logger.error("[SENTINEL] Code is not valid Python (SyntaxError)")
            return False
        except Exception as e:
            logger.error(f"[SENTINEL] Analysis failed: {e}")
            return False # Fail safe

    def _is_banned_module(self, module_name: str) -> bool:
        banned = ['os', 'sys', 'subprocess', 'shutil', 'socket', 'requests', 'urllib', 'http']
        if not module_name: return False
        return module_name.split('.')[0] in banned

    def analyze_request(self, client_id: str, payload_size: int, is_error: bool = False) -> bool:
        """
        Analyze a single request.
        Returns False if request should be BLOCKED.
        Returns True if request is ALLOWED.
        """
        # 1. Check Ban List
        if client_id in self.banned_ips:
            return False

        # Get or Create Profile
        if client_id not in self.profiles:
            self.profiles[client_id] = ClientProfile(ip=client_id)
            
        profile = self.profiles[client_id]
        now = time.time()
        
        # 2. Update Stats
        profile.request_counts.append(now)
        profile.payload_sizes.append(payload_size)
        if is_error:
            profile.error_counts += 1
            
        # Clean old history (last 60s)
        profile.request_counts = [t for t in profile.request_counts if now - t < 60]
        
        # 3. Learning Phase (Grace Period)
        # We need at least 10 requests to establish a "Normal Baseline"
        if len(profile.request_counts) < 10:
            return True
            
        # 4. Anomaly Detection (The AI Logic)
        
        # Feature A: Request Frequency (DDoS Check - BURST AWARE)
        # Calculate rate over the actual duration of the history buffer
        if len(profile.request_counts) > 20:
             duration = profile.request_counts[-1] - profile.request_counts[0]
             if duration < 1.0: duration = 1.0 # Avoid div zero
             
             req_per_sec = len(profile.request_counts) / duration
             
             if req_per_sec > 20: # Limit: 20 req/sec (Human max is ~5)
                self._ban_client(client_id, f"Rate Anomaly ({req_per_sec:.1f} req/s)")
                return False
            
        # Feature B: Error Rate (Fuzzing/Exploit Check)
        if profile.error_counts > 5 and len(profile.request_counts) > 0:
            error_rate = profile.error_counts / len(profile.request_counts)
            if error_rate > 0.8: # 80% errors? They are attacking.
                self._ban_client(client_id, "High Error Rate (Fuzzing Attempt)")
                return False

        # Feature C: Payload Size Outlier (Buffer Overflow Check)
        # Calculate Mean and StdDev of their previous payloads (Historical Baseline)
        if len(profile.payload_sizes) > 20:
            # Exclude the current newest item from the baseline calculation
            history = profile.payload_sizes[:-1]
            
            avg_size = sum(history) / len(history)
            variance = sum([((x - avg_size) ** 2) for x in history]) / len(history)
            std_dev = math.sqrt(variance)
            
            if std_dev > 0:
                z_score = (payload_size - avg_size) / std_dev
                

                # If current payload is 4 standard deviations larger than normal
                if z_score > 4.0 and payload_size > 1024*1024: 
                    # AND it's physically large (>1MB)
                    logger.warning(f"Anomaly detected for {client_id}: Payload Z-Score {z_score:.2f}")
                    # We don't ban immediately for size, just warn
                    profile.threat_score += 20
        
        # 5. Final Decision
        if profile.threat_score >= 100:
            self._ban_client(client_id, "Cumulative Threat Score exceeded")
            return False
            
        return True

    def _ban_client(self, client_id: str, reason: str):
        """Ban a client permanently"""
        logger.error(f"SECURITY ALERT: Banning {client_id}. Reason: {reason}")
        self.banned_ips.add(client_id)

    def get_stats(self):
        """Get stats for dashboard"""
        return {
            'active_profiles': len(self.profiles),
            'banned_count': len(self.banned_ips),
            'sensitivity': self.sensitivity
        }
