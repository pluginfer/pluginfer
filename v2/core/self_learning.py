"""
Self-Learning AI Optimizer
Uses Reinforcement Learning (RL) concepts to optimize mesh performance over time.
- Learns user activity patterns (Predictive Scaling)
- Learns task affinity (Which tasks run best on this node?)
- Optimizes earnings vs power consumption
"""
import time
import json
import logging
import math
import psutil
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ActivityWindow:
    hour: int
    task_count: int = 0
    total_earnings: float = 0.0

class SelfLearningOptimizer:
    def __init__(self, data_dir: str = "./user_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.history_file = self.data_dir / "learning_history.json"
        
        # Knowledge Base
        self.hourly_activity: Dict[int, ActivityWindow] = {} # 0-23 hours
        self.task_affinity: Dict[str, float] = {} # plugin_name -> speed_multiplier
        self.optimization_score: float = 0.0 # How much have we improved?
        
        self._load_history()

    def monitor_resources(self) -> Dict[str, float]:
        """
        [HEALER] Monitor System Health
        Returns CPU and RAM usage to detect overload.
        """
        try:
            return {
                "cpu_usage": psutil.cpu_percent(interval=0.1),
                "ram_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
        except Exception as e:
            logger.error(f"Healer Resource Check Failed: {e}")
            return {"cpu_usage": 0.0, "ram_usage": 0.0}

    # Track per-plugin running mean of duration so affinity = (mean / observed)
    # is a proper relative-speed score. Faster than mean -> affinity > 1.
    def record_task_completion(self, plugin_name: str, duration: float, earnings: float):
        """Learn from a completed task — real running-mean affinity update."""
        # 1. Hourly pattern.
        current_hour = time.localtime().tm_hour
        if current_hour not in self.hourly_activity:
            self.hourly_activity[current_hour] = ActivityWindow(hour=current_hour)
        stats = self.hourly_activity[current_hour]
        stats.task_count += 1
        stats.total_earnings += earnings

        # 2. Affinity = exponential moving average of (mean_duration / this_duration).
        # If we're faster than our historical mean, ratio > 1 -> affinity grows.
        if not hasattr(self, "_duration_means"):
            self._duration_means: Dict[str, float] = {}
        prev_mean = self._duration_means.get(plugin_name)
        if prev_mean is None or prev_mean <= 0:
            self._duration_means[plugin_name] = max(duration, 1e-3)
            new_affinity = 1.0
        else:
            ratio = prev_mean / max(duration, 1e-3)
            current_affinity = self.task_affinity.get(plugin_name, 1.0)
            new_affinity = current_affinity * 0.9 + ratio * 0.1
            # update running mean
            self._duration_means[plugin_name] = prev_mean * 0.95 + duration * 0.05

        self.task_affinity[plugin_name] = new_affinity
        self._save_history()
        logger.info("Affinity[%s]=%.3f (this dur=%.3fs, mean=%.3fs)",
                    plugin_name, new_affinity, duration,
                    self._duration_means.get(plugin_name, duration))

    def predict_usage(self, next_hours: int = 1) -> float:
        """
        Predict probability of high load in next N hours.
        Return 0.0 to 1.0
        """
        current_hour = time.localtime().tm_hour
        probability = 0.0
        
        for h in range(current_hour, current_hour + next_hours):
            hour_idx = h % 24
            if hour_idx in self.hourly_activity:
                # Normalize count (simple heuristic)
                count = self.hourly_activity[hour_idx].task_count
                if count > 5: probability += 0.3
                if count > 20: probability += 0.5
        
        return min(1.0, probability)

    def improve_scheduling(self, pending_tasks: List[Any]) -> List[Any]:
        """
        Reorder tasks based on Affinity (Do what we are good at first).
        """
        # Sort tasks by affinity score descending
        return sorted(pending_tasks, 
                     key=lambda t: self.task_affinity.get(t.plugin_name, 1.0), 
                     reverse=True)

    def get_optimization_suggestions(self) -> List[str]:
        """
        Suggest actions to the user/system based on learning.
        """
        suggestions = []
        
        # pattern recognition
        prob = self.predict_usage(2)
        if prob > 0.8:
            suggestions.append("PRE_WARM_GPU") # High load expected
            suggestions.append("DOWNLOAD_CACHE")
            
        return suggestions

    def monitor_health(self) -> Dict[str, Any]:
        """
        [GOLD STANDARD] Ops Agent (The Healer)
        Monitors system vitals (CPU, RAM).
        Returns an Action Plan if health is critical.
        """
        try:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            
            action = "OK"
            reason = ""
            
            # 1. Critical Overload -> Throttle
            if cpu > 90 or ram > 90:
                action = "THROTTLE"
                reason = f"Critical Load (CPU: {cpu}%, RAM: {ram}%)"
            
            # 2. Dangerous RAM -> Restart Required
            if ram > 95:
                action = "EMERGENCY_RESTART"
                reason = f"Memory Leak Detected (RAM: {ram}%)"
                
            return {
                'cpu': cpu,
                'ram': ram,
                'action': action,
                'reason': reason
            }
        except Exception as e:
            logger.error(f"[HEALER] Monitor failed: {e}")
            return {'action': 'ERROR', 'error': str(e)}

    def _load_history(self):
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    # Restore dicts
                    self.task_affinity = data.get('affinity', {})
                    # Restore activity windows
                    raw_activity = data.get('activity', {})
                    for h_str, info in raw_activity.items():
                        self.hourly_activity[int(h_str)] = ActivityWindow(
                            hour=int(h_str),
                            task_count=info['c'],
                            total_earnings=info['e']
                        )
            except Exception as e:
                logger.error(f"Failed to load AI history: {e}")

    def _save_history(self):
        try:
            # Serialize
            data = {
                'affinity': self.task_affinity,
                'activity': {
                    h: {'c': w.task_count, 'e': w.total_earnings}
                    for h, w in self.hourly_activity.items()
                }
            }
            with open(self.history_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save AI history: {e}")
