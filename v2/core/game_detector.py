"""
Game Detector Module
Automatically pauses mesh computation when games are detected.
"""
import time
import logging
import threading
import psutil
from typing import List, Set

logger = logging.getLogger(__name__)

# Common game executables (expandable)
KNOWN_GAMES = {
    "valorant", "fortniteclient-win64-shipping", "csgo", "cs2",
    "dota2", "league of legends", "overwatch",
    "minecraft", "gta5", "rdr2", "cyberpunk2077",
    "eldenring", "cod", "apex", "robloxplayerbeta"
}

class GameDetector:
    """
    Monitors system for running games and manages MeshController pause state.
    """
    
    def __init__(self, mesh_controller):
        self.controller = mesh_controller
        self.running = False
        self.check_interval = 5.0 # Check every 5 seconds
        self.is_gaming = False
        self.known_games = KNOWN_GAMES
        
    def start(self):
        """Start the game detection service"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("🎮 Game Detector started")
        
    def stop(self):
        """Stop the game detection service"""
        self.running = False
        logger.info("Game Detector stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                found_game = self._check_for_games()
                
                if found_game and not self.is_gaming:
                    # Game started
                    self.is_gaming = True
                    logger.info(f"🎮 Game Detected: {found_game}")
                    if self.controller:
                        self.controller.pause_computation()
                        
                elif not found_game and self.is_gaming:
                    # Game stopped
                    self.is_gaming = False
                    logger.info("🎮 Game closed. Resuming work...")
                    if self.controller:
                        self.controller.resume_computation()
                        
            except Exception as e:
                logger.error(f"Game Detector error: {e}")
                
            time.sleep(self.check_interval)
            
    def _check_for_games(self) -> str:
        """
        Check running processes for known games.
        Returns the name of the first game found, or None.
        """
        # Optimized check: iterate process names
        # Note: psutil.process_iter is heavy, so we limit attributes
        try:
            for proc in psutil.process_iter(['name']):
                try:
                    # Normalize: Lowercase AND strip .exe
                    raw_name = proc.info['name'].lower()
                    
                    # OS-Agnostic Check:
                    # 'valorant.exe' -> 'valorant'
                    # 'dota2' -> 'dota2'
                    clean_name = raw_name.replace('.exe', '')
                    
                    if clean_name in self.known_games or raw_name in self.known_games:
                        return raw_name
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
        except Exception:
            pass
            
        return None
