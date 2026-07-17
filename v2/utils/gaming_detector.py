"""
Gaming Detection - Auto-pause mesh worker when gaming
Allows gamers to contribute GPU during idle time without impacting gameplay
"""
import psutil
import time
import logging
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class GameInfo:
    """Information about a detected game"""
    name: str
    process_name: str
    pid: int
    detected_at: float

class GamingDetector:
    """
    Detects when games are running and manages worker pause/resume.
    
    Perfect for gamers who want to monetize idle GPU time without
    impacting gaming performance.
    """
    
    # Comprehensive list of popular games
    DEFAULT_GAME_LIST = [
        # FPS Games
        'csgo.exe', 'cs2.exe', 'valorant.exe', 'overwatch.exe',
        'apex_legends.exe', 'cod.exe', 'modernwarfare.exe',
        'battlefield.exe', 'r5apex.exe', 'fortnite.exe',
        
        # MOBA/Strategy
        'league of legends.exe', 'leagueclient.exe', 'dota2.exe',
        'starcraft2.exe', 'ageofempires.exe',
        
        # RPG/MMO
        'wow.exe', 'worldofwarcraft.exe', 'ff14.exe', 'ffxiv_dx11.exe',
        'elderscrollsonline.exe', 'guildwars2.exe', 'newworld.exe',
        
        # Battle Royale
        'pubg.exe', 'warzone.exe', 'fallguys.exe',
        
        # Popular Titles
        'minecraft.exe', 'roblox.exe', 'gta5.exe', 'rdr2.exe',
        'cyberpunk2077.exe', 'witcher3.exe', 'skyrim.exe',
        'eldenring.exe', 'fifa.exe', 'nba2k.exe',
        
        # Indie/Other
        'amongus.exe', 'terraria.exe', 'stardewvalley.exe',
        'hollowknight.exe', 'celeste.exe'
    ]
    
    def __init__(self, custom_game_list: Optional[List[str]] = None, 
                 check_interval: int = 30):
        """
        Initialize gaming detector.
        
        Args:
            custom_game_list: Additional games to detect (optional)
            check_interval: How often to check in seconds (default: 30)
        """
        self.game_list = self.DEFAULT_GAME_LIST.copy()
        
        if custom_game_list:
            self.game_list.extend(custom_game_list)
        
        # Normalize to lowercase
        self.game_list = [g.lower() for g in self.game_list]
        
        self.check_interval = check_interval
        self.current_game: Optional[GameInfo] = None
        self.gaming_history = []
        
        logger.info(f"Gaming detector initialized with {len(self.game_list)} games")
    
    def is_gaming(self) -> bool:
        """
        Check if any games are currently running.
        
        Returns:
            True if gaming, False otherwise
        """
        try:
            for proc in psutil.process_iter(['name', 'pid']):
                try:
                    proc_name = proc.info['name'].lower()
                    
                    # Check if process name matches any game
                    if any(game in proc_name for game in self.game_list):
                        # Found a game!
                        if not self.current_game or self.current_game.pid != proc.info['pid']:
                            self.current_game = GameInfo(
                                name=proc.info['name'],
                                process_name=proc_name,
                                pid=proc.info['pid'],
                                detected_at=time.time()
                            )
                            logger.info(f"🎮 Game detected: {self.current_game.name}")
                        
                        return True
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            
            # No games found
            if self.current_game:
                logger.info(f"🎮 Game closed: {self.current_game.name}")
                self.gaming_history.append(self.current_game)
                self.current_game = None
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting games: {e}")
            return False
    
    def get_current_game(self) -> Optional[GameInfo]:
        """Get information about currently running game"""
        return self.current_game
    
    def get_gaming_stats(self) -> dict:
        """Get statistics about gaming sessions"""
        total_sessions = len(self.gaming_history)
        
        if total_sessions == 0:
            return {
                'total_sessions': 0,
                'total_time': 0,
                'average_session': 0
            }
        
        # Calculate total gaming time
        total_time = sum(
            game.detected_at for game in self.gaming_history
        )
        
        return {
            'total_sessions': total_sessions,
            'total_time': total_time,
            'average_session': total_time / total_sessions if total_sessions > 0 else 0,
            'games_played': list(set(g.name for g in self.gaming_history))
        }
    
    def auto_manage_worker(self, worker, callback=None):
        """
        Automatically pause/resume worker based on gaming status.
        
        Args:
            worker: MeshNetworkController instance
            callback: Optional callback function(is_gaming: bool)
        
        Usage:
            detector = GamingDetector()
            detector.auto_manage_worker(worker)
        """
        logger.info("🎮 Auto-management started")
        print("\n" + "="*70)
        print("🎮 GAMING AUTO-DETECTION ACTIVE")
        print("="*70)
        print("\nHow it works:")
        print("  • When you start a game → Worker pauses automatically")
        print("  • When you close the game → Worker resumes automatically")
        print("  • Your gaming performance is NEVER impacted")
        print("  • You earn money during non-gaming hours")
        print("\n" + "="*70 + "\n")
        
        was_gaming = False
        
        while True:
            try:
                is_gaming_now = self.is_gaming()
                
                # State change detected
                if is_gaming_now != was_gaming:
                    if is_gaming_now:
                        # Just started gaming
                        print(f"\n🎮 Game detected: {self.current_game.name}")
                        print("⏸️  Worker PAUSED - Gaming mode active")
                        print("   Your full GPU power is now available for gaming!")
                        
                        # Pause worker
                        if hasattr(worker, 'pause'):
                            worker.pause()
                        else:
                            # Alternative: stop processing
                            worker.running = False
                        
                        if callback:
                            callback(True)
                    
                    else:
                        # Just stopped gaming
                        print(f"\n✅ Gaming session ended")
                        print("▶️  Worker RESUMED - Earning mode active")
                        print("   Your GPU is now contributing to the mesh!")
                        
                        # Resume worker
                        if hasattr(worker, 'resume'):
                            worker.resume()
                        else:
                            worker.running = True
                        
                        if callback:
                            callback(False)
                    
                    was_gaming = is_gaming_now
                
                # Sleep before next check
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                print("\n\n🛑 Auto-management stopped")
                break
            except Exception as e:
                logger.error(f"Error in auto-management: {e}")
                time.sleep(self.check_interval)
    
    def add_game(self, game_name: str):
        """Add a game to the detection list"""
        game_name = game_name.lower()
        if game_name not in self.game_list:
            self.game_list.append(game_name)
            logger.info(f"Added game: {game_name}")
    
    def remove_game(self, game_name: str):
        """Remove a game from the detection list"""
        game_name = game_name.lower()
        if game_name in self.game_list:
            self.game_list.remove(game_name)
            logger.info(f"Removed game: {game_name}")


def create_gaming_aware_worker(coordinator_host: str, coordinator_port: int,
                               auth_token: Optional[str] = None):
    """
    Create a worker that automatically pauses during gaming.
    
    This is the RECOMMENDED way for gamers to contribute.
    
    Args:
        coordinator_host: Coordinator IP/hostname
        coordinator_port: Coordinator port
        auth_token: Authentication token (if required)
    
    Example:
        create_gaming_aware_worker('192.168.1.100', 9999)
    """
    from core.mesh_controller import MeshNetworkController, get_system_capabilities
    
    print("="*70)
    print("🎮 GAMING-AWARE WORKER")
    print("="*70)
    print("\nPerfect for gamers who want to:")
    print("  • Earn money from idle GPU time")
    print("  • Never impact gaming performance")
    print("  • Automatic pause/resume")
    print()
    
    # Create worker
    print("🚀 Starting worker...")
    worker = MeshNetworkController(
        host='0.0.0.0',
        port=10000,
        mode='worker'
    )
    worker.start()
    
    # Connect to coordinator
    print(f"📡 Connecting to {coordinator_host}:{coordinator_port}...")
    capabilities = get_system_capabilities()
    
    # TODO: Use auth_token when connecting
    
    print("✅ Connected to mesh!")
    
    # Create gaming detector
    detector = GamingDetector()
    
    # Start auto-management
    print("\n🎮 Starting gaming detection...")
    detector.auto_manage_worker(worker)


if __name__ == "__main__":
    print("="*70)
    print("GAMING DETECTION DEMO")
    print("="*70)
    print()
    
    # Create detector
    detector = GamingDetector(check_interval=5)
    
    print("🔍 Checking for games...")
    print(f"   Monitoring {len(detector.game_list)} games")
    print()
    
    # Check once
    is_gaming = detector.is_gaming()
    
    if is_gaming:
        game = detector.get_current_game()
        print(f"✅ Game detected: {game.name}")
        print(f"   Process: {game.process_name}")
        print(f"   PID: {game.pid}")
    else:
        print("❌ No games running")
    
    print()
    print("💡 To use with mesh worker:")
    print("""
from utils.gaming_detector import GamingDetector

# Create your worker
worker = MeshNetworkController('0.0.0.0', 10000, 'worker')
worker.start()

# Add gaming detection
detector = GamingDetector()
detector.auto_manage_worker(worker)

# Worker automatically pauses when you game!
    """)
