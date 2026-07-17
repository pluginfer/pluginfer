import time
import sys
import os
import shutil
import threading
import logging

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.complete_mesh_controller import CompleteMeshController
from core.game_detector import GameDetector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestGamingMode")

def test_gaming_mode():
    print("\n" + "="*70)
    print("🎮 TEST: Gaming Mode (Auto-Pause)")
    print("="*70)
    
    # 1. Setup Dummy Game Executable
    game_exe = "VALORANT.exe"
    if not os.path.exists(game_exe):
        shutil.copy(sys.executable, game_exe)
        print(f"[+] Created dummy game: {game_exe}")
        
    # 2. Start Worker Node
    # Note: GameDetector needs the controller to call pause on
    node = CompleteMeshController('127.0.0.1', 10005, 'worker')
    node.start()
    
    # Ensure GameDetector is running (it starts in node.start())
    print("[+] Node Started. Game Detector Active.")
    time.sleep(2)
    
    # 3. Verify Initial State
    if node.is_paused:
        print("❌ ERROR: Node started in PAUSED state!")
        return
    print("✅ Node is RUNNING (Idle)")
    
    # 4. Launch Game
    print(f"[*] Launching {game_exe}...")
    import subprocess
    game_proc = subprocess.Popen([game_exe, "-c", "import time; time.sleep(10)"])
    
    # 5. Wait for Detection (Check interval is 5s)
    print("[*] Waiting for detection (max 10s)...")
    detected = False
    for _ in range(10):
        if node.is_paused:
            detected = True
            break
        time.sleep(1)
        
    if detected:
        print("✅ SUCCESS: Node entered PAUSED state!")
    else:
        print("❌ FAILURE: Node did NOT pause.")
        
    # 6. Stop Game
    print("[*] Stopping Game...")
    game_proc.terminate()
    game_proc.wait()
    
    # 7. Wait for Resume
    print("[*] Waiting for resume (max 10s)...")
    resumed = False
    for _ in range(10):
        if not node.is_paused:
            resumed = True
            break
        time.sleep(1)
        
    if resumed:
        print("✅ SUCCESS: Node RESUMED!")
    else:
        print("❌ FAILURE: Node did NOT resume.")

    # Cleanup
    node.stop()
    if os.path.exists(game_exe):
        os.remove(game_exe)
        
if __name__ == "__main__":
    test_gaming_mode()
