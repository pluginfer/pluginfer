"""
Verification Script for Gaming Mode
"""
import sys
import os
import time
import logging
import threading
from unittest.mock import MagicMock, patch

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.game_detector import GameDetector
from core.mesh_controller import MeshNetworkController

logging.basicConfig(level=logging.INFO)

def verify_gaming_mode():
    print("\n[Test] Gaming Mode Auto-Pause...")
    
    # 1. Setup Mock Controller
    mock_controller = MagicMock()
    detector = GameDetector(mock_controller)
    detector.check_interval = 0.1 # Fast checks
    
    # 2. Start Detector
    detector.start()
    time.sleep(0.2)
    
    # 3. Simulate Game Launch (Mocking psutil)
    print("   Simulating 'valorant.exe' launch...")
    
    # Mock psutil.process_iter to return a game process
    mock_proc = MagicMock()
    mock_proc.info = {'name': 'valorant.exe'}
    
    with patch('psutil.process_iter', return_value=[mock_proc]):
        time.sleep(0.2) # Wait for detector cycle
        
    # Verify Pause called
    if mock_controller.pause_computation.called:
        print("✅ PAUSE triggered successfully!")
    else:
        print("❌ PAUSE NOT triggered!")
        return False
        
    # 4. Simulate Game Close (Empty process list)
    print("   Simulating game exit...")
    with patch('psutil.process_iter', return_value=[]):
        time.sleep(0.2)
        
    # Verify Resume called
    if mock_controller.resume_computation.called:
        print("✅ RESUME triggered successfully!")
    else:
        print("❌ RESUME NOT triggered!")
        return False
        
    detector.stop()
    return True

if __name__ == "__main__":
    if verify_gaming_mode():
        print("\n🎉 Gaming Mode Verification PASSED")
        sys.exit(0)
    else:
        print("\n💥 Gaming Mode Verification FAILED")
        sys.exit(1)
