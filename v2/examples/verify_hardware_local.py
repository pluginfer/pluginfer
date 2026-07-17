"""
Local Hardware Verification
Prints the exact hardware specs detected by Pluginfer on this machine.
"""
import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from core.hardware_detector import HardwareDetector
import platform
import psutil

def verify_local_hardware():
    print("="*60)
    print("HARDWARE DETECTION REPORT")
    print("="*60)
    
    # 1. Initialize Detector
    detector = HardwareDetector()
    
    # 2. Get Device Info
    device = detector.get_best_device()
    
    print(f"\n[SYSTEM IDENTITY]")
    print(f"  OS System:    {platform.system()}")
    print(f"  OS Release:   {platform.release()}")
    print(f"  Machine:      {platform.machine()}")
    print(f"  Processor:    {platform.processor()}")
    
    print(f"\n[CPU SPECS]")
    # Using psutil directly to cross-verify what our detector might abstract
    print(f"  Physical Cores: {psutil.cpu_count(logical=False)}")
    print(f"  Logical Cores:  {psutil.cpu_count(logical=True)}")
    print(f"  Total RAM:      {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    print(f"\n[PLUGINFER DETECTED COMPUTE]")
    print(f"  Best Device:    {device['name']}")
    print(f"  Device Type:    {device['type']}")
    print(f"  Est. Memory:    {device.get('memory', 'N/A')}")
    
    print("\n" + "="*60)
    if device['type'] == 'cpu' and platform.machine():
        print("✅ SUCCESS: Hardware successfully identified.")
    elif device['type'] != 'cpu':
        print("✅ SUCCESS: Hardware successfully identified (Accelerator Found).")
    else:
        print("❌ FAILURE: Could not identify hardware.")

if __name__ == "__main__":
    verify_local_hardware()
