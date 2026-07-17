import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Sanity Check: Starting Hardware Detector Test...")

try:
    from core.hardware_detector import HardwareDetector
    
    print("Initializing Detector...")
    start = time.time()
    detector = HardwareDetector()
    print(f"Init took {time.time()-start:.4f}s")
    
    print("Running detect_all_devices()...")
    start = time.time()
    devices = detector.detect_all_devices()
    print(f"\nDetection took {time.time()-start:.4f}s")
    
    print("\nDevices Found:")
    for d in devices:
        print(f" - {d['name']} ({d['type']})")
        
    print("\nBest Device:")
    start = time.time()
    best = detector.get_best_device()
    print(f" {best['name']} ({best['type']})")
    print(f"Selection took {time.time()-start:.4f}s")
    
    print("\nChecking PyTorch Device:")
    import torch
    print(f"Torch CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Torch CUDA Device: {torch.cuda.get_device_name(0)}")
        
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
