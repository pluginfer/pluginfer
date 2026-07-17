"""
Hardware Detection Module
Detects available compute devices (CPU, NVIDIA GPU, AMD GPU, Intel GPU, Apple Silicon)
"""
import platform
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class HardwareDetector:
    """Detects and profiles available compute hardware"""
    
    # Singleton Cache (Class Level)
    _shared_devices = None
    
    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
        # Instance level mirror
        self._devices = None
        
    def detect_all_devices(self) -> List[Dict]:
        """Detect all available compute devices"""
        # Check Shared Cache First
        if HardwareDetector._shared_devices is not None:
             self._devices = HardwareDetector._shared_devices
             return self._devices
             
        if self._devices is not None:
            return self._devices
            
        devices = []
        
        # Always have CPU
        print("   ...Scanning CPU...", flush=True)
        cpu_name = self._get_cpu_name() or "Unknown CPU"
        devices.append({
            'type': 'cpu',
            'name': cpu_name,
            'priority': 10,
            'available': True
        })
        
        # Check for CUDA (NVIDIA)
        print("   ...Scanning CUDA...", flush=True)
        cuda_device = self._check_cuda()
        if cuda_device:
            devices.append(cuda_device)
            
        # Check for MPS (Apple Silicon)
        mps_device = self._check_mps()
        if mps_device:
            devices.append(mps_device)
            
        # Check for ROCm (AMD)
        rocm_device = self._check_rocm()
        if rocm_device:
            devices.append(rocm_device)
            
        # Check for Intel GPU
        print("   ...Scanning Drivers...", flush=True)
        intel_device = self._check_intel()
        if intel_device:
            devices.append(intel_device)
        
        self._devices = devices
        HardwareDetector._shared_devices = devices # Populate Singleton
        return devices
    
    def get_best_device(self) -> Dict:
        """Get the best available device based on priority"""
        # Call internal method to avoid re-scan if already cached
        devices = self.detect_all_devices()
        # Sort by priority (lower number = higher priority)
        # Note: detect_all_devices caches the result in self._devices
        devices.sort(key=lambda x: x['priority'])
        return devices[0]
    
    def get_performance_score(self) -> float:
        """
        Calculate a relative performance score for routing and billing.
        Base Score (CPU) = 1.0
        """
        best = self.get_best_device()
        dtype = best['type']
        count = best.get('count', 1)
        
        # Tiered Performance Scoring
        # Multipliers roughly approximate FP32 TFLOPS relative to a basic CPU
        # BOOSTED for visual impact (User Expectation Management)
        if dtype == 'cuda':     return 50.0 * count # Scale with multi-GPU!
        if dtype == 'rocm':     return 40.0 * count
        if dtype == 'mps':      return 5.0          # Apple Neural Engine is fast for inference
        if dtype == 'directml': return 3.0          # Intel/Integrated Graphics
        
        # CPU Fallback
        # Check core count to boost strong CPUs (Threadripper/Epyc)
        try:
            import multiprocessing
            cores = multiprocessing.cpu_count()
            if cores > 32: return 4.0   # Server Grade
            if cores > 16: return 2.0   # High End Desktop
        except:
            pass
            
        return 1.0 # Standard Laptop CPU
    
    def get_torch_device(self):
        """Get PyTorch device object for best available hardware"""
        import torch
        
        best = self.get_best_device()
        device_type = best['type']
        
        if device_type == 'cuda' or device_type == 'rocm':
            return torch.device('cuda') # ROCm uses the cuda namespace in PyTorch
        elif device_type == 'mps':
            return torch.device('mps')
        elif device_type == 'xpu':
            try:
                import intel_extension_for_pytorch as ipex
                return torch.device('xpu')
            except ImportError:
                return torch.device('cpu')
        elif device_type == 'directml':
            try:
                import torch_directml
                return torch_directml.device()
            except ImportError:
                # User requested to silence this old warning
                # logger.warning("⚠️ PERFORMANCE WARNING: DirectML capable hardware detected, but 'torch-directml' is not installed.")
                # logger.warning("   -> To unlock 10x-100x speedup, run: `pip install torch-directml`")
                # logger.warning("   -> Falling back to CPU (Slow Mode) for now.")
                return torch.device('cpu')
        else:
            return torch.device('cpu')
    
    def _check_cuda(self) -> Optional[Dict]:
        """Check for NVIDIA CUDA support"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                return {
                    'type': 'cuda',
                    'name': gpu_name,
                    'count': torch.cuda.device_count(),
                    'priority': 1,
                    'available': True
                }
        except Exception as e:
            logger.debug(f"CUDA not available: {e}")
        return None
    
    def _check_mps(self) -> Optional[Dict]:
        """Check for Apple Metal Performance Shaders"""
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return {
                    'type': 'mps',
                    'name': 'Apple Silicon GPU',
                    'priority': 2,
                    'available': True
                }
        except Exception as e:
            logger.debug(f"MPS not available: {e}")
        return None
    
    def _check_rocm(self) -> Optional[Dict]:
        """Check for AMD ROCm support"""
        try:
            import torch
            # ROCm uses the same API as CUDA in PyTorch
            if hasattr(torch.version, 'hip') and torch.version.hip:
                return {
                    'type': 'rocm',
                    'name': 'AMD GPU (ROCm)',
                    'priority': 3,
                    'available': True
                }
        except Exception as e:
            logger.debug(f"ROCm not available: {e}")
        return None
    
    def _check_intel(self) -> Optional[Dict]:
        """Check for Intel/AMD GPU support via DirectML (Windows) or OneAPI (Linux)"""
        try:
            # 1. Windows: Try DirectML (Works for Intel Arc, AMD Radeon, NVIDIA)
            if self.system == 'Windows':
                
                # A. PRIORITY: Check for NVIDIA using nvidia-smi (Fast & Reliable)
                try:
                    # print("   ...Checking nvidia-smi...", flush=True)
                    import subprocess
                    cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
                    
                    # Robust execution: capture output, text mode, timeout, NO INPUT
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=3,
                        stdin=subprocess.DEVNULL, # Critical to prevent hang
                        creationflags=subprocess.CREATE_NO_WINDOW if self.system == 'Windows' else 0
                    )
                    
                    if result.returncode == 0:
                        gpu_name = result.stdout.strip()
                        if gpu_name:
                            return {
                                'type': 'cuda', # User wants it treated as CUDA capable
                                'name': f"{gpu_name} (Driver Detected)",
                                'priority': 1,
                                'available': False # Driver present, software missing
                            }
                except Exception as e:
                     logger.debug(f"nvidia-smi check failed: {e}")
                     pass
            
                # B. Try importing the library (Best Case)
                # try:
                #     # print("   [DEBUG] Importing torch_directml...", flush=True) 
                #     import torch_directml
                #     # print("   [DEBUG] torch_directml imported", flush=True)
                #     dml = torch_directml.device()
                #     return {
                #         'type': 'directml',
                #         'name': f'DirectML GPU (Id: {dml.index})',
                #         'priority': 4,
                #         'available': True
                #     }
                # except ImportError:
                #     pass
                # except Exception as e:
                #     # logger.error(f"   [DEBUG] torch_directml error: {e}")
                #     pass


                # C. Final Fallback: WMI (Only if nvidia-smi missing)
                # Commented out to prevent stress test hangs
                # try:
                #     print("   ...Checking WMI...", flush=True)
                #     ... (Legacy WMI code) ...

            # 2. Linux: Intel OneAPI (XPU)
            if self.system == 'Linux':
                try:
                    import intel_extension_for_pytorch as ipex
                    return {
                        'type': 'xpu',
                        'name': 'Intel Max GPU (XPU)',
                        'priority': 3,
                        'available': True
                    }
                except ImportError:
                    pass

        except Exception as e:
            logger.debug(f"Intel/DirectML check failed: {e}")
        return None
    
    def _get_cpu_name(self) -> str:
        """Get CPU name"""
        try:
            # Fallback to platform.processor for speed/stability
            if self.system == 'Windows':
                 return platform.processor() or "Intel/AMD x64 Processor"
            return f"{self.machine} CPU"
        except:
            return f"{self.machine} CPU"
    
    def print_device_info(self):
        """Print detailed device information"""
        devices = self.detect_all_devices()
        print("\n" + "="*60)
        print("[*] DETECTED COMPUTE DEVICES")
        print("="*60)
        
        for i, device in enumerate(devices, 1):
            status = "[+]" if device['available'] else "[!]"
            print(f"\n{i}. {status} {device['type'].upper()}")
            print(f"   Name: {device['name']}")
            print(f"   Priority: {device['priority']}")
            if 'count' in device:
                print(f"   Count: {device['count']}")
        
        best = self.get_best_device()
        print(f"\n[>] Selected Device: {best['type'].upper()} - {best['name']}")
        print("="*60 + "\n")


if __name__ == "__main__":
    detector = HardwareDetector()
    detector.print_device_info()
