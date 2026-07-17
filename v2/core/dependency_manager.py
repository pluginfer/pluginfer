"""
Dependency Manager
Handles resolution of external binaries (like FFmpeg) bundled with the application.
"""
import sys
import os
import shutil
import platform
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

class DependencyManager:
    @staticmethod
    def ensure_accelerators():
        """
        Auto-installs hardware acceleration libraries based on OS.
        - Windows: directml
        - Linux: intel_extension_for_pytorch (xpu)
        """
        system = platform.system()
        
        # 1. Windows: DirectML
        if system == 'Windows':
            # Check for Python 3.13+ (DirectML not fully supported yet)
            if sys.version_info >= (3, 13):
                 logger.info(f"[INFO] Python {sys.version_info.major}.{sys.version_info.minor} Detected. Using Stable CPU Mode.")
                 return

            try:
                import torch_directml
                logger.debug("Dependency check: torch-directml is available.")
            except ImportError:
                # In frozen mode, missing directml is expected if not explicitly bundled or supported.
                # We simply fallback to CPU or CUDA without complaining to the user.
                logger.debug("Dependency check: 'torch-directml' not found. DirectML acceleration unavailable.")
                pass
            except Exception:
                pass

        # 2. Linux: Intel OneAPI (XPU)
        elif system == 'Linux':
            # Heuristic: Check if Intel GPU is present via lspci (if available) or just try import if intended
            # For now, we only auto-install if we detect we are likely on an Intel platform 
            # OR simply logging that the user should install it, as IPEX installation is complex.
            # IPEX usually requires specific index-urls, making auto-pip risky.
            pass

    @staticmethod
    def get_ffmpeg_path() -> str:
        """
        Get the absolute path to the FFmpeg executable.
        1. Checks PyInstaller bundle path (_MEIPASS).
        2. Checks local 'bin' directory.
        3. Checks system PATH.
        """
        executable_name = "ffmpeg.exe" if os.name == 'nt' else "ffmpeg"
        
        # 1. Check Bundled (PyInstaller)
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            bundled_path = os.path.join(sys._MEIPASS, 'bin', executable_name)
            if os.path.exists(bundled_path):
                logger.debug(f"Found bundled FFmpeg: {bundled_path}")
                return bundled_path
                
        # 2. Check Local 'bin' folder (Dev Mode)
        # Assuming we are in core/, so go up one level
        base_dir = Path(__file__).parent.parent.absolute()
        local_bin = base_dir / "bin" / executable_name
        if local_bin.exists():
            logger.debug(f"Found local FFmpeg: {local_bin}")
            return str(local_bin)
            
        # 3. System PATH
        system_path = shutil.which('ffmpeg')
        if system_path:
            logger.debug(f"Found system FFmpeg: {system_path}")
            return system_path
            
        logger.warning("FFmpeg not found in bundle, local bin, or PATH.")
        return None
