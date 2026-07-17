"""
Pluginfer Release Builder
Automates the PyInstaller build process for the final .EXE
"""
import os
import sys
import shutil
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Builder")

def build_exe():
    print("\n" + "🧱" * 40)
    print("PLUGINFER BUILD SYSTEM STARTING...")
    print("🧱" * 40)
    
    # Check execution context
    if not os.path.exists("example_mesh_network.py"):
        logger.warning("Please run this script from the 'examples' directory!")
        
    # 1. Environment Check
    try:
        import PyInstaller
        logger.info("✅ PyInstaller detected.")
    except ImportError:
        logger.error("❌ PyInstaller not found!")
        print("\nPlease install PyInstaller to build the EXE:")
        print("   pip install pyinstaller")
        print("\nThen run this script again.")
        return False
        
    # 2. Cleanup
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    if os.path.exists("build"):
        shutil.rmtree("build")
        
    # 3. Define Build Command
    # Entry point: pluginfer_node.py (The "Zero-Touch" Node)
    entry_point = os.path.abspath("pluginfer_node.py")
    
    # Paths relative to examples/
    core_path = os.path.abspath("../core")
    ui_path = os.path.abspath("../ui")
    utils_path = os.path.abspath("../utils")
    plugins_path = os.path.abspath("../plugins")
    
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--onefile",
        # "--windowed", # DISABLED: We want the console to show the connection status!
        "--name", "PluginferNode",
        "--clean",
        # Include paths - Format: source;dest
        f"--add-data", f"{core_path};core",
        f"--add-data", f"{ui_path};ui",
        f"--add-data", f"{utils_path};utils",
        f"--add-data", f"{plugins_path};plugins", # ✅ NEW: Include all 20+ plugins 
        # Hidden imports
        "--hidden-import", "PIL",
        "--hidden-import", "tkinter",
        "--hidden-import", "socket",
        "--hidden-import", "json",
        "--hidden-import", "threading",
        "--hidden-import", "platform",
        "--hidden-import", "uuid",
        "--hidden-import", "hashlib",
        "--hidden-import", "ctypes",
        "--hidden-import", "secrets",
        "--hidden-import", "hmac",
        "--hidden-import", "ssl",
        "--hidden-import", "base64",
        "--hidden-import", "concurrent",          # ✅ FIX: Missing module
        "--hidden-import", "concurrent.futures",  # ✅ FIX: Explicit submodule
        "--hidden-import", "multiprocessing",     # Proactive
        "--hidden-import", "asyncio",             # Proactive
        "--hidden-import", "logging.handlers",    # Proactive
        "--hidden-import", "requests",            # ✅ FIX: Missing module
        "--hidden-import", "psutil",              # ✅ FIX: Missing module for Gaming Detector
        "--hidden-import", "urllib3",             # Dependencies for requests
        "--hidden-import", "idna",
        "--hidden-import", "chardet",
        "--hidden-import", "certifi",
        "--hidden-import", "flask",               # ✅ FIX: Missing module for UI
        "--hidden-import", "werkzeug",            # Flask dep
        "--hidden-import", "jinja2",              # Flask dep
        "--hidden-import", "itsdangerous",        # Flask dep
        "--hidden-import", "click",               # Flask dep
        "--hidden-import", "markupsafe",          # Flask dep
        "--hidden-import", "wasmtime",            # ✅ NEW: WASM Support (Zero-Touch)
        "--hidden-import", "core.wasm_executor",  # ✅ NEW: Our sandbox module
        "--hidden-import", "core.ai_sentinel",    # ✅ NEW: Anti-Hack System
        "--hidden-import", "core.hardware_detector", # ✅ NEW: Dashboard Stats
        "--hidden-import", "torch",               # ✅ NEW: Full AI (PyTorch)
        "--hidden-import", "numpy",               # ✅ NEW: Math Ops
        entry_point
    ]
    
    print("\n🔨 Executing Build Command:")
    print(" ".join(cmd))
    
    # 4. Run Build
    try:
        # We use subprocess calling 'pyinstaller' executable
        subprocess.check_call(cmd)
        
        print("\n" + "✨" * 40)
        print("BUILD SUCCESSFUL!")
        print("✨" * 40)
        print(f"\nFinal EXE is located at: {os.path.abspath('dist/Pluginfer.exe')}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Build failed: {e}")
        return False
    except FileNotFoundError:
        # Fallback if pyinstaller is in path but not callable directly
        logger.warning("Could not find 'pyinstaller' in PATH. Trying python module...")
        try:
             subprocess.check_call([sys.executable, "-m", "PyInstaller"] + cmd[1:])
             return True
        except Exception as e2:
             logger.error(f"Module build failed: {e2}")
             return False

if __name__ == "__main__":
    success = build_exe()
    if not success:
        sys.exit(1)
