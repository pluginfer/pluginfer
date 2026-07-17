"""
Comprehensive System Audit
"Pre-Flight Check" that verifies every component before the final build.
"""
import sys
import os
import json
import logging
import importlib

# Add parent dir to path so we can import 'core', 'plugins', etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup Logging to FILE
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler("audit_errors.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AUDIT")

def audit_system():
    print("\n" + "="*60)
    print(" COMPREHENSIVE SYSTEM AUDIT")
    print("="*60)
    
    all_passed = True
    
    # Check 1: Core Modules Import
    # ---------------------------------------------------------
    print("\n[1] Verifying Core Modules...")
    core_modules = [
        'core.complete_mesh_controller',
        'core.security_manager',
        'core.ai_sentinel',
        'core.hardware_detector',
        'core.wasm_executor',
        'core.plugin_registry',
        'core.inference_engine',
        'core.discovery',
        'core.payments' # Optional
    ]
    
    for mod in core_modules:
        try:
            importlib.import_module(mod)
            print(f"    Imported {mod}")
        except Exception as e:
            print(f"    FAILED to import {mod}: {e}")
            all_passed = False

    # Check 2: Class Instantiation (Wiring Check)
    # ---------------------------------------------------------
    print("\n[2] Verifying Component Initialization...")
    
    try:
        from core.hardware_detector import HardwareDetector
        hw = HardwareDetector()
        print(f"    HardwareDetector: OK (Detected: {hw.get_best_device()['name']})")
    except Exception as e:
        print(f"    HardwareDetector: FAILED ({e})")
        all_passed = False

    try:
        from core.ai_sentinel import AISentinel
        sentinel = AISentinel()
        print(f"    AI Sentinel: OK (Sensitivity: {sentinel.sensitivity})")
    except Exception as e:
        print(f"    AI Sentinel: FAILED ({e})")
        all_passed = False

    try:
        from core.wasm_executor import WasmExecutor
        wasm = WasmExecutor()
        if wasm.is_ready():
            print(f"    WASM Engine: OK (Runtime Available)")
        else:
            print(f"    WAENING: WASM Engine: Instantiated but runtime missing/disabled")
    except Exception as e:
        print(f"    WASM Engine: FAILED ({e})")
        all_passed = False
        
    try:
        from core.complete_mesh_controller import CompleteMeshController
        # Dry run init
        ctrl = CompleteMeshController(host='127.0.0.1', port=12345, mode='audit')
        print(f"    Mesh Controller: OK (ID: {ctrl.node_id})")
        ctrl.stop()
    except Exception as e:
        print(f"    Mesh Controller: FAILED ({e})")
        all_passed = False

    # Check 3: External Dependencies (AI)
    # ---------------------------------------------------------
    print("\n[3] Verifying AI Engines...")
    try:
        import torch
        print(f"    PyTorch: {torch.__version__}")
    except ImportError:
        print("    PyTorch: MISSING")
        all_passed = False
        
    try:
        import wasmtime
        print("    Wasmtime: OK")
    except ImportError:
        print("    Wasmtime: MISSING")
        all_passed = False

    # Check 4: Plugin Ecosystem
    # ---------------------------------------------------------
    print("\n[4] Verifying Plugins...")
    try:
        from core.plugin_registry import PluginRegistry
        reg = PluginRegistry(plugin_dir="../plugins")
        count = reg.discover_plugins()
        print(f"    Discovered {count} plugins")
        
        # Verify specific important ones exist
        if count < 10:
             print("    WARNING: Low plugin count. Expected >15.")
    except Exception as e:
        print(f"    Plugin Registry: FAILED ({e})")
        all_passed = False


    print("\n" + "="*60)
    if all_passed:
        print("AUDIT PASSED: System is Ready for Launch")
        return True
    else:
        print("AUDIT FAILED: Fix errors before building.")
        return False

if __name__ == "__main__":
    success = audit_system()
    sys.exit(0 if success else 1)
