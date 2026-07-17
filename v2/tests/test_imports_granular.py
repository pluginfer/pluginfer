
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

modules = [
    "core.plugin_registry",
    "core.networking",
    "core.inference_engine",
    "core.security_manager",
    "core.hardware_detector",
    "core.dependency_manager",
    "core.advanced_mesh_features",
    "core.ai_sentinel",
    "core.updater",
    "core.self_learning",
    "core.compute_ledger",
    "core.tokenomics",
    "core.discovery"
]

print(f"Testing imports on Python {sys.version}")

for mod in modules:
    print(f"Importing {mod}...", end=" ")
    try:
        __import__(mod)
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
