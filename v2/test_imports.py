#!/usr/bin/env python
# Test individual core imports to find the hanging one
import sys
sys.path.insert(0, '.')

print("Testing imports...")
print("1. plugin_base...", end='', flush=True)
from core.plugin_base import PluginBase
print(" OK")

print("2. plugin_registry...", end='', flush=True)
from core.plugin_registry import PluginRegistry
print(" OK")

print("3. inference_engine...", end='', flush=True)
from core.inference_engine import InferenceEngine
print(" OK")

print("4. hardware_detector...", end='', flush=True)
from core.hardware_detector import HardwareDetector
print(" OK")

print("5. qal_controller...", end='', flush=True)
from core.qal_controller import QALController
print(" OK")

print("6. license_validator...", end='', flush=True)
from core.license_validator import LicenseValidator, LicenseTier
print(" OK")

print("7. mesh_controller...", end='', flush=True)
from core.mesh_controller import MeshNetworkController
print(" OK")

print("8. complete_mesh_controller...", end='', flush=True)
from core.complete_mesh_controller import CompleteMeshController
print(" OK")

print("\nAll imports successful!")
