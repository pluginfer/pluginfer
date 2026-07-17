
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("[TEST] Importing CompleteMeshController...")
try:
    from core.complete_mesh_controller import CompleteMeshController
    print("✅ Import SUCCESS")
except Exception as e:
    print(f"❌ Import FAILED: {e}")
    import traceback
    traceback.print_exc()
