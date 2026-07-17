import sys
import os
import importlib.util

# Add project root to path
sys.path.append(os.path.abspath('..'))

plugin_path = os.path.abspath("../plugins/img_resize.py")
print(f"Loading {plugin_path}...")

try:
    spec = importlib.util.spec_from_file_location("img_resize", plugin_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("✅ Success!")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
