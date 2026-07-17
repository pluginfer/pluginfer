"""
Fix Plugin Imports
Bulk replacer to fix the erroneous relative import in generated plugins.
"""
import os
import glob

def fix_imports():
    plugin_dir = os.path.abspath("../plugins")
    print(f"Scanning {plugin_dir}...")
    
    files = glob.glob(os.path.join(plugin_dir, "*.py"))
    count = 0
    
    for file_path in files:
        if "__init__" in file_path:
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # The Bug: "from .plugin_base import PluginBase"
        # The Fix: "from core.plugin_base import PluginBase"
        
        # 1. Fix Import
        if "from .plugin_base import PluginBase" in content:
            content = content.replace("from .plugin_base import PluginBase", "from core.plugin_base import PluginBase")
            is_modified = True
            
        # 2. Inject sys.path (The fix for dynamic loading)
        sys_path_code = "import sys\nsys.path.insert(0, '..')\n"
        
        if "sys.path.insert(0, '..')" not in content:
            # Insert after docstring or at top
            if '"""' in content[:10]:
                # Find end of docstring
                parts = content.split('"""', 2)
                if len(parts) >= 3:
                     # parts[0] is empty, parts[1] is doctring, parts[2] is code
                     content = '"""' + parts[1] + '"""\n' + sys_path_code + parts[2]
                     is_modified = True
            else:
                 content = sys_path_code + content
                 is_modified = True
                 
        if is_modified:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Patched {os.path.basename(file_path)}")
            count += 1
            
    print(f"\nTotal fixed: {count} files.")

if __name__ == "__main__":
    fix_imports()
