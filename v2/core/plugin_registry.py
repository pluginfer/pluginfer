"""
Plugin Registry
Manages plugin discovery, loading, and lifecycle
"""
import os
import sys
import importlib.util
import inspect
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from .plugin_base import PluginBase

logger = logging.getLogger(__name__)

class PluginRegistry:
    """
    Central registry for managing plugins.
    Handles plugin discovery, loading, and validation.
    """
    
    def __init__(self, plugin_dir: str = "plugins"):
        if getattr(sys, 'frozen', False):
            # In release mode, plugins are bundled inside the EXE temp folder
            # PyInstaller extracts to sys._MEIPASS
            if hasattr(sys, '_MEIPASS'):
                base_dir = Path(sys._MEIPASS)
                
                # Recursive search for 'plugins' directory
                # This handles _internal, nested folders, etc.
                found_path = None
                for root, dirs, files in os.walk(base_dir):
                    if 'plugins' in dirs:
                        potential_path = Path(root) / 'plugins'
                        # Check if it contains .py files
                        if list(potential_path.glob('*.py')):
                            found_path = potential_path
                            break
                
                if found_path:
                    self.plugin_dir = found_path
                else:
                    logger.warning("Could not automatically locate plugins dir in bundle. Using default.")
                    # Fallbacks
                    self.plugin_dir = base_dir / "plugins"
                    
            logger.info(f"Running in Frozen Mode. Loading plugins from: {self.plugin_dir}")
        else:
            # In dev mode, use the relative path
            self.plugin_dir = Path(plugin_dir)
            
        self._plugins: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
        
    def discover_plugins(self) -> int:
        """
        Discover all plugins in the plugin directory.
        """
        # DEBUG LOGGING TO FILE
        with open("debug_plugins.log", "w") as f:
            f.write(f"=== Plugin Discovery Log ===\n")
            f.write(f"CWD: {os.getcwd()}\n")
            f.write(f"Frozen: {getattr(sys, 'frozen', False)}\n")
            
            if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                f.write(f"MEIPASS: {sys._MEIPASS}\n")
                # Dump MEIPASS structure
                f.write("\n--- MEIPASS Structure ---\n")
                try:
                    for root, dirs, files in os.walk(sys._MEIPASS):
                        for file in files:
                            f.write(f"{os.path.join(root, file)}\n")
                except Exception as e:
                    f.write(f"Error walking MEIPASS: {e}\n")
                f.write("-------------------------\n")
            
        print(f"[DEBUG] Discovering plugins in: {self.plugin_dir}")
        
        if not self.plugin_dir.exists():
            logger.warning(f"Plugin directory not found: {self.plugin_dir}")
            # Try CWD fallback
            cwd_plugins = Path("plugins")
            if cwd_plugins.exists():
                self.plugin_dir = cwd_plugins
                with open("debug_plugins.log", "a") as f: f.write(f"Fallback to CWD: {cwd_plugins}\n")
            else:
                with open("debug_plugins.log", "a") as f: f.write(f"ERROR: No plugin dir found at {self.plugin_dir} or CWD\n")
                return 0
        
        discovered = 0
        
        with open("debug_plugins.log", "a") as f:
            f.write(f"\nScanning: {self.plugin_dir}\n")
            for file_path in self.plugin_dir.glob("*.py"):
                if file_path.name.startswith("__"): continue
                
                try:
                    self._load_plugin_from_file(file_path)
                    discovered += 1
                    # FORCE PRINT for visibility in Installer/Console
                    print(f"   [+] Loaded Plugin: {file_path.stem}")
                    f.write(f"Loaded: {file_path.name}\n")
                except Exception as e:
                    logger.error(f"Failed to load plugin {file_path.name}: {e}")
                    print(f"   [!] Failed to load {file_path.name}: {e}")
                    f.write(f"FAILED: {file_path.name} - {e}\n")
                    
            f.write(f"Total Discovered: {discovered}\n")
        
        self._loaded = True
        logger.info(f"Discovered {discovered} plugins")
        return discovered
    
    def _load_plugin_from_file(self, file_path: Path):
        """Load a plugin from a Python file"""
        module_name = file_path.stem
        
        # ✅ FIX: Ensure project root is in sys.path so 'import core' works
        import sys
        
        # Calculate project root: file_path (plugin) -> parent (plugins dir) -> parent (project root)
        project_root = str(file_path.parent.parent.absolute())
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec from {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find plugin classes in the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # logger.info(f"Inspecting class: {name} in {module_name}")
            
            # Skip base class and non-plugin classes
            if obj is PluginBase:
                continue
                
            try:
                is_sub = issubclass(obj, PluginBase)
            except Exception:
                is_sub = False
            
            if not is_sub:
                # logger.debug(f"Skipping {name}: Not a subclass of PluginBase")
                continue
            
            # Skip abstract classes
            if inspect.isabstract(obj):
                # logger.debug(f"Skipping {name}: Abstract")
                continue
            
            # Instantiate and register
            try:
                plugin_instance = obj()
                config = plugin_instance.config()
                
                plugin_name = config.get('name', name)
                
                self._plugins[plugin_name] = {
                    'instance': plugin_instance,
                    'class': obj,
                    'module': module_name,
                    'file': str(file_path),
                    'config': config
                }
                
                logger.info(f"Loaded plugin: {plugin_name} v{config.get('version', '0.0.0')}")
                
            except Exception as e:
                logger.error(f"Failed to instantiate plugin {name}: {e}")

    
    def register_plugin(self, name: str, plugin_instance: PluginBase):
        """
        Manually register a plugin instance.
        
        Args:
            name: Plugin name
            plugin_instance: Instance of PluginBase
        """
        if not isinstance(plugin_instance, PluginBase):
            raise TypeError(f"Plugin must inherit from PluginBase")
        
        config = plugin_instance.config()
        
        self._plugins[name] = {
            'instance': plugin_instance,
            'class': plugin_instance.__class__,
            'module': 'manual',
            'file': 'manual_registration',
            'config': config
        }
        
        logger.info(f"Manually registered plugin: {name}")
    
    def get_plugin(self, name: str) -> Optional[PluginBase]:
        """
        Get a plugin instance by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None if not found
        """
        if not self._loaded:
            self.discover_plugins()
        
        plugin_data = self._plugins.get(name)
        return plugin_data['instance'] if plugin_data else None
    
    def list_plugins(self) -> List[tuple]:
        """
        List all registered plugins.
        
        Returns:
            List of (name, metadata) tuples
        """
        if not self._loaded:
            self.discover_plugins()
        
        return [(name, data['config']) for name, data in self._plugins.items()]
    
    def get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin metadata dictionary or None
        """
        return self._plugins.get(name)
    
    def reload_plugin(self, name: str) -> bool:
        """
        Reload a plugin from disk.
        
        Args:
            name: Plugin name
            
        Returns:
            True if successfully reloaded
        """
        plugin_data = self._plugins.get(name)
        if not plugin_data or plugin_data['file'] == 'manual_registration':
            return False
        
        try:
            file_path = Path(plugin_data['file'])
            # Remove old plugin
            del self._plugins[name]
            # Reload from file
            self._load_plugin_from_file(file_path)
            logger.info(f"Reloaded plugin: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to reload plugin {name}: {e}")
            return False
    
    def unregister_plugin(self, name: str) -> bool:
        """
        Unregister a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            True if successfully unregistered
        """
        if name in self._plugins:
            del self._plugins[name]
            logger.info(f"Unregistered plugin: {name}")
            return True
        return False
    
    def get_plugins_by_category(self, category: str) -> List[str]:
        """
        Get all plugins in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of plugin names
        """
        return [
            name for name, data in self._plugins.items()
            if data['config'].get('category') == category
        ]
    
    def print_summary(self):
        """Print a summary of all registered plugins"""
        if not self._loaded:
            self.discover_plugins()
        
        print("\n" + "="*60)
        print(f"[*] PLUGIN REGISTRY ({len(self._plugins)} plugins loaded)")
        print("="*60)
        
        if not self._plugins:
            print("No plugins found.")
        else:
            for name, data in self._plugins.items():
                config = data['config']
                print(f"\n- {name}")
                print(f"  Version: {config.get('version', 'N/A')}")
                print(f"  Description: {config.get('description', 'N/A')}")
                print(f"  Category: {config.get('category', 'general')}")
                print(f"  File: {data['file']}")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    registry = PluginRegistry()
    registry.discover_plugins()
    registry.print_summary()
