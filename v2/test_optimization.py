
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Ensure core is in path
sys.path.append(os.getcwd())

from core.inference_engine import InferenceEngine
from core.plugin_registry import PluginRegistry

def main():
    print("Initializing Pluginfer Optimization Engine...")
    
    # Init Engine (Auto-detects hardware)
    engine = InferenceEngine(auto_detect_hardware=True)
    
    # Load Plugins
    registry = PluginRegistry()
    registry.discover_plugins()
    
    print("DEBUG: Registry Keys:")
    for k in registry._plugins.keys():
        print(f"'{k}'")

    
    plugin = registry.get_plugin("Smart Tensor Ops")
    if not plugin:
        print("Error: Smart Tensor Ops plugin not found!")
        return
        
    print(f"\nRunning Optimization Test on: {engine.hardware.get_best_device()['name']}")
    
    # Run Benchmark
    # Small size for quick test
    input_data = {'size': 1024, 'iterations': 5}
    
    try:
        result = engine.run(plugin, input_data)
        
        print("\n" + "="*50)
        print(" OPTIMIZATION RESULT")
        print("="*50)
        if result.get('status') == 'success':
            print(f"Device Used: {result.get('device')}")
            print(f"TFLOPS:      {result.get('tflops'):.4f}")
            print(f"Total Time:  {result.get('total_time'):.4f}s")
            print(f"Meta Device: {result.get('_metadata', {}).get('device_used')}")
            print(f"Acc Mode:    {result.get('_metadata', {}).get('optimized_mode')}")
        else:
            print(f"Error: {result.get('error')}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")

if __name__ == "__main__":
    main()
