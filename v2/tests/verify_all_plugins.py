
import sys
import os
import time
import logging

# Ensure path includes root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.plugin_registry import PluginRegistry
from core.inference_engine import InferenceEngine

# Inputs for various plugins to test
TEST_INPUTS = {
    "img_grayscale": {"image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgADNjd8qAAAAABJRU5ErkJggg=="},
    "img_invert": {"image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgADNjd8qAAAAABJRU5ErkJggg=="},
    "img_blur": {"image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgADNjd8qAAAAABJRU5ErkJggg==", "radius": 2},
    "img_resize": {"image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgADNjd8qAAAAABJRU5ErkJggg==", "width": 10, "height": 10},
    "img_rotate": {"image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgADNjd8qAAAAABJRU5ErkJggg==", "angle": 90},
    "txt_upper": {"text": "hello world"},
    "txt_wordcount": {"text": "hello world this is a test"},
    "math_prime_factors": {"number": 12345},
    "math_matrix_mul": {"matrix_a": [[1,2],[3,4]], "matrix_b": [[5,6],[7,8]]},
    "json_formatter": {"json_str": "{\"a\":1, \"b\": 2}"},
    "data_sort_csv": {"csv_data": "name,age\nbob,30\nalice,25", "column": "age"},
    "simple_ai": {"input_text": "hello"},
    "text_processor": {"text": "hello world", "operation": "upper"},
    "txt_sentiment": {"text": "I love this product"},
    "txt_anonymize": {"text": "My email is test@example.com"},
    # Add dummy inputs for others or catch errors
}

def verify_all():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("PluginVerifier")
    
    registry = PluginRegistry()
    registry.discover_plugins()
    plugins = registry.list_plugins()
    
    logger.info(f"Discovered {len(plugins)} plugins: {plugins}")
    
    engine = InferenceEngine(auto_detect_hardware=False)
    
    passed = 0
    failed = 0
    skipped = 0
    
    results = {}
    
    for plugin_name in plugins:
        logger.info(f"Testing {plugin_name}...")
        
        plugin = registry.get_plugin(plugin_name)
        if not plugin:
            logger.error(f"❌ Failed to load {plugin_name}")
            failed += 1
            results[plugin_name] = "Load Failed"
            continue
            
        # Get test input
        input_data = TEST_INPUTS.get(plugin_name)
        if not input_data:
            # Try generic input if possible or skip
            if "txt" in plugin_name:
                input_data = {"text": "test"}
            elif "img" in plugin_name:
                input_data = TEST_INPUTS["img_grayscale"]
            else:
                logger.warning(f"⚠️  Skipping {plugin_name} (No test input defined)")
                skipped += 1
                results[plugin_name] = "Skipped (No Input)"
                continue
                
        try:
            result = engine.run(plugin, input_data)
            
            if result.get('status') == 'failed' or result.get('error'):
                logger.error(f"❌ {plugin_name} Execution Failed: {result.get('error')}")
                failed += 1
                results[plugin_name] = f"Error: {result.get('error')}"
            else:
                logger.info(f"✅ {plugin_name} Passed")
                passed += 1
                results[plugin_name] = "Passed"
                
        except Exception as e:
            logger.error(f"❌ {plugin_name} Crashed: {e}")
            failed += 1
            results[plugin_name] = "Crashed"
            
    print("\n" + "="*60)
    print("PLUGIN VERIFICATION REPORT")
    print("="*60)
    print(f"Total: {len(plugins)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print("-" * 60)
    for p, status in results.items():
        print(f"{p:<25} : {status}")
    print("="*60)

if __name__ == "__main__":
    verify_all()
