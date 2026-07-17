import sys
import os
sys.path.insert(0, '..')
import traceback

print("Diagnostic Start")

try:
    from core.plugin_registry import PluginRegistry
    from core.inference_engine import InferenceEngine

    registry = PluginRegistry("../plugins")
    print("Discovering plugins...")
    count = registry.discover_plugins()
    print(f"Discovered {count} plugins")
    
    tp = registry.get_plugin('TextProcessor')
    if not tp:
        print("TextProcessor NOT found")
        sys.exit(1)
        
    engine = InferenceEngine()
    print("Engine initialized")

    test_input = {'text': 'Hello World', 'operation': 'uppercase'}
    print(f"Running input: {test_input}")
    
    result = engine.run(tp, test_input)
    print("Execution result keys:", result.keys())
    
    if 'error' in result:
        print(f"Plugin execution returned error: {result['error']}")
    else:
        print(f"Result: {result['result']}")

except Exception as e:
    print(f"Diagnostic Crashed: {e}")
    traceback.print_exc()

print("Diagnostic End")
