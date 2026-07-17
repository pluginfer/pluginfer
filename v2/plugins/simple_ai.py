"""
Simple AI Model Plugin
Example plugin that runs a simple neural network
"""
import sys
sys.path.insert(0, '..')

from core.plugin_base import PluginBase
from typing import Dict, Any
import time

class SimpleAIPlugin(PluginBase):
    """
    Simple AI plugin that simulates model inference.
    In a real scenario, this would load and run an actual model.
    """
    
    def __init__(self):
        super().__init__()
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Simulate loading a model"""
        time.sleep(0.1)  # Simulate load time
        self.model_loaded = True
    
    def config(self) -> Dict[str, Any]:
        return {
            'name': 'SimpleAI',
            'version': '1.0.0',
            'description': 'Simple AI inference plugin (demonstration)',
            'category': 'ai',
            'author': 'Pluginfer Team',
            'model_type': 'demo',
            'input_format': 'vector',
            'output_format': 'classification'
        }
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on input data.
        
        Expected input:
            {
                'data': list or dict,
                'task': str (classify, predict, embed)
            }
        """
        self.validate_input(input_data, ['data'])
        
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        data = input_data['data']
        task = input_data.get('task', 'classify')
        
        # Simulate inference
        time.sleep(0.05)  # Simulate computation time
        
        if task == 'classify':
            result = {
                'prediction': 'ClassA',
                'confidence': 0.95,
                'classes': ['ClassA', 'ClassB', 'ClassC']
            }
        elif task == 'predict':
            result = {
                'prediction': 42.5,
                'range': [40.0, 45.0]
            }
        elif task == 'embed':
            result = {
                'embedding': [0.1, 0.2, 0.3, 0.4, 0.5],
                'dimensions': 5
            }
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        result['task'] = task
        result['input_shape'] = str(type(data))
        
        return result


if __name__ == "__main__":
    # Test the plugin
    plugin = SimpleAIPlugin()
    
    print("Testing SimpleAIPlugin...")
    print("\nPlugin Config:", plugin.config())
    
    test_cases = [
        {'data': [1, 2, 3, 4, 5], 'task': 'classify'},
        {'data': [1, 2, 3, 4, 5], 'task': 'predict'},
        {'data': [1, 2, 3, 4, 5], 'task': 'embed'},
    ]
    
    for test_input in test_cases:
        result = plugin.execute(test_input)
        print(f"\nTask: {test_input['task']}")
        print(f"Result: {result}")
        print(f"Time: {result['_metadata']['execution_time']:.4f}s")
