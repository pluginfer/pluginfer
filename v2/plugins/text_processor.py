"""
Text Processing Plugin
Example plugin that performs simple text operations
"""
import sys
sys.path.insert(0, '..')

from core.plugin_base import PluginBase
from typing import Dict, Any

class TextProcessorPlugin(PluginBase):
    """
    Simple text processing plugin for demonstration.
    Supports operations: uppercase, lowercase, reverse, word_count
    """
    
    def config(self) -> Dict[str, Any]:
        return {
            'name': 'TextProcessor',
            'version': '1.0.0',
            'description': 'Process text with various operations',
            'category': 'text',
            'author': 'Pluginfer Team',
            'supported_operations': ['uppercase', 'lowercase', 'reverse', 'word_count']
        }
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process text based on requested operation.
        
        Expected input:
            {
                'text': str,
                'operation': str (uppercase, lowercase, reverse, word_count)
            }
        """
        # Validate input
        self.validate_input(input_data, ['text', 'operation'])
        
        text = input_data['text']
        operation = input_data['operation']
        
        # Perform operation
        if operation == 'uppercase':
            result = text.upper()
        elif operation == 'lowercase':
            result = text.lower()
        elif operation == 'reverse':
            result = text[::-1]
        elif operation == 'word_count':
            result = len(text.split())
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        
        return {
            'original': text,
            'operation': operation,
            'result': result
        }


if __name__ == "__main__":
    # Test the plugin
    plugin = TextProcessorPlugin()
    
    print("Testing TextProcessorPlugin...")
    print("\nPlugin Config:", plugin.config())
    
    test_cases = [
        {'text': 'Hello World', 'operation': 'uppercase'},
        {'text': 'Hello World', 'operation': 'lowercase'},
        {'text': 'Hello World', 'operation': 'reverse'},
        {'text': 'Hello World', 'operation': 'word_count'},
    ]
    
    for test_input in test_cases:
        result = plugin.execute(test_input)
        print(f"\nInput: {test_input}")
        print(f"Result: {result['result']}")
        print(f"Time: {result['_metadata']['execution_time']:.4f}s")
