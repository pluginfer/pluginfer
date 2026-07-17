#!/usr/bin/env python3
"""
Example: Basic Pluginfer Usage
Demonstrates simple plugin execution
"""
import sys
sys.path.insert(0, '..')

from core import PluginRegistry, InferenceEngine

def main():
    print("="*70)
    print("EXAMPLE 1: Basic Plugin Execution")
    print("="*70 + "\n")
    
    # Initialize components
    registry = PluginRegistry("../plugins")
    engine = InferenceEngine()
    
    # Discover plugins
    print("[*] Discovering plugins...")
    count = registry.discover_plugins()
    print(f"   Found {count} plugins\n")
    
    # List available plugins
    plugins = registry.list_plugins()
    print("Available plugins:")
    for name, config in plugins:
        print(f"  - {name} - {config.get('description', 'No description')}")
    print()
    
    # Get TextProcessor plugin
    plugin = registry.get_plugin('TextProcessor')
    if not plugin:
        print("[X] TextProcessor plugin not found!")
        return
    
    # Run inference
    print("[*] Running inference...")
    test_inputs = [
        {'text': 'Hello World', 'operation': 'uppercase'},
        {'text': 'GOODBYE WORLD', 'operation': 'lowercase'},
        {'text': 'Python', 'operation': 'reverse'},
    ]
    
    for input_data in test_inputs:
        result = engine.run(plugin, input_data)
        
        print(f"\n  Input: {input_data['text']}")
        print(f"  Operation: {input_data['operation']}")
        print(f"  Result: {result['result']}")
        print(f"  Time: {result['_metadata']['execution_time']:.4f}s")
    
    # Show stats
    print("\n" + "="*70)
    engine.print_stats()


if __name__ == "__main__":
    main()
