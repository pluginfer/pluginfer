#!/usr/bin/env python3
"""
Example: QAL Batch Processing
Demonstrates distributed workload execution
"""
import sys
sys.path.insert(0, '..')

from core import PluginRegistry, QALController

def main():
    print("="*70)
    print("EXAMPLE 2: QAL Batch Processing")
    print("="*70 + "\n")
    
    # Initialize components
    registry = PluginRegistry("../plugins")
    qal = QALController()
    
    # Discover plugins
    registry.discover_plugins()
    
    # Get plugin
    plugin = registry.get_plugin('TextProcessor')
    if not plugin:
        print("❌ TextProcessor plugin not found!")
        return
    
    # Create batch of inputs
    print("📦 Creating batch of 10 inputs...\n")
    batch_inputs = [
        {'text': f'Test message {i}', 'operation': 'uppercase'}
        for i in range(10)
    ]
    
    # Run batch with QAL
    print("⚡ Distributing workload with QAL...")
    results = qal.distribute_workload(plugin, batch_inputs, strategy='auto')
    
    # Show results
    print(f"\n✅ Processed {len(results)} items")
    print("\nFirst 3 results:")
    for i, result in enumerate(results[:3]):
        print(f"  {i+1}. {result['result']} ({result['_metadata']['execution_time']:.4f}s)")
    
    # Show performance summary
    qal.print_performance_summary()


if __name__ == "__main__":
    main()
