#!/usr/bin/env python3
"""
Pluginfer Demo Script
Comprehensive demonstration of all features
"""
import sys
import time
sys.path.insert(0, '.')

from core import (
    PluginRegistry,
    InferenceEngine,
    HardwareDetector,
    QALController,
    LicenseValidator
)

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def demo_hardware_detection():
    """Demo 1: Hardware Detection"""
    print_section("DEMO 1: Hardware Detection")
    
    detector = HardwareDetector()
    devices = detector.detect_all_devices()
    
    print(f"Detected {len(devices)} compute device(s):\n")
    for i, device in enumerate(devices, 1):
        status = "[OK] Available" if device['available'] else "[WARNING]  Unavailable"
        print(f"{i}. {device['type'].upper()}")
        print(f"   Name: {device['name']}")
        print(f"   Status: {status}")
        print(f"   Priority: {device['priority']}")
        if 'count' in device:
            print(f"   Count: {device['count']}")
        print()
    
    best = detector.get_best_device()
    print(f"[TARGET] Best Device: {best['type'].upper()} - {best['name']}")
    
    input("\nPress Enter to continue...")

def demo_plugin_discovery():
    """Demo 2: Plugin Discovery"""
    print_section("DEMO 2: Plugin Discovery & Registry")
    
    registry = PluginRegistry("plugins")
    
    print("[SEARCH] Discovering plugins...")
    count = registry.discover_plugins()
    print(f"   Found {count} plugin(s)\n")
    
    plugins = registry.list_plugins()
    print("[PACKAGE] Available Plugins:\n")
    
    for name, config in plugins:
        print(f"• {name}")
        print(f"  Version: {config.get('version', 'N/A')}")
        print(f"  Description: {config.get('description', 'N/A')}")
        print(f"  Category: {config.get('category', 'general')}")
        print()
    
    input("Press Enter to continue...")
    
    return registry

def demo_basic_inference(registry):
    """Demo 3: Basic Inference"""
    print_section("DEMO 3: Basic Inference Execution")
    
    engine = InferenceEngine()
    plugin = registry.get_plugin('TextProcessor')
    
    if not plugin:
        print("[ERROR] TextProcessor plugin not found!")
        return
    
    print("Running text processing operations...\n")
    
    test_cases = [
        {'text': 'hello world', 'operation': 'uppercase'},
        {'text': 'PLUGINFER ROCKS', 'operation': 'lowercase'},
        {'text': 'AI Inference', 'operation': 'reverse'},
        {'text': 'GPU agnostic runtime system', 'operation': 'word_count'},
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"Test {i}: {test_input['operation'].upper()}")
        print(f"  Input:  '{test_input['text']}'")
        
        result = engine.run(plugin, test_input)
        
        print(f"  Output: '{result['result']}'")
        print(f"  Time:   {result['_metadata']['execution_time']*1000:.2f}ms")
        print()
    
    input("Press Enter to continue...")

def demo_ai_inference(registry):
    """Demo 4: AI Inference"""
    print_section("DEMO 4: AI Model Inference")
    
    engine = InferenceEngine()
    plugin = registry.get_plugin('SimpleAI')
    
    if not plugin:
        print("[ERROR] SimpleAI plugin not found!")
        return
    
    print("Running AI inference tasks...\n")
    
    tasks = [
        {'data': [1, 2, 3, 4, 5], 'task': 'classify'},
        {'data': [10, 20, 30], 'task': 'predict'},
        {'data': [5, 15, 25, 35], 'task': 'embed'},
    ]
    
    for i, task_input in enumerate(tasks, 1):
        print(f"Task {i}: {task_input['task'].upper()}")
        
        result = engine.run(plugin, task_input)
        
        print(f"  Input: {task_input['data']}")
        print(f"  Result: {result}")
        print(f"  Time: {result['_metadata']['execution_time']*1000:.2f}ms")
        print()
    
    input("Press Enter to continue...")

def demo_batch_processing(registry):
    """Demo 5: Batch Processing"""
    print_section("DEMO 5: Batch Processing")
    
    engine = InferenceEngine()
    plugin = registry.get_plugin('TextProcessor')
    
    if not plugin:
        print("[ERROR] TextProcessor plugin not found!")
        return
    
    print("Creating batch of 10 inputs...\n")
    
    batch = [
        {'text': f'Message {i}', 'operation': 'uppercase'}
        for i in range(1, 11)
    ]
    
    print("Processing batch...")
    start = time.time()
    results = engine.run_batch(plugin, batch)
    total_time = time.time() - start
    
    print(f"\n[OK] Processed {len(results)} items in {total_time*1000:.2f}ms")
    print(f"   Average: {(total_time/len(results))*1000:.2f}ms per item\n")
    
    print("First 3 results:")
    for i, result in enumerate(results[:3], 1):
        print(f"  {i}. {result['result']}")
    
    input("\nPress Enter to continue...")

def demo_qal_distribution(registry):
    """Demo 6: QAL Workload Distribution"""
    print_section("DEMO 6: QAL Workload Distribution")
    
    qal = QALController()
    plugin = registry.get_plugin('SimpleAI')
    
    if not plugin:
        print("[ERROR] SimpleAI plugin not found!")
        return
    
    print("[*] Quantum Acceleration Layer Active\n")
    print("Creating batch of 20 AI inference tasks...\n")
    
    batch = [
        {'data': [i, i+1, i+2], 'task': 'classify'}
        for i in range(20)
    ]
    
    print("Distributing workload with AUTO strategy...")
    start = time.time()
    results = qal.distribute_workload(plugin, batch, strategy='auto')
    total_time = time.time() - start
    
    print(f"\n[OK] Completed {len(results)} tasks in {total_time*1000:.2f}ms")
    print(f"   Average: {(total_time/len(results))*1000:.2f}ms per task\n")
    
    # Show performance summary
    summary = qal.get_performance_summary()
    print("[STATS] Performance Summary:")
    for device, stats in summary['devices'].items():
        print(f"  • {device}")
        print(f"    Tasks: {stats['count']}")
        print(f"    Avg Time: {stats['average_time']*1000:.2f}ms")
    
    input("\nPress Enter to continue...")

def demo_benchmarking(registry):
    """Demo 7: Performance Benchmarking"""
    print_section("DEMO 7: Performance Benchmarking")
    
    engine = InferenceEngine()
    plugin = registry.get_plugin('TextProcessor')
    
    if not plugin:
        print("[ERROR] TextProcessor plugin not found!")
        return
    
    print("Running benchmark (100 iterations)...\n")
    
    test_input = {'text': 'benchmark test', 'operation': 'uppercase'}
    
    print("⏱[*]  Benchmarking in progress...")
    benchmark = engine.benchmark_plugin(plugin, test_input, iterations=100)
    
    print(f"\n[STATS] Benchmark Results:")
    print(f"  Iterations: {benchmark['iterations']}")
    print(f"  Successful: {benchmark['successful']}")
    print(f"  Failed: {benchmark['failed']}")
    print(f"  Average Time: {benchmark['average_time']*1000:.2f}ms")
    print(f"  Min Time: {benchmark['min_time']*1000:.2f}ms")
    print(f"  Max Time: {benchmark['max_time']*1000:.2f}ms")
    print(f"  Total Time: {benchmark['total_time']*1000:.2f}ms")
    
    input("\nPress Enter to continue...")

def demo_license_system():
    """Demo 8: License System"""
    print_section("DEMO 8: License & Feature Management")
    
    validator = LicenseValidator()
    
    tier = validator.get_tier()
    usage = validator.get_usage_info()
    
    print(f"[LICENSE] License Information:\n")
    print(f"  Tier: {tier.upper()}")
    print(f"  Usage: {usage['daily_usage']} / {usage['daily_limit']}")
    print(f"  Device: {usage['device_fingerprint']}\n")
    
    print("[*] Feature Availability:")
    features = [
        ('GPU Support', 'gpu_support'),
        ('QAL Enabled', 'qal_enabled'),
        ('Multi-GPU', 'multi_gpu'),
        ('Clustering', 'clustering')
    ]
    
    for feature_name, feature_key in features:
        available = validator.check_feature(feature_key)
        status = "[OK]" if available else "[ERROR]"
        print(f"  {status} {feature_name}")
    
    max_plugins = validator.get_feature_value('max_plugins')
    batch_size = validator.get_feature_value('batch_size')
    
    print(f"\n[STATS] Limits:")
    print(f"  Max Plugins: {max_plugins if max_plugins != -1 else 'Unlimited'}")
    print(f"  Batch Size: {batch_size}")
    
    input("\nPress Enter to continue...")

def demo_statistics(registry):
    """Demo 9: Statistics & Monitoring"""
    print_section("DEMO 9: Statistics & Monitoring")
    
    engine = InferenceEngine()
    qal = QALController()
    
    print("[STATS] Execution Statistics:\n")
    
    stats = engine.get_stats()
    
    if stats['total_executions'] == 0:
        print("  No executions recorded yet")
    else:
        print(f"  Total Executions: {stats['total_executions']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Average Time: {stats['average_time']*1000:.2f}ms")
        
        if stats['by_plugin']:
            print(f"\n  By Plugin:")
            for plugin_name, plugin_stats in stats['by_plugin'].items():
                avg = plugin_stats['time'] / plugin_stats['count']
                print(f"    • {plugin_name}: {plugin_stats['count']} runs, {avg*1000:.2f}ms avg")
    
    input("\nPress Enter to continue...")

def main():
    """Main demo orchestrator"""
    print("\n" + "="*70)
    print("  [START] PLUGINFER COMPREHENSIVE DEMO")
    print("  GPU-Agnostic AI Execution Runtime")
    print("="*70)
    print("\nThis demo will showcase all major features of Pluginfer.")
    print("Press Ctrl+C at any time to exit.\n")
    
    input("Press Enter to start the demo...")
    
    try:
        # Run all demos
        demo_hardware_detection()
        registry = demo_plugin_discovery()
        demo_basic_inference(registry)
        demo_ai_inference(registry)
        demo_batch_processing(registry)
        demo_qal_distribution(registry)
        demo_benchmarking(registry)
        demo_license_system()
        demo_statistics(registry)
        
        # Final message
        print_section("[SUCCESS] DEMO COMPLETE!")
        print("Thank you for trying Pluginfer!")
        print("\n[INFO] Next Steps:")
        print("  • Read README.md for documentation")
        print("  • Check examples/ for more code samples")
        print("  • Create your own plugins in plugins/")
        print("  • Run tests with: python tests/test_all.py")
        print("\n[TIP] Get Started:")
        print("  python pluginfer.py --help")
        print("\n[*] Upgrade to Pro/Enterprise for:")
        print("  • GPU acceleration")
        print("  • Unlimited inferences")
        print("  • Advanced features")
        print("\n[NETWORK] Visit: https://pluginfer.ai")
        print("[CONTACT] Contact: support@pluginfer.ai\n")
        
    except KeyboardInterrupt:
        print("\n\n[WARNING]  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n[ERROR] Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
