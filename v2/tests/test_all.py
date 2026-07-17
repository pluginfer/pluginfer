#!/usr/bin/env python3
"""
Pluginfer Test Suite
Comprehensive tests for all components
"""
import sys
import time
sys.path.insert(0, '..')

from core import (
    PluginBase,
    PluginRegistry,
    InferenceEngine,
    HardwareDetector,
    QALController,
    LicenseValidator
)

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def add_test(self, name, passed, message=""):
        self.tests.append({
            'name': name,
            'passed': passed,
            'message': message
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def print_summary(self):
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        for test in self.tests:
            status = "PASS" if test['passed'] else "FAIL"
            print(f"{status} - {test['name']}")
            if test['message']:
                print(f"         {test['message']}")
        
        print(f"\nTotal: {len(self.tests)} tests")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        
        if self.failed == 0:
            print("\nAll tests passed!")
        else:
            print(f"\n{self.failed} test(s) failed")
        
        print("="*70 + "\n")
        
        return self.failed == 0

def test_hardware_detection(results):
    """Test hardware detection"""
    print("\nTesting Hardware Detection...")
    
    try:
        detector = HardwareDetector()
        devices = detector.detect_all_devices()
        
        # Should always have at least CPU
        results.add_test(
            "Hardware: Detect devices",
            len(devices) > 0,
            f"Found {len(devices)} device(s)"
        )
        
        # Should have CPU
        cpu_found = any(d['type'] == 'cpu' for d in devices)
        results.add_test(
            "Hardware: CPU detected",
            cpu_found,
            "CPU should always be available"
        )
        
        # Best device should be valid
        best = detector.get_best_device()
        results.add_test(
            "Hardware: Get best device",
            best is not None,
            f"Best device: {best['name']}"
        )
        
    except Exception as e:
        results.add_test("Hardware: Detection failed", False, str(e))

def test_plugin_system(results):
    """Test plugin registry and loading"""
    print("\nTesting Plugin System...")
    
    try:
        registry = PluginRegistry("../plugins")
        count = registry.discover_plugins()
        
        results.add_test(
            "Plugins: Discovery",
            count > 0,
            f"Discovered {count} plugin(s)"
        )
        
        # Test listing
        plugins = registry.list_plugins()
        results.add_test(
            "Plugins: List plugins",
            len(plugins) > 0,
            f"Listed {len(plugins)} plugin(s)"
        )
        
        # Test getting a plugin
        if plugins:
            first_plugin_name = plugins[0][0]
            plugin = registry.get_plugin(first_plugin_name)
            results.add_test(
                "Plugins: Get plugin",
                plugin is not None,
                f"Retrieved '{first_plugin_name}'"
            )
        
    except Exception as e:
        results.add_test("Plugins: System failed", False, str(e))

def test_inference_engine(results):
    """Test inference engine"""
    print("\nTesting Inference Engine...")
    
    try:
        registry = PluginRegistry("../plugins")
        registry.discover_plugins()
        
        engine = InferenceEngine()
        
        # Get a plugin
        plugin = registry.get_plugin('TextProcessor')
        if not plugin:
            results.add_test(
                "Engine: No plugin available",
                False,
                "TextProcessor not found"
            )
            return
        
        # Test single inference
        input_data = {'text': 'test', 'operation': 'uppercase'}
        result = engine.run(plugin, input_data)
        
        results.add_test(
            "Engine: Single inference",
            'result' in result and result['result'] == 'TEST',
            "Executed successfully"
        )
        
        # Test batch inference
        batch = [
            {'text': 'test1', 'operation': 'uppercase'},
            {'text': 'test2', 'operation': 'uppercase'}
        ]
        batch_results = engine.run_batch(plugin, batch)
        
        results.add_test(
            "Engine: Batch inference",
            len(batch_results) == 2,
            f"Processed {len(batch_results)} items"
        )
        
        # Test error handling
        bad_input = {'text': 'test'}  # Missing 'operation'
        error_result = engine.run(plugin, bad_input)
        
        results.add_test(
            "Engine: Error handling",
            'error' in error_result,
            "Handled invalid input correctly"
        )
        
    except Exception as e:
        results.add_test("Engine: Test failed", False, str(e))

def test_qal_controller(results):
    """Test QAL controller"""
    print("\nTesting QAL Controller...")
    
    try:
        registry = PluginRegistry("../plugins")
        registry.discover_plugins()
        
        qal = QALController()
        plugin = registry.get_plugin('TextProcessor')
        
        if not plugin:
            results.add_test("QAL: No plugin available", False, "TextProcessor not found")
            return
        
        # Test workload distribution
        batch = [
            {'text': f'test{i}', 'operation': 'uppercase'}
            for i in range(5)
        ]
        
        qal_results = qal.distribute_workload(plugin, batch, strategy='auto')
        
        results.add_test(
            "QAL: Workload distribution",
            len(qal_results) == 5,
            f"Distributed {len(batch)} tasks"
        )
        
        # Test performance summary
        summary = qal.get_performance_summary()
        results.add_test(
            "QAL: Performance tracking",
            summary['total_executions'] > 0,
            f"Tracked {summary['total_executions']} executions"
        )
        
    except Exception as e:
        results.add_test("QAL: Test failed", False, str(e))

def test_license_system(results):
    """Test license validation"""
    print("\nTesting License System...")
    
    try:
        validator = LicenseValidator()
        
        # Test validation
        is_valid = validator.validate()
        results.add_test(
            "License: Validation",
            True,  # Should not raise exception
            f"Current tier: {validator.get_tier()}"
        )
        
        # Test feature checking
        has_gpu = validator.check_feature('gpu_support')
        results.add_test(
            "License: Feature check",
            True,  # Should not raise exception
            f"GPU support: {has_gpu}"
        )
        
        # Test usage tracking
        usage = validator.get_usage_info()
        results.add_test(
            "License: Usage tracking",
            'daily_usage' in usage,
            f"Usage: {usage['daily_usage']}/{usage['daily_limit']}"
        )
        
    except Exception as e:
        results.add_test("License: Test failed", False, str(e))

def test_plugin_execution(results):
    """Test actual plugin execution"""
    print("\nTesting Plugin Execution...")
    
    try:
        registry = PluginRegistry("../plugins")
        registry.discover_plugins()
        
        # Test TextProcessor
        plugin = registry.get_plugin('TextProcessor')
        if plugin:
            test_cases = [
                ({'text': 'hello', 'operation': 'uppercase'}, 'HELLO'),
                ({'text': 'WORLD', 'operation': 'lowercase'}, 'world'),
                ({'text': 'test', 'operation': 'reverse'}, 'tset'),
            ]
            
            passed_all = True
            for input_data, expected in test_cases:
                result = plugin.execute(input_data)
                if result['result'] != expected:
                    passed_all = False
                    break
            
            results.add_test(
                "Plugin: TextProcessor operations",
                passed_all,
                "All operations passed"
            )
        
        # Test SimpleAI if available
        ai_plugin = registry.get_plugin('SimpleAI')
        if ai_plugin:
            ai_result = ai_plugin.execute({'data': [1,2,3], 'task': 'classify'})
            results.add_test(
                "Plugin: SimpleAI execution",
                'prediction' in ai_result,
                "AI inference successful"
            )
        
    except Exception as e:
        results.add_test("Plugin: Execution failed", False, str(e))

def test_performance(results):
    """Test performance benchmarks"""
    print("\nTesting Performance...")
    
    try:
        registry = PluginRegistry("../plugins")
        registry.discover_plugins()
        engine = InferenceEngine()
        
        plugin = registry.get_plugin('TextProcessor')
        if not plugin:
            results.add_test("Performance: No plugin", False, "TextProcessor not found")
            return
        
        # Run benchmark
        input_data = {'text': 'benchmark test', 'operation': 'uppercase'}
        
        start = time.time()
        for _ in range(100):
            engine.run(plugin, input_data)
        total_time = time.time() - start
        
        avg_time = total_time / 100
        
        results.add_test(
            "Performance: Throughput",
            avg_time < 0.01,  # Should be fast for simple operations
            f"Average: {avg_time*1000:.2f}ms per inference"
        )
        
    except Exception as e:
        results.add_test("Performance: Benchmark failed", False, str(e))

def main():
    print("="*70)
    print("PLUGINFER TEST SUITE")
    print("="*70)
    
    results = TestResults()
    
    # Run all tests
    test_hardware_detection(results)
    test_plugin_system(results)
    test_inference_engine(results)
    test_qal_controller(results)
    test_license_system(results)
    test_plugin_execution(results)
    test_performance(results)
    
    # Print summary
    success = results.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
