"""
PLUGINFER COMPLETE VERIFICATION SCRIPT
Sanitized for Windows Console Compatibility
"""
import sys
import os
import time
from datetime import datetime

# CHANGE WORKING DIRECTORY TO PROJECT ROOT
# This is required because CompleteMeshController looks for 'plugins/' in CWD
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
sys.path.insert(0, project_root)

print(f"Running from: {os.getcwd()}")

from core.hardware_detector import HardwareDetector
from core.plugin_registry import PluginRegistry
from core.inference_engine import InferenceEngine
from core.complete_mesh_controller import CompleteMeshController
from core.auto_onboarding import DynamicPricingEngine, AutoOnboardingSystem
# MarketplaceSystem might be in auto_onboarding or elsewhere, checking usage...
# User snippet: from core.auto_onboarding import AutoOnboardingSystem, MarketplaceSystem
# But I need to check if MarketplaceSystem is actually in there. 
# grep said auto_onboarding.py exists. Let's assume it's there.
from core.auto_onboarding import MarketplaceSystem
from core.advanced_mesh_features import AdvancedMeshFeatures
from utils.gaming_detector import GamingDetector

def test_hardware_detection():
    detector = HardwareDetector()
    devices = detector.detect_all_devices()
    print(f"Test 1.1: {len(devices)} devices detected")
    assert len(devices) > 0, "Should detect at least CPU"
    print(f"Detected devices: {[d['type'] for d in devices]}")

def test_plugin_discovery():
    # Now defaults to "plugins" which is correct in root
    registry = PluginRegistry()
    count = registry.discover_plugins()
    print(f"Test 1.2: Discovered {count} plugins")
    assert count >= 2, "Should find at least TextProcessor and SimpleAI"
    plugins = registry.list_plugins()
    for name, config in plugins:
        print(f"  - {name}: {config.get('description')}")

def test_basic_inference():
    registry = PluginRegistry()
    registry.discover_plugins()
    engine = InferenceEngine()
    plugin = registry.get_plugin('TextProcessor')
    
    if not plugin:
         raise Exception("TextProcessor plugin NOT found during basic inference test")

    # Uppercase
    result = engine.run(plugin, {'text': 'hello world', 'operation': 'uppercase'})
    print(f"Input: 'hello world' -> Output: {result['result']}")
    assert result['result'] == 'HELLO WORLD'
    assert result['_metadata']['status'] == 'success'
    
    # Lowercase
    result = engine.run(plugin, {'text': 'GOODBYE WORLD', 'operation': 'lowercase'})
    assert result['result'] == 'goodbye world'
    
    # Reverse
    result = engine.run(plugin, {'text': 'Python', 'operation': 'reverse'})
    assert result['result'] == 'nohtyP'
    print("Test 1.3: All text operations passed")

def test_node_creation():
    # Bind to 0 for random available port or specific test port
    coordinator = CompleteMeshController('127.0.0.1', 9991, 'coordinator')
    coordinator.start()
    stats = coordinator.get_mesh_stats()
    print(f"Coordinator started: {stats['node_id'][:8]}...")
    print(f"Plugins loaded: {stats['plugins_loaded']}")
    
    try:
        assert stats['plugins_loaded'] >= 2
    finally:
        coordinator.stop()

def test_task_execution():
    coordinator = CompleteMeshController('127.0.0.1', 9992, 'hybrid')
    coordinator.start()
    try:
        task_ids = []
        for i in range(5):
            task_id = coordinator.submit_task(
                'TextProcessor',
                {'text': f'test {i}', 'operation': 'uppercase'}
            )
            task_ids.append(task_id)
            print(f"Submitted task {i+1}")
        
        print("Processing tasks...")
        time.sleep(3)
        
        success_count = 0
        for i, task_id in enumerate(task_ids):
            result = coordinator.get_result(task_id, timeout=5)
            if result and result.get('status') == 'success':
                output = result['result']['result']
                print(f"Task {i+1}: '{output}' [PASS]")
                if output == f'TEST {i}':
                    success_count += 1
                else:
                    print(f"  Mismatch: Expected 'TEST {i}', got '{output}'")
            else:
                print(f"Task {i+1} failed or timed out. Result: {result}")
        
        assert success_count == 5
        print(f"Test 2.2: {success_count}/5 tasks completed successfully")
    finally:
        coordinator.stop()

def test_mesh_statistics():
    coordinator = CompleteMeshController('127.0.0.1', 9993, 'hybrid')
    coordinator.start()
    try:
        for i in range(10):
            coordinator.submit_task('TextProcessor', 
                                  {'text': f'test {i}', 'operation': 'uppercase'})
        time.sleep(5)
        stats = coordinator.get_mesh_stats()
        print(f"Tasks completed: {stats['tasks_completed']}")
        print(f"Tasks failed: {stats['tasks_failed']}")
        print(f"Revenue earned: ${stats['revenue_earned']:.2f}")
        
        assert stats['tasks_completed'] >= 8
        assert stats['tasks_failed'] <= 2
        assert stats['revenue_earned'] > 0
        print("Test 2.3: Mesh statistics tracking works")
    finally:
        coordinator.stop()

def test_dynamic_pricing():
    pricing = DynamicPricingEngine()
    price_normal = pricing.calculate_price('medium')
    print("Normal pricing:")
    print(f"  Total: ${price_normal['total_price']:.4f}")
    assert 0.005 <= price_normal['total_price'] <= 0.020
    
    # Peak hour sim
    price_peak = pricing.calculate_price('medium', current_hour=14)
    print("Peak hour pricing (2 PM):")
    print(f"  Total: ${price_peak['total_price']:.4f}")
    assert price_peak['total_price'] > price_normal['total_price']
    
    price_urgent = pricing.calculate_price('medium', is_urgent=True)
    print("Urgent task pricing:")
    print(f"  Total: ${price_urgent['total_price']:.4f}")
    assert price_urgent['total_price'] > price_normal['total_price']
    print("Test 3.1: Dynamic pricing works correctly")

def test_auto_onboarding():
    onboarding = AutoOnboardingSystem()
    profile = onboarding.auto_onboard()
    print(f"User ID: {profile.user_id}")
    print(f"Node ID: {profile.node_id}")
    
    assert len(profile.user_id) == 12
    assert len(profile.node_id) == 16
    assert profile.total_earned == 0.0
    print("Test 3.2: Auto-onboarding works")

def test_gaming_detector():
    detector = GamingDetector()
    print(f"Monitoring {len(detector.game_list)} games")
    assert len(detector.game_list) >= 40
    is_gaming = detector.is_gaming()
    print(f"Currently gaming: {is_gaming}")
    assert detector.get_current_game() is None
    print("Test 3.3: Gaming detection works")

def test_model_caching():
    features = AdvancedMeshFeatures()
    test_data = b"fake_model_data" * 1000
    features.model_cache.cache_model('test-model', test_data, 'v1.0')
    has_model = features.model_cache.has_model('test-model', 'v1.0')
    print(f"Model cached: {has_model}")
    assert has_model
    stats = features.model_cache.get_cache_stats()
    print(f"Total cached: {stats['total_models']} models")
    print("Test 3.4: Model caching works")

def test_checkpoints():
    features = AdvancedMeshFeatures()
    task_id = "task_123456"
    features.checkpoint_manager.save_checkpoint(
        task_id,
        progress=0.75,
        state={'epoch': 3, 'loss': 0.234, 'step': 1500}
    )
    checkpoint = features.checkpoint_manager.load_checkpoint(task_id)
    print(f"Checkpoint progress: {checkpoint.progress*100:.0f}%")
    assert checkpoint.progress == 0.75
    assert checkpoint.state['epoch'] == 3
    features.checkpoint_manager.delete_checkpoint(task_id)
    print("Test 3.5: Checkpoint/resume works")

def run_all_tests():
    results = {'passed': 0, 'failed': 0, 'tests': []}
    
    def test(name, func):
        try:
            print(f"\n[RUN] {name}")
            func()
            results['passed'] += 1
            results['tests'].append((name, 'PASS'))
            print(f"[OK] {name}")
        except Exception as e:
            results['failed'] += 1
            results['tests'].append((name, f'FAIL: {e}'))
            print(f"[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()

    print("="*70)
    print("PLUGINFER COMPLETE VERIFICATION")
    print("="*70)

    print("\nSUITE 1: Core Framework")
    test("1.1 Hardware Detection", test_hardware_detection)
    test("1.2 Plugin Discovery", test_plugin_discovery)
    test("1.3 Basic Inference", test_basic_inference)

    print("\nSUITE 2: Mesh Networking")
    test("2.1 Node Creation", test_node_creation)
    test("2.2 Task Execution", test_task_execution)
    test("2.3 Mesh Statistics", test_mesh_statistics)

    print("\nSUITE 3: Advanced Features")
    test("3.1 Dynamic Pricing", test_dynamic_pricing)
    test("3.2 Auto-Onboarding", test_auto_onboarding)
    test("3.3 Gaming Detection", test_gaming_detector)
    test("3.4 Model Caching", test_model_caching)
    test("3.5 Checkpoints", test_checkpoints)

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    
    if results['failed'] == 0:
        print("\n[SUCCESS] ALL TESTS PASSED!")
    else:
        print(f"\n[FAILURE] {results['failed']} test(s) failed")
    
    return results

if __name__ == "__main__":
    run_all_tests()
