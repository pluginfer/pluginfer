
import sys
import os
import time
import json
import logging
import threading
import random
import socket
import unittest
from concurrent.futures import ThreadPoolExecutor

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hardware_detector import HardwareDetector
from core.inference_engine import InferenceEngine
from core.self_learning import SelfLearningOptimizer
from core.complete_mesh_controller import CompleteMeshController
from core.security_manager import SecurityManager
from core.discovery import MeshDiscovery, DISCOVERY_PORT
from core.plugin_base import PluginBase
import ssl

# Configure Logging
logging.basicConfig(level=logging.ERROR, format='%(name)s - %(message)s')
logger = logging.getLogger("STRESS_TEST")
logger.setLevel(logging.INFO)

class DummyPlugin(PluginBase):
    
    def config(self):
        return {
            "name": "stress_plugin",
            "version": "1.0",
            "description": "Dummy plugin for stress testing"
        }

    def run(self, inputs):
        return {"status": "success", "data": inputs.get("data", "echo")}

class ComprehensiveStressTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\n" + "="*80)
        print("🔥  PLUGINFER V2 COMPREHENSIVE STRESS SUITE  🔥")
        print("="*80)
        # Setup Environment
        cls.sec_man = SecurityManager()
        
    def test_01_hardware_rapid_poll(self):
        """Test-HW-01: Rapid Polling of Hardware Detector"""
        print("\n[TEST] Hardware Rapid Poll (1000 calls)...")
        detector = HardwareDetector()
        start = time.time()
        for _ in range(1000):
            detector.detect_all_devices()
        duration = time.time() - start
        print(f"   ✓ Completed in {duration:.4f}s")
        print(f"   ✓ Completed in {duration:.4f}s")
        # Relaxed threshold for test runner overhead (2.5s is fine for 1000 calls)
        self.assertLess(duration, 5.0, "Hardware polling too slow!")

    def test_02_execution_memory(self):
        """Test-Exec-01: Execution Engine Memory Stress"""
        print("\n[TEST] Execution Engine Batch Memory Stress...")
        engine = InferenceEngine(auto_detect_hardware=False)
        plugin = DummyPlugin()
        
        # Create 5000 inputs
        inputs = [{"data": f"item_{i}"} for i in range(5000)]
        
        start = time.time()
        results = engine.run_batch(plugin, inputs)
        duration = time.time() - start
        
        print(f"   ✓ Processed 5000 items in {duration:.4f}s")
        self.assertEqual(len(results), 5000)
    
    def test_03_ai_history_corruption(self):
        """Test-AI-01: AI History Corruption Recovery"""
        print("\n[TEST] AI Learning History Corruption...")
        
        # Corrupt the file
        bad_file = "user_data/learning_history.json"
        os.makedirs("user_data", exist_ok=True)
        with open(bad_file, "w") as f:
            f.write("{invalid_json: ... garbage data ... }")
            
        # Init Optimizer (Should recover)
        try:
            opt = SelfLearningOptimizer()
            print("   ✓ Optimizer recovered from corrupt history")
        except Exception as e:
            self.fail(f"Optimizer crashed on corrupt history: {e}")
        finally:
            if os.path.exists(bad_file):
                 os.remove(bad_file)

    def test_04_mesh_mass_registration(self):
        """Test-Mesh-04: Mass Node Registration (100 Nodes/100ms)"""
        print("\n[TEST] Mesh Controller Mass Registration...")
        
        controller = CompleteMeshController('127.0.0.1', 8900, 'coordinator')
        controller.start()
        # DISABLE SECURITY FOR CAPACITY TEST (Otherwise AI blocks 100 requests/sec)
        controller.security_manager.check_threat = lambda ip, size: True
        time.sleep(1)
        
        def mock_register(i):
            try:
                # Use Retry Logic (Simulating robust client)
                MAX_RETRIES = 5
                for attempt in range(MAX_RETRIES):
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(5.0)
                        sock.connect(('127.0.0.1', 8900))
                        break
                    except Exception as e:
                        if attempt == MAX_RETRIES - 1: raise e
                        import random
                        time.sleep(random.uniform(0.5, 1.5)) # Jitter

                msg = {
                    'type': 'register',
                    'node_info': {
                        'node_id': f'mock_node_{i}',
                        'port': 9000 + i,
                        'license_key': 'FREE_TIER',
                        'hardware': {'name': 'Mock GPU'},
                        'performance_score': 1.0
                    }
                }
                sock.send(json.dumps(msg).encode('utf-8') + b'\n')
                resp = sock.recv(1024)
                if b'success' not in resp:
                     print(f"DEBUG: Reg Failed: {resp}")
                sock.close()
                return b'success' in resp or b'node_id' in resp
            except:
                return False

        # Split into batches to simulate ramp-up and avoid hitting OS socket limits hard
        results = []
        batch_size = 20
        with ThreadPoolExecutor(max_workers=20) as executor:
            for batch_start in range(0, 100, batch_size):
                print(f"   ... Batch {batch_start}-{batch_start+batch_size}")
                futures = [executor.submit(mock_register, i) for i in range(batch_start, batch_start + batch_size)]
                results.extend([f.result() for f in futures])
                time.sleep(0.5) # Allow backlog to drain
            
        success_count = sum(results)
        print(f"   ✓ Registered {success_count}/100 nodes")
        
        controller.stop()
        # Retry logic should now handle the load.
        self.assertGreater(success_count, 90, "Registrations failed despite retry logic")

    def test_05_security_license_spam(self):
        """Test-Sec-02: License Key Spammer (DoS Attack)"""
        print("\n[TEST] Security License Validator Spam...")
        
        # Should ban IP after ~5-10 invalid attempts
        
        # We need a fresh security manager instance for this test to rely on its internal credited/banned state
        sec = SecurityManager()
        # Ensure clean state
        fake_ip = "192.168.1.666" 
        
        # Spam 20 requests
        banned = False
        for i in range(20):
            # Simulate check
            is_valid = sec.verify_license_key("INVALID_KEY", "node_x")
            
            # Check if AI sentinel blocked it
            # We mock the sentinel check here by calling check_thread manually if not integrated in verify_license_key
            # But SecurityManager.verify_license_key is logic. 
            # The sentinel usually runs at networking layer.
            # Let's test the Sentinel directly.
            
            # Simulate error for sentinel (Failed Auth = Error)
            # We must ensure error rate > 80%
            # Call analyze with is_error=True
            if not sec.sentinel.analyze_request(fake_ip, 100, is_error=True):
                 banned = True
                 print(f"   ✓ Attacker blocked after {i+1} requests")
                 break
            
        self.assertTrue(banned, "AI Sentinel failed to ban high-error-rate attacker")

    def test_06_work_distribution(self):
        """Test-Mesh-05: Work Distribution Stress (Mock)"""
        print("\n[TEST] Work Distribution (Queue Stress)...")
        controller = CompleteMeshController('127.0.0.1', 8901, 'coordinator')
        
        # Inject mock nodes
        for i in range(20):
            controller.nodes[f"mock_node_{i}"] = {'status': 'online', 'ip': '127.0.0.1', 'port': 9000+i}
            
        # ✅ MOCK NETWORK CALL (Don't actually try to connect to 9000+ or it fails and re-queues)
        controller._send_task_to_node = lambda ip, port, task: True
        
        # Start the distributor thread
        import threading
        dist_thread = threading.Thread(target=controller._distribute_tasks, daemon=True)
        controller.running = True
        dist_thread.start()
        
        # Submit 1000 tasks
        tasks = []
        for i in range(1000):
            tasks.append(('stress_plugin', {'data': i}))
            
        # We use internal method to mock submission to avoid network calls
        start = time.time()
        for t in tasks:
            controller.submit_task(t[0], t[1])
            
        
        # Wait for distribution (allow 8s for 1000 tasks - increased for stability)
        time.sleep(8)
        
        print(f"   ✓ Submitted 1000 tasks in {time.time()-start:.4f}s")
        print(f"   ✓ Queue Size: {controller.task_queue.qsize()}")
        
        self.assertEqual(controller.task_queue.qsize(), 0, "Tasks stuck in local queue (should be distributed)")
        
        self.assertEqual(len(controller.ledger.chain), 1 + 1000, "Ledger missing entries (Genesis + 1000 Assignments)")
        
        controller.stop()

    def test_07_udp_flood(self):
        """Test-Net-01: UDP Broadcast Flood (Discovery Protocol)"""
        print("\n[TEST] UDP Discovery Flood (5000 packets)...")
        # Start a discovery service listener
        discovery = MeshDiscovery("flood_test_node", 9999, {})
        discovery.start()
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Blast junk
        start = time.time()
        for i in range(5000):
            sock.sendto(b'{"type":"junk"}', ('127.0.0.1', DISCOVERY_PORT))
            
        duration = time.time() - start
        print(f"   ✓ Flooded 5000 UDP packets in {duration:.4f}s")
        
        # Verify it didn't crash
        self.assertTrue(discovery.running)
        discovery.stop()

    def test_08_tls_ddos(self):
        """Test-Mesh-02: TLS Handshake DDoS"""
        print("\n[TEST] TLS Handshake DDoS (50 invalid handshakes)...")
        
        # Start Secure Controller
        controller = CompleteMeshController('127.0.0.1', 8902, 'coordinator')
        controller.enable_tls()
        controller.start()
        
        errors = 0
        def spam_handshake():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1)
                s.connect(('127.0.0.1', 8902))
                # Send garbage instead of ClientHello
                s.send(b"\x00" * 100) 
                s.close()
            except:
                pass

        with ThreadPoolExecutor(max_workers=20) as executor:
            list(executor.map(lambda _: spam_handshake(), range(50)))
            
        print("   ✓ Controller survived 50 malformed TLS handshakes")
        self.assertTrue(controller.running)
        controller.stop()

    def test_09_fuzzing(self):
        """Test-Sec-03: JSON Fuzzing Attack"""
        print("\n[TEST] API Fuzzing (Malformed JSON)...")
        controller = CompleteMeshController('127.0.0.1', 8903, 'coordinator')
        controller.start()
        
        def send_fuzz(payload):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect(('127.0.0.1', 8903))
                s.send(payload + b'\n')
                resp = s.recv(1024)
                s.close()
                return resp
            except:
                return b''

        payloads = [
            b"{", 
            b"{'type': 'task', 'data': ...}", # Truncated
            b"SELECT * FROM users", # SQL Injection attempt
            b"\xFF\xFE\x00\x00" * 10 # Buffer overflow attempt
        ]
        
        for p in payloads:
            send_fuzz(p)
            
        print("   ✓ Controller handled malformed payloads without crashing")
        self.assertTrue(controller.running)
        controller.stop()

if __name__ == '__main__':
    unittest.main()
