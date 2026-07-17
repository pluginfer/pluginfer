"""
Security Layer Verification
"""
import sys
import os
import time
import json
import threading

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from core.security_manager import SecurityManager
from core.complete_mesh_controller import CompleteMeshController

def test_security_manager():
    print("\n[TEST] SecurityManager Core")
    sec = SecurityManager()
    
    # 1. Token Generation
    token = sec.generate_token("test_client", ["admin"])
    print(f"  ✓ Generated token: {token}")
    assert token.startswith("pk_")
    
    # 2. Token Validation
    ctx = sec.validate_token(token)
    assert ctx is not None
    assert ctx.client_id == "test_client"
    print("  ✓ Token validated successfully")
    
    # 3. Invalid Token
    assert sec.validate_token("invalid_token") is None
    print("  ✓ Invalid token rejected")
    
    # 4. Rate Limiting
    print("  Testing rate limiter (allow 60/min)...")
    client_id = "spammer"
    # Consuming 60 requests
    for _ in range(60):
        assert sec.check_rate_limit(client_id) is True
    
    # 61st request should fail
    assert sec.check_rate_limit(client_id) is False
    print("  ✓ Rate limit enforced correctly")

def test_secure_execution():
    print("\n[TEST] Secure Isolation Interface")
    sec = SecurityManager()
    
    def dangerous_code():
        return "I am safe"
    
    result = sec.run_isolated(dangerous_code)
    assert result == "I am safe"
    print("  ✓ Isolated execution wrapper working")

def start_secure_mesh():
    print("\n[TEST] Integrated Mesh Security")
    # Start controller
    ctl = CompleteMeshController('127.0.0.1', 8888, 'coordinator')
    ctl.start()
    
    try:
        time.sleep(1)
        # We just want to check if it crashes or rejects bad stuff
        # Connect manually
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('127.0.0.1', 8888))
        
        # Send garbage (should handle gracefully or error)
        msg = {'type': 'task', 'task_id': '1', 'auth_token': 'BAD_TOKEN'}
        sock.send(json.dumps(msg).encode('utf-8'))
        
        # In this version we just log rate limits, likely won't disconnect immediately 
        # unless token logic is strict.
        # But we verify the components integration via logs or lack of crash.
        print("  ✓ Connected to secure mesh without crash")
        sock.close()
        
    finally:
        ctl.stop()

if __name__ == "__main__":
    print("="*60)
    print("SECURITY LAYER VERIFICATION")
    print("="*60)
    
    try:
        test_security_manager()
        test_secure_execution()
        start_secure_mesh()
        print("\n" + "="*60)
        print("✅ ALL SECURITY TESTS PASSED")
        print("="*60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
