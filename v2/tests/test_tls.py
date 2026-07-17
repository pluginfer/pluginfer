import time
import sys
import os
import json
import ssl
import socket
import logging
import threading

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.complete_mesh_controller import CompleteMeshController

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestTLS")

def test_tls_connection():
    print("\n" + "="*70)
    print("🔒 TEST: TLS Encryption (Bank-Grade Security)")
    print("="*70)
    
    # 1. Start TLS Coordinator
    coord = CompleteMeshController('127.0.0.1', 8443, 'coordinator')
    coord.enable_tls() # Generate certs and enable flag
    coord.start()
    print("[+] Secure Coordinator Started on 8443")
    
    # Wait for startup
    time.sleep(1)
    
    # 2. Verify Certificate Exists
    if not os.path.exists("mesh_cert.pem"):
        print("❌ FAILURE: Certificate file not generated.")
        coord.stop()
        return
    print("[+] Certificate generated: mesh_cert.pem")
    
    # 3. Connect with Secure Client
    print("[*] connecting with TLS client...")
    
    # Using raw socket to verify handshake
    try:
        raw_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Create SSL context (Client side)
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE # Self-signed
        
        secure_sock = context.wrap_socket(raw_sock, server_hostname='localhost')
        secure_sock.connect(('127.0.0.1', 8443))
        
        print("   ✅ SSL Handshake Successful!")
        print(f"   🔒 Cipher: {secure_sock.cipher()}")
        print(f"   🔒 Protocol: {secure_sock.version()}")
        
        # Send Hello
        msg = {'type': 'heartbeat', 'node_id': 'tls_tester'}
        secure_sock.send(json.dumps(msg).encode('utf-8') + b'\n')
        
        # Get Response
        data = secure_sock.recv(1024).decode('utf-8')
        print(f"   ✅ Response: {data.strip()}")
        
        secure_sock.close()
        print("✅ SUCCESS: Encrypted communication verified.")
        
    except Exception as e:
        print(f"❌ FAILURE: TLS Connection Failed: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    coord.stop()

if __name__ == "__main__":
    test_tls_connection()
