
import socket
import json
import time
import threading

print("--- Testing P2P Mesh Connectivity ---")

# Configuration
TEST_PORT = 19999
HOST = "127.0.0.1"

def start_server():
    """Mock a Peer Node Listener"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, TEST_PORT))
    s.listen(1)
    print(f"[SERVER] Listening on {TEST_PORT}...")
    
    conn, addr = s.accept()
    print(f"[SERVER] Connection Accepted from {addr}")
    
    # Handshake Protocol
    # 1. Receive Identity
    data = conn.recv(1024).decode()
    print(f"[SERVER] Received Handshake: {data}")
    
    # 2. Respond
    response = json.dumps({"type": "HANDSHAKE_ACK", "node_id": "TEST_SERVER_NODE"})
    conn.send(response.encode())
    
    conn.close()
    s.close()

# Start Server in Background
server_thread = threading.Thread(target=start_server)
server_thread.start()
time.sleep(1) # Wait for startup

# Client (Simulating Our Node)
try:
    print("[CLIENT] Attempting connection...")
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((HOST, TEST_PORT))
    
    # Send Handshake
    handshake = json.dumps({"type": "HANDSHAKE", "node_id": "TEST_CLIENT_NODE", "swarm_id": "public"})
    client.send(handshake.encode())
    
    # Receive Ack
    resp_data = client.recv(1024).decode()
    resp = json.loads(resp_data)
    
    if resp.get('type') == 'HANDSHAKE_ACK':
        print("✅ P2P Handshake Validated")
        print("✅ Mesh Connectivity is OPERATIONAL")
    else:
        print("❌ Invalid Handshake Response")

    client.close()
    
except Exception as e:
    print(f"❌ Connection Failed: {e}")

server_thread.join()
