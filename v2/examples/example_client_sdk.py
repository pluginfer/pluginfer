"""
Pluginfer Client SDK Example
This shows how a Company/Researcher connects to the mesh to submit tasks.
"""
import socket
import json
import time

class PluginferClient:
    def __init__(self, coordinator_ip='127.0.0.1', coordinator_port=8888):
        self.coordinator = (coordinator_ip, coordinator_port)
        self.api_key = "sk_demo_key" # In prod, you'd buy this
        
    def submit_task(self, model_name, input_data, budget=1.0):
        print(f"🚀 [CLIENT] Submitting '{model_name}' task to Mesh...")
        
        # 1. Create Task Payload
        task = {
            'type': 'task',
            'auth_token': self.api_key,
            'task': {
                'id': f"job_{int(time.time())}",
                'type': 'inference',
                'model': model_name,
                'input_data': input_data,
                'max_cost': budget
            }
        }
        
        # 2. Send to Coordinator (The "Entry Node")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(self.coordinator)
            sock.send(json.dumps(task).encode('utf-8'))
            print("   ✓ Task sent successfully!")
            print("   ✓ Waiting for results from a distributed gamer node...")
            
            # 3. Wait for response
            # (In real async API, you'd get a webhook callback)
            sock.close()
            return True
            
        except ConnectionRefusedError:
            print("   ❌ Error: Mesh Coordinator is offline.")
            return False

if __name__ == "__main__":
    # Simulate a company using the mesh
    client = PluginferClient()
    
    # Submit an Image Recognition Job
    client.submit_task(
        model_name="PyTorchCNN",
        input_data={"image_path": "dataset/sample_01.jpg"},
        budget=0.05
    )
    
    print("\nNOTE: In production, companies connect via HTTP API (REST)")
    print("      We provide a Python SDK 'pip install pluginfer'")
