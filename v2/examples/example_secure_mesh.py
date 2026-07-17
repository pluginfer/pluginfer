#!/usr/bin/env python3
"""
Complete Example: Secure Mesh Networking with Dashboard
Demonstrates all Phase 2 features working together
"""
import sys
import time
sys.path.insert(0, '..')

from core.secure_mesh_controller import SecureMeshController
from utils.network_config import NetworkConfigurator
from utils.web_dashboard import start_dashboard
import threading

def example_secure_local_mesh():
    """Example: Secure mesh on local network"""
    print("="*70)
    print("EXAMPLE: Secure Local Mesh Network")
    print("="*70)
    print()
    
    print("This example demonstrates:")
    print("  ✅ TLS/SSL encryption")
    print("  ✅ Token authentication")
    print("  ✅ Rate limiting")
    print("  ✅ Web dashboard")
    print()
    
    # Step 1: Network configuration
    print("Step 1: Detecting network configuration...")
    configurator = NetworkConfigurator()
    local_ip = configurator.get_local_ip()
    print(f"   Local IP: {local_ip}")
    print()
    
    # Step 2: Start secure coordinator
    print("Step 2: Starting secure coordinator...")
    coordinator = SecureMeshController(
        host='0.0.0.0',
        port=9999,
        mode='coordinator',
        use_tls=True,
        require_auth=True
    )
    coordinator.start()
    print("   ✅ Coordinator started with TLS + Authentication")
    print()
    
    # Step 3: Generate tokens for workers
    print("Step 3: Generating authentication tokens...")
    tokens = []
    for i in range(3):
        token = coordinator.auth.generate_token(
            user_id=f'worker{i+1}',
            expires_hours=24,
            permissions=['worker']
        )
        tokens.append(token)
        print(f"   Worker {i+1} token: {token[:20]}...")
    print()
    
    # Step 4: Simulate workers connecting
    print("Step 4: Simulating worker connections...")
    print("   (In real scenario, run worker script on other machines)")
    print()
    
    # Simulate registration
    for i, token in enumerate(tokens):
        # In real scenario, this would be done from worker machines
        coordinator.nodes[f'192.168.1.10{i}:1000{i}'] = {
            'user_id': f'worker{i+1}',
            'capabilities': {
                'has_gpu': i < 2,  # First 2 have GPUs
                'gpu_type': 'cuda' if i == 0 else 'mps' if i == 1 else 'cpu',
                'gpu_count': 2 if i == 0 else 1,
                'cpu_cores': 16 if i == 0 else 8,
                'memory_gb': 32 if i == 0 else 16
            },
            'registered_at': time.time(),
            'status': 'online',
            'compute_power': 250 if i == 0 else 100 if i == 1 else 50,
            'tasks_completed': i * 10,
            'revenue': i * 1.0
        }
    
    print("   ✅ 3 workers registered")
    print()
    
    # Step 5: Show status
    print("Step 5: Current mesh status...")
    coordinator.print_status()
    
    # Step 6: Start web dashboard
    print("Step 6: Starting web dashboard...")
    print(f"   Dashboard will be available at: http://localhost:5000")
    print(f"   (Open this URL in your browser)")
    print()
    
    # Start dashboard in separate thread
    dashboard_thread = threading.Thread(
        target=start_dashboard,
        args=(coordinator, '127.0.0.1', 5000),
        daemon=True
    )
    dashboard_thread.start()
    
    time.sleep(2)
    
    print("\n" + "="*70)
    print("✅ SECURE MESH NETWORK IS RUNNING!")
    print("="*70)
    print()
    print("What you can do now:")
    print("  1. Open http://localhost:5000 in your browser")
    print("  2. Monitor nodes and statistics")
    print("  3. Connect real workers from other machines")
    print()
    print("To connect a worker:")
    print(f"  python examples/example_secure_worker.py \\")
    print(f"    --host {local_ip} \\")
    print(f"    --port 9999 \\")
    print(f"    --token {tokens[0][:20]}...")
    print()
    print("Press Ctrl+C to stop...")
    print("="*70)
    
    try:
        while True:
            time.sleep(5)
            # Update some stats for demo
            for node_id in coordinator.nodes:
                coordinator.nodes[node_id]['tasks_completed'] += 1
                coordinator.nodes[node_id]['revenue'] += 0.01
    except KeyboardInterrupt:
        print("\n\nStopping secure mesh network...")
        coordinator.stop()
        print("✅ Stopped")

def example_internet_ready_mesh():
    """Example: Internet-ready mesh with all security features"""
    print("="*70)
    print("EXAMPLE: Internet-Ready Secure Mesh")
    print("="*70)
    print()
    
    print("This example shows production-ready configuration:")
    print("  ✅ TLS encryption")
    print("  ✅ Token authentication")
    print("  ✅ IP whitelisting")
    print("  ✅ Rate limiting")
    print("  ✅ Web dashboard")
    print()
    
    # Network configuration
    configurator = NetworkConfigurator()
    configurator.print_network_info()
    
    print("Configuration Guide:")
    print("-" * 70)
    
    # Coordinator setup
    print("\n📝 COORDINATOR SETUP (Server/Main Machine):")
    print("""
1. Start secure coordinator:
   
   from core.secure_mesh_controller import SecureMeshController
   
   coordinator = SecureMeshController(
       host='0.0.0.0',
       port=9999,
       mode='coordinator',
       use_tls=True,
       require_auth=True
   )
   
   # Optional: Add IP whitelist for extra security
   coordinator.add_to_whitelist('203.0.113.50')  # Worker 1
   coordinator.add_to_whitelist('203.0.113.51')  # Worker 2
   
   coordinator.start()

2. Generate tokens for workers:
   
   token1 = coordinator.auth.generate_token('worker1', expires_hours=168)
   token2 = coordinator.auth.generate_token('worker2', expires_hours=168)
   
   # Share tokens securely with workers

3. Start dashboard:
   
   from utils.web_dashboard import start_dashboard
   start_dashboard(coordinator, host='0.0.0.0', port=5000)
    """)
    
    # Worker setup
    print("\n👷 WORKER SETUP (Remote Machines):")
    print("""
1. Connect to coordinator securely:
   
   from core.secure_mesh_controller import SecureMeshController
   from core.mesh_controller import get_system_capabilities
   
   worker = SecureMeshController(
       host='0.0.0.0',
       port=10000,
       mode='worker',
       use_tls=True,
       require_auth=True
   )
   worker.start()
   
   # Connect with token
   success = worker.connect_secure(
       coordinator_host='coordinator.example.com',  # Or IP
       coordinator_port=9999,
       auth_token='YOUR_TOKEN_HERE',
       capabilities=get_system_capabilities()
   )

2. Worker starts processing tasks automatically
    """)
    
    # VPN recommendation
    print("\n🔒 RECOMMENDED: Use VPN for extra security")
    print("""
1. Install Tailscale on all machines:
   https://tailscale.com/download

2. Connect all machines to same Tailscale network

3. Use Tailscale IPs instead of public IPs:
   coordinator_host='100.64.0.1'  # Tailscale IP
   
Benefits:
  ✅ Encrypted tunnel
  ✅ No port forwarding needed
  ✅ Works behind firewalls
  ✅ Easy to manage
    """)
    
    print("\n" + "="*70)
    input("\nPress Enter to continue...")

def main():
    """Main menu"""
    print("\n" + "="*70)
    print("  🔐 SECURE MESH NETWORKING EXAMPLES")
    print("="*70)
    print("\nChoose an example:")
    print("  1. Secure Local Mesh (Live Demo)")
    print("  2. Internet-Ready Configuration Guide")
    print("  3. Exit")
    print()
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        example_secure_local_mesh()
    elif choice == '2':
        example_internet_ready_mesh()
    elif choice == '3':
        print("Goodbye!")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
