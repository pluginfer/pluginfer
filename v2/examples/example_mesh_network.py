#!/usr/bin/env python3
"""
Example: Mesh Network for Distributed Computing
Demonstrates how to create a compute mesh with multiple nodes
"""
import sys
import time
sys.path.insert(0, '..')

from core.mesh_controller import (
    MeshNetworkController,
    get_system_capabilities,
    connect_to_mesh,
    find_coordinator
)

def example_coordinator():
    """Example: Run as coordinator node"""
    print("="*70)
    print("EXAMPLE: Mesh Network Coordinator")
    print("="*70)
    print("\nThis node will coordinate tasks across worker nodes.")
    print()
    
    # Start coordinator
    print("[*] Starting coordinator...")
    
    # Ask for internet access
    enable_internet = input("Enable Internet Access (for remote workers)? (y/N): ").lower().startswith('y')
    if enable_internet:
        print("[*] Attempting to configure UPnP for Internet Access...")
    
    coordinator = MeshNetworkController(
        host='0.0.0.0',  # Listen on all interfaces
        port=9999,
        mode='coordinator',
        enable_internet_access=enable_internet
    )
    coordinator.start()
    
    print(f"\n[+] Coordinator started!")
    print(f"   Node ID: {coordinator.node_id}")
    print(f"   Listening on: {coordinator.host}:{coordinator.port}")
    print("\n[*] Worker nodes can connect to this coordinator.")
    print("\n[*] Waiting for nodes to join...")
    
    # Simulate some worker registrations
    print("\n[*] Simulating worker registrations...")
    
    # Register 3 simulated worker nodes
    coordinator.register_node('192.168.1.100', 10000, {
        'has_gpu': True,
        'gpu_type': 'cuda',
        'gpu_count': 2,
        'cpu_cores': 16,
        'memory_gb': 32
    })
    
    coordinator.register_node('192.168.1.101', 10001, {
        'has_gpu': True,
        'gpu_type': 'mps',
        'gpu_count': 1,
        'cpu_cores': 8,
        'memory_gb': 16
    })
    
    coordinator.register_node('192.168.1.102', 10002, {
        'has_gpu': False,
        'cpu_cores': 8,
        'memory_gb': 16
    })
    
    time.sleep(1)
    
    # Show mesh status
    coordinator.print_mesh_status()
    
    # Distribute tasks
    print("\n[*] Distributing batch of 20 AI inference tasks...")
    batch = [
        {'data': [i, i+1, i+2], 'task': 'classify'}
        for i in range(20)
    ]
    
    task_ids = coordinator.distribute_batch(
        plugin_name='SimpleAI',
        batch=batch,
        strategy='load_balanced'
    )
    
    print(f"\n[+] Distributed {len(task_ids)} tasks across the mesh")
    print(f"   Tasks are being processed on worker nodes...")
    
    # Wait for processing
    time.sleep(3)
    
    # Show final status
    print("\n[*] Final Status:")
    coordinator.print_mesh_status()
    
    # Cleanup
    input("\nPress Enter to stop coordinator...")
    coordinator.stop()
    print("\n[+] Coordinator stopped")

def example_worker():
    """Example: Run as worker node"""
    print("="*70)
    print("EXAMPLE: Mesh Network Worker")
    print("="*70)
    print("\nThis node will join a mesh and process tasks.")
    print()
    
    # Get system capabilities
    print("[*] Detecting system capabilities...")
    capabilities = get_system_capabilities()
    
    print(f"\n[*] System Capabilities:")
    for key, value in capabilities.items():
        print(f"   {key}: {value}")
    
    # Start worker
    print("\n[*] Starting worker node...")
    worker = MeshNetworkController(
        host='0.0.0.0',
        port=10003,  # Different port than coordinator
        mode='worker'
    )
    worker.start()
    
    print(f"\n[+] Worker started!")
    print(f"   Node ID: {worker.node_id}")
    print(f"   Ready to process tasks")
    
    # Connect to coordinator (in real scenario)
    # Connect to coordinator (in real scenario)
    print("\n[*] Connecting to coordinator...")
    
    use_auto = input("Use Auto-Discovery? (Y/n): ").lower() != 'n'
    
    host = '127.0.0.1'
    port = 9999
    
    if use_auto:
        print("   Searching for coordinator...")
        info = find_coordinator()
        if info:
            host = info['ip']
            port = info['port']
            print(f"   Found coordinator at {host}:{port}")
        else:
            print("   Auto-discovery failed. Please enter details manually.")
            use_auto = False
            
    if not use_auto:
        host_input = input("Coordinator Host (IP or domain) [127.0.0.1]: ").strip()
        if host_input:
            host = host_input
            # Handle host:port format
            if ':' in host:
                host, p = host.split(':')
                port = int(p)
                
    print(f"   Connecting to {host}:{port}...")
    if connect_to_mesh(host, port, capabilities):
        print("[+] Connected successfully!")
    else:
        print("[-] Connection failed")
        return
    
    # Simulate processing
    print("\n[*] Worker is ready to process tasks...")
    print("   Tasks will be executed on this machine's hardware")
    
    # Keep running
    try:
        print("\n[*] Press Ctrl+C to stop worker")
        while True:
            worker.print_mesh_status()
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n\n[*] Stopping worker...")
        worker.stop()
        print("[+] Worker stopped")

def example_auto_discovery():
    """Example: Auto-discover coordinator"""
    print("="*70)
    print("EXAMPLE: Auto-Discovery Worker")
    print("="*70)
    print("\nThis node will search for a coordinator on the local network.")
    print()
    
    print("[*] Detecting system capabilities...")
    capabilities = get_system_capabilities()
    
    print("\n[*] Searching for coordinator (UDP Broadcast)...")
    coordinator_info = find_coordinator(timeout=5)
    
    if coordinator_info:
        print(f"\n[+] Found Coordinator!")
        print(f"   IP: {coordinator_info['ip']}")
        print(f"   Port: {coordinator_info['port']}")
        print(f"   Node ID: {coordinator_info['node_id']}")
        print(f"   Hostname: {coordinator_info['hostname']}")
        
        print(f"\n[*] Connecting...")
        if connect_to_mesh(coordinator_info['ip'], coordinator_info['port'], capabilities):
            print("\n[+] Successfully connected to mesh!")
            print("   You are now part of the compute cluster.")
        else:
            print("\n[-] Failed to connect to coordinator.")
    else:
        print("\n[-] No coordinator found on local network.")
        print("   Make sure the coordinator is running and on the same WiFi/LAN.")


def example_hybrid():
    """Example: Run as hybrid node (coordinator + worker)"""
    print("="*70)
    print("EXAMPLE: Hybrid Mesh Node")
    print("="*70)
    print("\nThis node can both coordinate AND execute tasks.")
    print()
    
    # Start hybrid node
    print("[*] Starting hybrid node...")
    node = MeshNetworkController(
        host='0.0.0.0',
        port=9999,
        mode='hybrid'
    )
    node.start()
    
    print(f"\n[+] Hybrid node started!")
    print(f"   Can coordinate tasks AND execute them")
    
    # Register some other nodes
    print("\n[*] Registering peer nodes...")
    node.register_node('192.168.1.100', 10000, {
        'has_gpu': True,
        'gpu_type': 'cuda',
        'gpu_count': 1,
        'cpu_cores': 8,
        'memory_gb': 16
    })
    
    # Submit tasks that will be distributed
    print("\n[*] Submitting tasks to mesh...")
    task_ids = []
    for i in range(10):
        task_id = node.submit_task('SimpleAI', {'data': [i]})
        task_ids.append(task_id)
    
    print(f"   Submitted {len(task_ids)} tasks")
    print("   Tasks will be processed across the mesh")
    
    # Wait for processing
    time.sleep(3)
    
    # Show status
    node.print_mesh_status()
    
    # Cleanup
    input("\nPress Enter to stop...")
    node.stop()
    print("[+] Hybrid node stopped")

def example_real_world_scenario():
    """Example: Real-world distributed AI inference"""
    print("="*70)
    print("REAL-WORLD SCENARIO: Distributed AI Inference Farm")
    print("="*70)
    print()
    print("Scenario: You have 3 machines:")
    print("  1. Server with 2x NVIDIA GPUs (coordinator)")
    print("  2. MacBook with M2 chip (worker)")
    print("  3. Linux box with CPU only (worker)")
    print()
    print("You want to distribute 1000 AI inference tasks across them.")
    print()
    
    # Setup coordinator (simulated)
    print("[*] Machine 1: Starting coordinator...")
    coordinator = MeshNetworkController(host='0.0.0.0', port=9999, mode='hybrid')
    coordinator.start()
    
    # Register workers
    print("\n[*] Machine 2 & 3: Connecting to coordinator...")
    
    # Worker 1: MacBook M2
    coordinator.register_node('192.168.1.50', 10000, {
        'has_gpu': True,
        'gpu_type': 'mps',
        'gpu_count': 1,
        'cpu_cores': 8,
        'memory_gb': 16,
        'gpu_name': 'Apple M2'
    })
    
    # Worker 2: Linux CPU
    coordinator.register_node('192.168.1.51', 10001, {
        'has_gpu': False,
        'cpu_cores': 12,
        'memory_gb': 32
    })
    
    time.sleep(1)
    
    # Show mesh
    coordinator.print_mesh_status()
    
    # Distribute large batch
    print("\n[*] Creating batch of 1000 AI inference tasks...")
    batch = [
        {'image_id': i, 'data': [i % 255] * 784}  # Simulated image data
        for i in range(1000)
    ]
    
    print(f"\n[*] Distributing across mesh with load-balanced strategy...")
    print("   Tasks will be distributed based on compute power:")
    print("   - NVIDIA GPUs get ~60% of tasks")
    print("   - Apple M2 gets ~30% of tasks")
    print("   - CPU gets ~10% of tasks")
    print()
    
    start_time = time.time()
    task_ids = coordinator.distribute_batch(
        plugin_name='ImageClassifier',
        batch=batch,
        strategy='load_balanced'
    )
    distribution_time = time.time() - start_time
    
    print(f"[+] Distributed {len(task_ids)} tasks in {distribution_time:.2f}s")
    print(f"\n[*] Processing...")
    
    # Simulate processing time
    time.sleep(5)
    
    # Show results
    print("\n[+] Processing Complete!")
    coordinator.print_mesh_status()
    
    stats = coordinator.get_mesh_stats()
    
    print(f"\n[*] Revenue Distribution:")
    print(f"   @ $0.01 per task")
    print(f"   Machine 1 (NVIDIA): ${stats['revenue_earned'] * 0.6:.2f}")
    print(f"   Machine 2 (M2): ${stats['revenue_earned'] * 0.3:.2f}")
    print(f"   Machine 3 (CPU): ${stats['revenue_earned'] * 0.1:.2f}")
    
    # Cleanup
    input("\nPress Enter to shutdown mesh...")
    coordinator.stop()
    print("[+] Mesh shutdown complete")

def main():
    """Main menu"""
    print("\n" + "="*70)
    print("  [*] MESH NETWORKING EXAMPLES")
    print("="*70)
    print("\nChoose an example:")
    print("  1. Coordinator Node")
    print("  2. Worker Node")
    print("  3. Hybrid Node (Coordinator + Worker)")
    print("  4. Auto-Discovery Worker (New!)")
    print("  5. Real-World Scenario (Inference Farm)")
    print("  6. Exit")
    print()
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == '1':
        example_coordinator()
    elif choice == '2':
        example_worker()
    elif choice == '3':
        example_hybrid()
    elif choice == '4':
        example_auto_discovery()
    elif choice == '5':
        example_real_world_scenario()
    elif choice == '6':
        print("Goodbye!")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
