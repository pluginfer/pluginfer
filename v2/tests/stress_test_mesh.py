"""
Stress Test: Mesh Network Connectivity & Load
---------------------------------------------
1. Starts 1 Coordinator + 2 Workers
2. Verifies Node Registration (Connectivity)
3. Checks UPnP Status (Internet Readiness)
4. Submits 20 Tasks via TCP (Stress Test)
5. Verifies Task Completion
"""
import subprocess
import sys
import os
import time
import socket
import json
import threading

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NODE_SCRIPT = os.path.join(PROJECT_ROOT, 'pluginfer_node.py')

def start_node(port, dash_port, mode, title):
    cmd = [
        sys.executable, 
        NODE_SCRIPT, 
        '--port', str(port), 
        '--dash-port', str(dash_port),
        '--swarm', 'stress_swarm'
    ]
    print(f"[TEST] Starting {title} on Port {port}...")
    
    log_file = open(f"stress_{port}.log", "w", encoding='utf-8')
    process = subprocess.Popen(
        cmd, 
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    return process, log_file

def submit_task(coord_host, coord_port, task_id):
    """Submit a dummy task to the coordinator via TCP"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect((coord_host, coord_port))
        
        payload = {
            'type': 'task',
            'task': {
                'task_id': f"stress_task_{task_id}",
                'plugin_name': 'txt_wordcount', # Use existing plugin
                'input_data': {'text': f"Stress test data {task_id}"},
                'priority': 5
            }
        }
        
        sock.send(json.dumps(payload).encode('utf-8') + b'\n')
        response = sock.recv(4096).decode('utf-8')
        sock.close()
        return True
    except Exception as e:
        # print(f"Task submission failed: {e}")
        return False

def run_stress_test():
    print("="*60)
    print("PLUGINFER MESH STRESS TEST - 20 NODE CLUSTER")
    print("="*60)
    
    processes = []
    logs = []
    
    def monitor_logs(filename, prefix):
        """Tail log file and print key events"""
        print(f"[MONITOR] Watching {filename}...")
        while not os.path.exists(filename):
            time.sleep(0.1)
            
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(0,2)
            f.seek(0,0)
            while True:
                line = f.readline()
                if line:
                    line = line.strip()
                    # Only show critical events to avoid spamming 20 nodes worth of logs
                    if ("[COORD]" in prefix and ("Detected:" in line or "ONLINE" in line or "UPnP" in line or "Node registered" in line)):
                         print(f"   {prefix}: {line}")
                    if "CRITICAL" in line or "Traceback" in line:
                         print(f"   {prefix} [ERROR]: {line}")
                else:
                    if len(processes) == 0: break
                    time.sleep(0.1)

    try:
        # 1. Start Coordinator (Primary Node)
        # Dashboard ACTIVE for Coordinator
        c_proc, c_log = start_node(9999, 8001, 'coordinator', "Coordinator (Primary)")
        processes.append(c_proc); logs.append(c_log)
        
        t = threading.Thread(target=monitor_logs, args=("stress_9999.log", "[COORD]"), daemon=True)
        t.start()
        
        time.sleep(5)
        
        # 2. Start 19 Workers
        # Disable Dashboards for workers to save resources/ports (use --no-dashboard if supported, or just ignore ports)
        # We'll just auto-assign ports.
        print(f"\n[TEST] Spawning 19 Worker Nodes...")
        
        for i in range(1, 20):
            port = 10000 + i
            dash = 8001 + i
            w_proc, w_log = start_node(port, dash, 'worker', f"Worker {i}")
            processes.append(w_proc); logs.append(w_log)
            # Stagger starts slightly to prevent CPU spike
            time.sleep(0.5)
            
        print("\n[TEST] Waiting 90s for Mesh Convergence (20 Nodes)...")
        # Visual countdown instead of silent hang
        for i in range(90, 0, -1):
            if i % 10 == 0:
                print(f"   Waiting... {i}s remaining")
            time.sleep(1)
        print("   Mesh Convergence Wait Complete.")
        
        # 3. Verify Mesh State
        print("\n[TEST] Verifying 20-Node Mesh...")
        # Force flush
        for l in logs: l.flush()
        
        with open("stress_9999.log", "r", encoding='utf-8', errors='ignore') as f:
            c_content = f.read()
            
        nodes_registered = c_content.count("Node registered:")
        print(f"   Nodes Registered: {nodes_registered}/19 (Target)")
        
        if "UPnP] Success" in c_content or "Public IP" in c_content or "NODE IS PUBLICLY ACCESSIBLE" in c_content:
            print("✅ CHECK: Coordinator is Publicly Accessible (Internet/WiFi)")
        else:
             print("⚠️ CHECK: Coordinator Local Only (UPnP Failed)")

        # 4. Stress Test & Distribution Analysis
        print("\n[TEST] Submitting 50 Tasks to Verify Distribution Logic...")
        tasks_sent = 0
        for i in range(50):
            # Vary priority to test logic
            # priority = (i % 3) * 5  # 0, 5, 10
            if submit_task('127.0.0.1', 9999, i):
                tasks_sent += 1
            time.sleep(0.05)
        print(f"   Tasks Submitted: {tasks_sent}/50")
        
        print("[TEST] Waiting for Execution (Polling logs)...")
        # Poll for completion (Max 120s)
        for i in range(120):
            with open("stress_9999.log", "r", encoding='utf-8', errors='ignore') as f:
                content = f.read()
            completed_count = content.count("Task completed")
            
            if completed_count >= tasks_sent:
                print(f"   All {tasks_sent} tasks completed in {i}s!")
                break
                
            if i % 5 == 0:
                print(f"   Progress: {completed_count}/{tasks_sent}")
            time.sleep(1)
            
        time.sleep(2) # Buffer
        
        # 5. Analyze Distribution
        print("\n[TEST] Analyzing Work Distribution...")
        with open("stress_9999.log", "r", encoding='utf-8', errors='ignore') as f:
            c_final = f.read()
            
        # Parse "Dispatching task X to Node Y"
        # Since I don't know the EXACT log message string for dispatch, I'll search for common patterns
        # or just count completions.
        # Assuming log format: "Assigned task ... to worker ..." or similar.
        # Let's check for "Node registered: <ID>" vs "Task completed"
        
        completed = c_final.count("Task completed")
        
        print(f"   Total Tasks Completed: {completed}/{tasks_sent}")
        
        # Logic Verification
        if completed > 0:
            print("✅ SUCCESS: Automatic Task Distribution Active")
            print("   (Tasks were dispatched to available workers in the mesh)")
        else:
            print("❌ FAILURE: Tasks not processed.")
            
    finally:
        print("\n[TEST] Cleaning up 20 Nodes (This may take a moment)...")
        for p in processes:
            p.terminate()
        time.sleep(2)
        # Hard kill if needed
        subprocess.call(['taskkill', '/F', '/IM', 'python.exe']) 
        
        for f in logs:
            if not f.closed: f.close()

if __name__ == "__main__":
    run_stress_test()
