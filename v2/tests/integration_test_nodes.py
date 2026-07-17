import subprocess
import sys
import os
import time
import requests

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NODE_SCRIPT = os.path.join(PROJECT_ROOT, 'pluginfer_node.py')

def start_node(port, dash_port, mode, title):
    cmd = [
        sys.executable, 
        NODE_SCRIPT, 
        '--port', str(port), 
        '--dash-port', str(dash_port),
        '--swarm', 'test_swarm'
    ]
    print(f"START {title} {port}")
    
    log_file = open(f"test_node_{port}.log", "w", encoding='utf-8')
    process = subprocess.Popen(
        cmd, 
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    return process, log_file

def check_dashboard(port):
    try:
        response = requests.get(f"http://localhost:{port}/", timeout=2)
        return response.status_code == 200
    except:
        return False

def run_test():
    print("TEST START")
    
    # 1. Start Coordinator
    coord_proc, coord_log = start_node(9001, 8001, 'coordinator', "Coordinator")
    time.sleep(5)
    
    # 2. Start Worker
    worker_proc, worker_log = start_node(9002, 8002, 'worker', "Worker")
    
    print("WAITING 15s")
    time.sleep(15)
    
    # 3. Verification
    print("VERIFYING")
    
    coord_up = check_dashboard(8001)
    worker_up = check_dashboard(8002)
    
    print(f"COORD UI: {coord_up}")
    print(f"WORKER UI: {worker_up}")
    
    coord_log.close()
    worker_log.close()
    
    # Read logs safely
    try:
        with open(f"test_node_9001.log", "r", encoding='utf-8', errors='ignore') as f:
            coord_output = f.read()
    except:
        coord_output = ""
        
    try:
        with open(f"test_node_9002.log", "r", encoding='utf-8', errors='ignore') as f:
            worker_output = f.read()
    except:
        worker_output = ""
        
    if "Peers: 1" in coord_output or "Peers: 1" in worker_output:
        print("SUCCESS PEERS FOUND")
    elif "Connected to mesh" in worker_output or "Accepted connection" in coord_output:
         print("SUCCESS CONNECTION FOUND")
    else:
        print("FAILURE NO CONNECTION")
        print("LOG TAILS:")
        try:
            print(coord_output[-500:].encode('ascii', 'replace').decode())
            print(worker_output[-500:].encode('ascii', 'replace').decode())
        except:
            print("ERROR PRINTING LOGS")

    # 4. Cleanup
    print("CLEANUP")
    coord_proc.terminate()
    worker_proc.terminate()
    try:
        coord_proc.wait(timeout=2)
    except:
        coord_proc.kill()
    try:
        worker_proc.wait(timeout=2)
    except:
        worker_proc.kill()
    
if __name__ == "__main__":
    run_test()
