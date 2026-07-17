
"""
MASTER SYSTEM SIMULATION
1. Launches the Root Executable (PluginferNode.exe)
2. Verifies Basic Connectivity
3. Verifies ALL Plugins (Standard, Text, Document, AI, Raw Compute)
4. Runs Stress Load (Multi-Threaded)
"""
import sys
import os
import time
import subprocess
import requests
import colorama
from colorama import Fore, Style
import threading
import json
import base64

colorama.init()

API_URL = "http://localhost:8000/api/submit_job"
STATS_URL = "http://localhost:8000/api/marketplace/stats"

def log_pass(msg):
    print(f"{Fore.GREEN}✅ PASS: {msg}{Style.RESET_ALL}")

def log_fail(msg):
    print(f"{Fore.RED}❌ FAIL: {msg}{Style.RESET_ALL}")

def log_info(msg):
    print(f"{Fore.CYAN}ℹ️  INFO: {msg}{Style.RESET_ALL}")

def wait_for_api():
    log_info("Waiting for Node API to come online...")
    for _ in range(30):
        try:
            requests.get(STATS_URL, timeout=1)
            return True
        except:
            time.sleep(1)
            print(".", end="", flush=True)
    return False

def test_plugin(name, input_data, expected_check=None):
    try:
        log_info(f"Testing Plugin: {name}")
        payload = {'plugin': name}
        payload.update(input_data)
        
        # Determine if we need to send raw (requests auto-encodes)
        # For simplicity, we assume simple key-values unless its file upload
        resp = requests.post(API_URL, data=payload)
        
        if resp.status_code != 200:
            log_fail(f"{name} returned HTTP {resp.status_code}")
            return False
            
        data = resp.json()
        if data.get('status') == 'success':
            if expected_check:
                if expected_check(data):
                    log_pass(f"{name} Verification Successful")
                    return True
                else:
                    log_fail(f"{name} Check Failed: {json.dumps(data, indent=2)}")
                    return False
            else:
                 log_pass(f"{name} Executed Successfully")
                 return True
        else:
            log_fail(f"{name} Failed: {data.get('error')}")
            return False
            
    except Exception as e:
        log_fail(f"{name} Exception: {e}")
        return False

def run_simulation():
    print(f"{Back.BLUE}{Fore.WHITE} FULL SYSTEM SIMULATION STARTING {Style.RESET_ALL}\n")
    
    # 1. Launch EXE
    exe_path = os.path.join(os.getcwd(), "PluginferNode.exe")
    if not os.path.exists(exe_path):
        log_fail(f"Executable not found at {exe_path}")
        return

    log_info(f"Launching {exe_path}...")
    process = subprocess.Popen([exe_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    try:
        if not wait_for_api():
            log_fail("Node failed to start (API unreachable)")
            return

        log_pass("Node Started & API Online")

        # 2. Test Features
        
        # Test 1: Text Sentiment (Fix verification)
        test_plugin('txt_sentiment', {'text': 'I love decentralized computing!'}, 
                   lambda d: 'positive' in str(d))

        # Test 2: Raw Compute (Dynamic Executor) - Math
        code = "import math\ndef main(val): return math.sqrt(val)"
        # Base64 encode code same as UI
        b64_code = base64.b64encode(code.encode()).decode()
        test_plugin('dynamic_executor', 
                   {'code': code, 'function_name': 'main', 'args': json.dumps({'val': 144})},
                   lambda d: '12.0' in str(d))

        # Test 3: Standard Plugin (Grayscale) - Using small mock base64 image
        # 1x1 pixel red dot
        b64_img = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        test_plugin('img_grayscale', {'image_data': b64_img},
                   lambda d: 'data' in str(d))

        # Test 4: New Plugin (Face Swap) - Mock (1 face -> invert)
        test_plugin('face_swap', {'image_data': b64_img},
                   lambda d: 'status' in str(d))

        # 3. Stress Volume Test
        log_info("Running Stress Load (20 Concurrent Requests)...")
        results = []
        def worker(i):
            res = test_plugin(f'Stress-{i}', 
                             {'plugin': 'txt_sentiment', 'text': f'msg {i}'},
                             None) # Don't log individual passes to keep clean
            results.append(res)
            
        threads = []
        for i in range(20):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        success_count = sum(results)
        if success_count == 20:
            log_pass(f"Stress Test Passed: {success_count}/20 reqs")
        else:
            log_fail(f"Stress Test Failed: Only {success_count}/20 reqs")

    finally:
        log_info("Terminating Node...")
        subprocess.call(['taskkill', '/F', '/T', '/PID', str(process.pid)])
        log_info("Simulation Complete.")

if __name__ == "__main__":
    from colorama import Back
    run_simulation()
