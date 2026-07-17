
import requests
import json
import base64
import time
import sys

BASE_URL = "http://localhost:8000"

# Dummy 1x1 PNG
IMG_BYTES = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgADNjd8qAAAAABJRU5ErkJggg==")

def submit_task(plugin, input_type, input_value, args=None):
    print(f"\nTesting '{plugin}'...", end=" ", flush=True)
    
    payload = {'plugin': plugin}
    files = None
    
    if input_type == 'image':
        files = {'file': ('test.png', IMG_BYTES, 'image/png')}
    elif input_type == 'text':
        payload['text'] = input_value
    elif input_type == 'code':
        payload['code'] = input_value
        if args:
            payload['args'] = json.dumps(args)
    elif input_type == 'json':
        # specific plugins might need raw json in 'args' or specific fields?
        # The UI maps 'text', 'code', 'file'.
        # Assuming most other plugins take inputs via 'args' if not text/file.
        # But wait, api_submit_job ONLY maps 'text', 'code', 'file'.
        # It DOES map 'args' -> input_data['args'].
        # So for math plugins, we might need to rely on 'args'.
        # But 'math_prime_factors' expects 'number' at top level? 
        # API doesn't seem to support arbitrary top-level fields easily without modification.
        # Let's check api_submit_job again. It DOES NOT map arbitrary fields. 
        # It expects plugins to look in 'args', or 'text', or 'data'.
        # If a plugin expects 'number' at root, it won't work with this API unless we verify plugin reads from 'args'.
        if args:
             payload['args'] = json.dumps(args)

    try:
        resp = requests.post(f"{BASE_URL}/api/submit_job", data=payload, files=files)
        if resp.status_code != 200:
            print(f"[HTTP ERROR] {resp.status_code}: {resp.text}")
            return False
            
        data = resp.json()
        if data.get('status') != 'success':
            print(f"[Rejected]: {data}")
            return False
            
        task_id = data['task_id']
        
        # Poll
        for _ in range(30):
            time.sleep(0.5)
            r = requests.get(f"{BASE_URL}/api/results/{task_id}")
            res = r.json()
            if res.get('status') == 'success':
                print("[PASSED]")
                return True
            elif res.get('status') == 'error':
                print(f"[FAILED]: {res.get('error')}")
                return False
                
        print("[TIMEOUT]")
        return False
        
    except Exception as e:
        print(f"[EXCEPTION]: {e}")
        return False

def run_suite():
    print("STARTING COMPREHENSIVE PLUGIN VERIFICATION")
    print(f"Target: {BASE_URL}")
    print("-" * 50)
    
    results = {}
    
    # 1. Image Plugins
    img_plugins = ['img_grayscale', 'img_invert', 'img_blur', 'img_rotate', 'img_resize']
    for p in img_plugins:
        results[p] = submit_task(p, 'image', None)
        
    # 2. Text Plugins
    txt_plugins = ['txt_upper', 'txt_wordcount', 'txt_sentiment', 'txt_anonymize']
    for p in txt_plugins:
        results[p] = submit_task(p, 'text', "Hello Pluginfer World!")
        
    # 3. Dynamic Executor (Python Code)
    py_code = """
def main(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
"""
    results['dynamic_executor'] = submit_task('dynamic_executor', 'code', py_code, args={'n': 10})
    
    print("-" * 50)
    passed_count = sum(results.values())
    print(f"Passed: {passed_count}/{len(results)}")
    
    if all(results.values()):
        print("[ALL TESTS PASSED]")
        sys.exit(0)
    else:
        print("[SOME TESTS FAILED]")
        sys.exit(1)

if __name__ == "__main__":
    run_suite()
