
import requests
import time
import sys
import json

BASE_URL = "http://localhost:8000"

def wait_for_server(timeout=60):
    print(f"Waiting for server at {BASE_URL}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{BASE_URL}/api/marketplace/stats")
            if resp.status_code == 200:
                print("[Server is UP!]")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
        print(".", end="", flush=True)
    print("\n[Server timed out.]")
    return False

def test_grayscale():
    print("\nTesting 'img_grayscale' (The PicklingError Trigger)...")
    # Create a dummy 1x1 PNG image (base64 decoded back to bytes)
    import base64
    img_bytes = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgADNjd8qAAAAABJRU5ErkJggg==")
    
    files = {
        'file': ('test.png', img_bytes, 'image/png')
    }
    data = {
        'plugin': 'img_grayscale'
    }
    
    try:
        # Submit (Use files= for File Upload)
        resp = requests.post(f"{BASE_URL}/api/submit_job", data=data, files=files)
        try:
            data = resp.json()
        except:
            print(f"[Request Failed] Status: {resp.status_code}, Text: {resp.text}")
            return False
        
        # API returns 'success' even if queued, so we check that.
        if data.get('status') != 'success':
            print(f"[Submission Failed] {data}")
            return False
            
        task_id = data['task_id']
        print(f"Task Submitted: {task_id}")
        
        # Check if result is already included (short polling)
        if 'result' in data and 'status' in data['result']:
             res_status = data['result']['status']
             if res_status == 'success':
                 print(f"[SUCCESS] Result (Immediate): {data['result']}")
                 return True
             elif res_status == 'error':
                 print(f"[FAILED] Error (Immediate): {data['result'].get('error')}")
                 return False
                 
        # Poll Result
        for _ in range(20):
            time.sleep(1)
            res_resp = requests.get(f"{BASE_URL}/api/results/{task_id}")
            res_data = res_resp.json()
            
            if res_data.get('status') == 'success':
                print(f"[SUCCESS] Result: {res_data}")
                return True
            elif res_data.get('status') == 'error':
                print(f"[FAILED] Error: {res_data.get('error')}")
                return False
                
        print("[Task Timed Out]")
        return False
        
    except Exception as e:
        print(f"[Exception] {e}")
        return False

if __name__ == "__main__":
    if not wait_for_server():
        sys.exit(1)
        
    if test_grayscale():
        print("\n[VERIFICATION PASSED] The build is NOT broken.")
        sys.exit(0)
    else:
        print("\n[VERIFICATION FAILED] The build is STILL broken.")
        sys.exit(1)
