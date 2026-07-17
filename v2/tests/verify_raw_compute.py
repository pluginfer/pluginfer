
import requests
import base64
import json
import time

# Configuration
API_URL = "http://localhost:8000/api/submit_job"

def verify_raw_compute():
    print(f"Testing Raw Compute API at {API_URL}...")
    
    # 1. Define Python Code to execute
    # A simple function that imports math (checking sandbox) and returns a calculation
    script = """
import math

def calculate_hypotenuse(a, b):
    return math.sqrt(a**2 + b**2)
"""

    print("Payload Script:")
    print(script.strip())

    # 2. Prepare Payload
    payload = {
        'plugin': 'dynamic_executor',
        'code': script, # Text will be base64 encoded by UI, but if sending RAW via API...
                        # WAIT! The API `api_submit_job` expects:
                        # request.form.get('code') which is the raw string from textarea
                        # and THEN it base64 encodes it internally.
                        # So we send raw string here.
        'function_name': 'calculate_hypotenuse',
        'args': json.dumps({'a': 3, 'b': 4})
    }
    
    # 3. Send Request
    try:
        start_time = time.time()
        response = requests.post(API_URL, data=payload)
        
        if response.status_code != 200:
            print(f"❌ HTTP Error: {response.status_code}")
            print(response.text)
            return

        data = response.json()
        print(f"\nResponse ({time.time() - start_time:.2f}s):")
        print(json.dumps(data, indent=2))
        
        # 4. Verify Result
        if data.get('status') == 'success':
            result_block = data.get('result', {})
            # Unwrap nested result if present (old issue)
            if 'result' in result_block:
                result_block = result_block['result']
                
            value = result_block.get('result')
            
            if value == 5.0:
                print("\n✅ VERIFICATION PASSED: 3^2 + 4^2 = 5^2")
            else:
                print(f"\n❌ VERIFICATION FAILED: Expected 5.0, got {value}")
        else:
             print("\n❌ TASK FAILED")

    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("Ensure PluginferNode is running with UI enabled.")

if __name__ == "__main__":
    verify_raw_compute()
