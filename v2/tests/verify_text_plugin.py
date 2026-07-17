
import requests
import json
import time

# Configuration
API_URL = "http://localhost:8000/api/submit_job"

def verify_text_plugin():
    print(f"Testing Text Plugin API at {API_URL}...")
    
    # 1. Payload
    text_content = "The new dynamic compute feature is absolutely fantastic and powerful!"
    print(f"Input Text: '{text_content}'")
    
    payload = {
        'plugin': 'txt_sentiment',
        'text': text_content
    }
    
    # 2. Send Request
    try:
        response = requests.post(API_URL, data=payload)
        
        if response.status_code != 200:
            print(f"❌ HTTP Error: {response.status_code}")
            return

        data = response.json()
        print(f"\nResponse:")
        print(json.dumps(data, indent=2))
        
        # 3. Verify
        if data.get('status') == 'success':
            # Handle potential nesting
            res = data.get('result', {})
            if 'result' in res: res = res['result']
            
            sentiment = res.get('sentiment')
            if sentiment == 'positive':
                 print("\n✅ VERIFICATION PASSED: Sentiment is Positive")
            else:
                 print(f"\n❌ VERIFICATION FAILED: Expected positive, got {sentiment}")
        else:
             print("\n❌ TASK FAILED")

    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    verify_text_plugin()
