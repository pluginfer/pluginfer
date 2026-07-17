"""
Verify AI Sentinel
Simulates attacks to test if the AI blocks them while allowing normal users.
"""
import sys
sys.path.append('..')

import time
import random
from core.ai_sentinel import AISentinel

def verify_sentinel():
    print("🛡️ Initializing AI Sentinel (Sensitivity: 3.5 sigma)...")
    sentinel = AISentinel(sensitivity=3.5)
    
    # Scene 1: The Normal User
    # Sends 1 small request every second
    print("\n[Scene 1] Checking 'Normal User' (Should be ALLOWED)")
    user_id = "192.168.1.100"
    allowed_count = 0
    blocked_count = 0
    
    for i in range(15): # 15 seconds
        success = sentinel.analyze_request(user_id, payload_size=512)
        if success:
            allowed_count += 1
            # print(".", end="", flush=True) # Reduced noise
        else:
            blocked_count += 1
            # print("X", end="", flush=True) # Reduced noise
        time.sleep(0.1) # Accelerated time
        
    print(f"\n   Stats: {allowed_count} Allowed, {blocked_count} Blocked")
    if blocked_count == 0:
        print("   ✅ PASS: Normal user was not bothered.")
    else:
        print("   ❌ FAIL: Normal user was blocked!")

    # Scene 2: The DDoS Bot
    # Sends 100 requests instantly
    print("\n[Scene 2] Checking 'DDoS Bot' (Should be BLOCKED)")
    bot_id = "10.0.0.666"
    allowed_count = 0
    blocked_count = 0
    
    for i in range(100):
        # No sleep = Instant
        success = sentinel.analyze_request(bot_id, payload_size=512)
        if success:
            allowed_count += 1
        else:
            blocked_count += 1
            
    print(f"   Stats: {allowed_count} Allowed, {blocked_count} Blocked")
    if blocked_count > 0:
        print("   ✅ PASS: Bot was detected and blocked.")
    else:
        print("   ❌ FAIL: Bot was let through!")

    # Scene 3: The Data Exfiltration (Anomaly)
    # Behaving normally, then suddenly sending 10MB
    print("\n[Scene 3] Checking 'Anomaly' (Should be FLAGGED)")
    spy_id = "172.16.5.5"
    
    # Establish baseline
    print("   Establishing baseline...", end="")
    for i in range(20):
        sentinel.analyze_request(spy_id, payload_size=1024 + random.randint(0, 50))
        time.sleep(0.06) # Sleep just enough to satisfy DDoS check (20 fps = 16 req/sec < 20 req/sec limit)
    print(" Done.")
    
    # The Attack
    print("   Sending 10MB payload...")
    success = sentinel.analyze_request(spy_id, payload_size=10 * 1024 * 1024)
    
    # Note: Our conservative logic currently *Warns* on size but doesn't ban immediately unless threat score is high.
    # Let's check the threat score.
    profile = sentinel.profiles[spy_id]
    print(f"   Threat Score: {profile.threat_score}/100")
    
    if profile.threat_score > 0:
         print("   ✅ PASS: Anomaly was scored.")
    else:
         print("   ❌ FAIL: Anomaly ignored.")

if __name__ == "__main__":
    verify_sentinel()
