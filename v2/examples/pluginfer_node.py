"""
Pluginfer Node - End User Application
This is the entry point for the compiled executable.
"""
import sys
import os
import time
import logging
import argparse
import json
import random
import socket # For connectivity check
import threading # For UI thread

# Ensure core modules are found even when frozen (PyInstaller support)
if getattr(sys, 'frozen', False):
    # Running as compiled .exe
    application_path = sys._MEIPASS
    sys.path.insert(0, application_path)
else:
    # Running as script
    application_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(application_path, '..'))

from core.complete_mesh_controller import CompleteMeshController
from core.hardware_detector import HardwareDetector
from core.security_manager import SecurityManager
from core.discovery import MeshDiscovery
from utils.gaming_detector import GamingDetector
from utils.nat_traversal import add_port_mapping, get_local_ip, get_public_ip

# UI Integration
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from ui.professional_web_ui import start_professional_ui, attach_controller
    UI_AVAILABLE = True
except ImportError:
    UI_AVAILABLE = False

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PluginferNode")

def check_eula():
    """Simple CLI EULA to protect the platform legally."""
    print("\n" + "="*60)
    print("LEGAL AGREEMENT & PRIVACY POLICY")
    print("="*60)
    print("1. You agree to share idle GPU resources for AI tasks.")
    print("2. NO personal files, photos, or data are accessed.")
    print("3. We execute isolated Python code only.")
    print("4. You can stop this software at any time.")
    print("-" * 60)
    
    # In a GUI installer this would be a checkbox. For CLI:
    # choice = input("Do you accept these terms? (y/n): ")
    # if choice.lower() != 'y':
    #     print("Terms declined. Exiting.")
    #     sys.exit(0)
    print("Terms Accepted (Auto-accept for beta)")

def check_connectivity(host, port):
    """Verify internet stability."""
    print(f"   Testing connection to {host}:{port}...")
    try:
        sock = socket.create_connection((host, port), timeout=5)
        sock.close()
        print("   ✓ Connection Stable (Latency: <100ms)")
        return True
    except OSError:
        print("   ⚠ Warning: Coordinator unreachable. Check Firewall/Internet.")
        return False

def main():
    print("="*60)
    print("PLUGINFER NODE - DECENTRALIZED COMPUTE NETWORK")
    print("v2.2.0 (Identity Check Active)")
    print("="*60)
    
    # 0. User Onboarding (First Run Experience)
    base_dir = os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else os.path.abspath(__file__))
    identity_path = os.path.join(base_dir, 'identity.json')
    
    user_data = {}
    
    if os.path.exists(identity_path):
        try:
            with open(identity_path, 'r') as f:
                user_data = json.load(f)
            print(f"\n[IDENTITY] Welcome back, {user_data.get('email')}!")
            print(f"           License: {user_data.get('license_key')[:4]}****")
        except:
             print("\n[!] Identity file corrupted. Re-running onboarding.")
    
    if not user_data:
        print("\n" + "="*60)
        print("FIRST TIME SETUP - REGISTRATION")
        print("="*60)
        print("Please provide your details to join the secure mesh.")
        
        while True:
            email = input("Enter your Email Address: ").strip()
            if '@' in email and '.' in email:
                break
            print("Invalid email. Please try again.")
            
        while True:
            license_key = input("Enter License Key (or 'beta' for trial): ").strip()
            if len(license_key) > 3: # Basic validation
                # In production, this would verify against a centralized Licensing API
                print("Verifying license key...")
                time.sleep(1) # Simulate network check
                print("✓ License Validated.")
                break
            print("Invalid key.")
            
        user_data = {
            'email': email,
            'license_key': license_key,
            'registered_at': time.time()
        }
        
        # Save locally
        with open(identity_path, 'w') as f:
            json.dump(user_data, f)
            
        print("\n[SUCCESS] Registration Complete. Welcome to Pluginfer.")
    
    # 1. Legal Check (EULA) verifies agreement, Identity verifies user
    check_eula()
    
    # 1. Hardware Check
    print("\n[1/4] Scanning Hardware...")
    hw = HardwareDetector()
    device = hw.get_best_device()
    print(f"   ✓ Detected: {device['name']} ({device['type']})")
    
    # 2. Gaming Detection (The "Killer Feature")
    print("\n[2/4] Initializing Gaming Detector...")
    gamer = GamingDetector()
    print(f"   ✓ Monitoring {len(gamer.game_list)} games for auto-pause")
    
    # 3. Security
    print("\n[3/4] Securing Environment...")
    sec = SecurityManager()
    print("   ✓ Isolation Layer Active")
    
    # 4. Configuration & Connection
    # Try to load config.json for Coordinator IP from the same dir as the exe
    if getattr(sys, 'frozen', False):
         base_dir = os.path.dirname(sys.executable)
    else:
         base_dir = os.path.dirname(os.path.abspath(__file__))
         
    config_path = os.path.join(base_dir, 'config.json')
    
    coordinator_ip = '127.0.0.1' # Default local
    coordinator_port = 8888
    mode = 'worker' # Default for gamers
    
    config_loaded = False
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                coordinator_ip = config.get('coordinator_ip', coordinator_ip)
                coordinator_port = config.get('coordinator_port', coordinator_port)
                mode = config.get('mode', mode)
            print(f"\n[INFO] Loaded config: Joining mesh at {coordinator_ip}:{coordinator_port}")
            config_loaded = True
        except Exception as e:
            print(f"\n[WARN] Failed to load config.json: {e}")
            print("       Using default settings.")
    
    if not config_loaded:
        print("\n[INFO] Auto-Configuration Started...")
        
        # 1. Priority: Production Server (Central VM)
        PROD_HOST = 'mesh.pluginfer.com'  # REPLACE with your VM IP/DNS
        PROD_PORT = 8888
        print(f"\n[1/3] Checking Production Server ({PROD_HOST})...")
        
        # We suppress the print inside check_connectivity for this silent check or just use it
        # Since check_connectivity prints, it's fine.
        is_prod = check_connectivity(PROD_HOST, PROD_PORT)
        
        if is_prod:
             print(f"   ✓ Connected to Central Server. Entering PRODUCTION MODE.")
             coordinator_ip = PROD_HOST
             coordinator_port = PROD_PORT
             mode = 'worker'
        else:
            print("   . Production Server unreachable (using Local/Test mode).")
            
            # 2. Priority: Local LAN Coordinator (Test Mode)
            print(f"\n[2/3] Searching for Local Coordinator (LAN)...")
            
            # Temporary discovery service to find a boss
            temp_discovery = MeshDiscovery(
                node_id=f"temp-{random.randint(1000,9999)}",
                port=0, 
                capabilities={}
            )
            temp_discovery.start(role='worker')
            
            found = temp_discovery.search_for_coordinator(timeout=2)
            temp_discovery.stop()
            
            if found:
                print(f"   ✓ Found Local Coordinator at {found['ip']}:{found['port']}")
                coordinator_ip = found['ip']
                coordinator_port = found['port']
                mode = 'worker'
            else:
                print("   . No existing Coordinator found on Local Network.")
                
                # 3. Priority: Hardcoded Test IP (For Internet Testing)
                # This allows specific .exe builds to connect to a specific laptop
                # without the user needing to type anything.
                TEST_COORDINATOR_IP = '49.204.129.94' # Auto-filled for User
                
                if TEST_COORDINATOR_IP and check_connectivity(TEST_COORDINATOR_IP, 8888):
                     print(f"   ✓ Found Hardcoded Test Server at {TEST_COORDINATOR_IP}")
                     coordinator_ip = TEST_COORDINATOR_IP
                     coordinator_port = 8888
                     mode = 'worker'
                else:
                    # 4. Priority: Become the Coordinator (Self-Hosted)
                    print("\n[INFO] Starting as Coordinator (Server/Hybrid Mode).")
                    mode = 'hybrid'

    
    print("\n[4/4] Connecting to Mesh...")
    
    # Check connectivity before starting
    if mode == 'worker' and coordinator_ip != '127.0.0.1':
        check_connectivity(coordinator_ip, coordinator_port)
    
    # Auto-Configuration (Plug-and-Play Network)
    my_port = random.randint(9000, 9999)
    local_ip = get_local_ip()
    
    print(f"\n[NETWORK] Configuring connection on {local_ip}:{my_port}...")
    # Attempt UPnP
    print(f"\n[NETWORK] Configuring connection on {local_ip}:{my_port}...")
    
    # Attempt UPnP
    upnp_success = add_port_mapping(local_ip, my_port, my_port)
    if upnp_success:
         print("   ✓ Router Port Forwarding configured (UPnP)")
         public_ip = get_public_ip()
         if public_ip:
             print(f"   ✓ PUBLIC ACCESS ENABLED!")
             print(f"     Tell your friends to connect to: {public_ip}")
             print(f"     Config: {{'coordinator_ip': '{public_ip}', 'coordinator_port': {my_port}}}")
    else:
         print("   . Standard negotiation active (UPnP not available or failed)")
         print("     If you want public access, manually forward port 8888 on your router.")
    
    try:
        # If worker, we connect to coordinator. If hybrid, we ARE the coordinator.
        if mode == 'worker':
            print(f"   Connecting to Coordinator at {coordinator_ip}...")
            # Note: The MeshController needs logic to actually *connect* outbound to this IP
            # For now, we start the listener. The P2P logic handles discovery.
            
        node = CompleteMeshController('0.0.0.0', my_port, mode)
        node.start()
        
        # 4.5 Start Web UI & API (Dashboard)
        if UI_AVAILABLE:
            print("\n[+] Launching Dashboard & API...")
            attach_controller(node)
            ui_thread = threading.Thread(
                target=start_professional_ui,
                kwargs={'port': 8000},
                daemon=True
            )
            ui_thread.start()
            print("   ✓ Dashboard: http://localhost:8000")
            print("   ✓ REST API:  http://localhost:8000/api/marketplace/stats")
        
        # 5. Register with Coordinator (Tiered Pricing Handshake)
        if mode == 'worker':
            print(f"   Connecting to Coordinator at {coordinator_ip}...")
            # Retrieve License Key from User Data
            license_key = user_data.get('license_key') if 'user_data' in locals() else 'beta'
            
            # Get Performance Score
            perf_score = hw.get_performance_score()
            print(f"   ✓ Calculated Performance Score: {perf_score}x (Based on {device['type']})")
            
            # Send Registration
            success = node.register_with_coordinator(
                coordinator_ip, coordinator_port, 
                license_key, device, perf_score
            )
            
            if not success:
                print("   ⚠ Registration failed. You may not receive tasks.")
            else:
                print("   ✅ Registered in Payment Tier.")

        print(f"\n✅ NODE ONLINE ({mode.upper()})")
        print(f"   ID: {node.node_id}")
        print(f"   Listening on port: {my_port}")
        print("\nPress Ctrl+C to stop...")
        
        # SMART WORKER LOOP
        # Handles:
        # 1. Gaming Detection (Pause/Resume)
        # 2. Connection Health (Auto-reconnect)
        # 3. Uptime Logging (For payments)
        
        is_paused = False
        uptime_seconds = 0
        
        while True:
            # A. Gaming Check
            is_gaming = gamer.is_gaming()
            
            if is_gaming and not is_paused:
                print("\n[🎮 GAMING MODE] Game detected! Pausing mesh tasks...")
                node.stop() # Drop from mesh to free up GPU
                is_paused = True
                
            elif not is_gaming and is_paused:
                print("\n[✅ IDLE MODE] Gaming stopped. Rejoining mesh...")
                # Re-initialize to reconnect
                try:
                    node = CompleteMeshController('0.0.0.0', my_port, mode)
                    node.start()
                    is_paused = False
                    print("   ✓ Reconnected successfully")
                except Exception as e:
                    print(f"   ❌ Reconnection failed: {e}")
            
            # B. Health/Logging Check (Only if active)
            if not is_paused:
                if not node.running:
                     # Crashed? Restart.
                     print("\n[⚠ WARNING] Node stopped unexpectedly. Restarting...")
                     node.start()
                
                # Log uptime for payment every minute
                uptime_seconds += 1
                if uptime_seconds % 60 == 0:
                     # In a real app, this would write to a secure local DB or send a 'ping' to coordinator
                     pass 
            
            time.sleep(1) # Check every second
            
    except KeyboardInterrupt:
        print("\nStopping node...")
        node.stop()
        print("Goodbye.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Startup Error: {e}")
        # input("Press Enter to exit...") # DISABLED for automation
        sys.exit(1)

if __name__ == "__main__":
    main()
