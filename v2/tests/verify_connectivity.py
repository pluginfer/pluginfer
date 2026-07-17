"""
Connectivity Verification Tool
This script checks if the connectivity fixes (Firewall, UPnP, Discovery) are working correctly.
"""
import sys
import os
import socket
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("ConnectivityCheck")

def check_firewall():
    print("\n[1/4] Checking Firewall Status...")
    try:
        from utils.firewall_manager import FirewallManager
        print("   [*] Loading Firewall Manager...")
        
        if not FirewallManager.is_admin():
            print("   [!] WARNING: Script not running as Admin. Cannot query firewall rules accurately.")
        
        # Check if rules exist
        if FirewallManager._rule_exists(FirewallManager.RULE_NAME):
            print("   [+] ✅ Firewall Rule Found: 'Pluginfer Mesh Node (App)'")
        else:
            print("   [!] ❌ Firewall Rule MISSING. This node may be blocked on LAN/WAN.")
            
    except ImportError:
        print("   [!] Error importing FirewallManager.")
    except Exception as e:
        print(f"   [!] Error checking firewall: {e}")

def check_local_ip():
    print("\n[2/4] Checking Local IP Detection...")
    try:
        from core.networking import NetworkManager
        nm = NetworkManager()
        print(f"   [+] Local IP detected as: {nm.local_ip}")
        
        if nm.local_ip.startswith("127."):
            print("   [!] ⚠️  Warning: Local IP is loopback. LAN discovery will fail.")
        else:
            print("   [+] ✅ Local IP looks valid for LAN.")
            
    except Exception as e:
        print(f"   [!] Error checking Local IP: {e}")

def check_upnp():
    print("\n[3/4] Testing UPnP/Internet Connectivity...")
    try:
        from core.networking import NetworkManager
        nm = NetworkManager()
        
        print("   [*] Discovering Public IP...")
        pub_ip = nm.get_public_ip()
        if pub_ip:
            print(f"   [+] ✅ Public IP: {pub_ip}")
        else:
            print(f"   [!] ❌ Could not fetch Public IP (Internet down?)")
            
        print("   [*] Attempting UPnP Port Mapping...")
        if nm.enable_upnp():
            print("   [+] ✅ UPnP SUCCESS: Port 9000 mapped.")
        else:
            print("   [!] ❌ UPnP FAILED. Ensure Router has UPnP enabled.")
            
    except Exception as e:
        print(f"   [!] Error checking UPnP: {e}")

def check_discovery_bind():
    print("\n[4/4] Testing Discovery Binding...")
    try:
        from core.discovery import MeshDiscovery
        md = MeshDiscovery("test_probe", 9000, {})
        md.start()
        time.sleep(1)
        if md.running:
            print("   [+] ✅ Discovery Socket Bound Successfully.")
            print(f"   [*] Broadcast Address: {md.broadcast_ip if hasattr(md, 'broadcast_ip') else '<broadcast>'}")
        else:
            print("   [!] ❌ Discovery Failed to Start.")
        md.stop()
    except Exception as e:
        print(f"   [!] Error testing Discovery: {e}")

if __name__ == "__main__":
    import time
    print("="*60)
    print("PLUGINFER CONNECTIVITY DIAGNOSTIC")
    print("="*60)
    
    check_firewall()
    check_local_ip()
    check_upnp()
    check_discovery_bind()
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    input("Press Enter to exit...")
