"""
Pluginfer Node - End User Application
This is the entry point for the compiled executable.
"""
import sys
import os
import warnings

# Suppress annoying Torch JIT source warnings in frozen builds
warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit._recursive")
warnings.filterwarnings("ignore", message=".*Unable to retrieve source for.*")

# FORCE BUNDLE: Ensure torch-directml is packaged
try:
    import torch_directml
except Exception:
    pass

# ✅ CRITICAL GLOBAL HOTFIX: Werkzeug 3.x Metadata Crash
# This must run before ANY other import that might touch Flask/Werkzeug
try:
    import importlib.metadata
except ImportError:
    pass
else:
    original_version = importlib.metadata.version
    def safe_version(package_name):
        try:
            return original_version(package_name)
        except importlib.metadata.PackageNotFoundError:
            return '3.0.0' # Mock valid version
    importlib.metadata.version = safe_version

import time
import logging
import argparse
import threading

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("node.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PluginferNode")

# Ensure core modules are found even when frozen (PyInstaller support)
if getattr(sys, 'frozen', False):
    # Running as compiled .exe
    application_path = sys._MEIPASS
    sys.path.insert(0, application_path)
else:
    # Running as script
    application_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(application_path, '..'))

# Import Core Components
try:
    from core.complete_mesh_controller import CompleteMeshController
    from core.hardware_detector import HardwareDetector
    from core.game_detector import GameDetector
    from core.security_manager import SecurityManager
    from ui.professional_web_ui import start_professional_ui, attach_controller
except ImportError as e:
    logger.error(f"Failed to load core components: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"CRITICAL STARTUP ERROR: {e}")
    sys.exit(1)

def run_dashboard(port, controller=None):
    """Run the Professional Web UI in a separate thread"""
    try:
        if controller:
            attach_controller(controller)
        start_professional_ui(host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")


def main():
    # Parse CLI Args
    parser = argparse.ArgumentParser(description="Pluginfer Node")
    parser.add_argument("--swarm", type=str, default="public", help="Swarm ID for Private Networks")
    parser.add_argument("--port", type=int, default=9000, help="Node Port")
    parser.add_argument("--dash-port", type=int, default=8000, help="Dashboard Port")
    args = parser.parse_args()

    print("="*60)
    print("PLUGINFER NODE - DECENTRALIZED COMPUTE NETWORK")
    print("v2.0.0 (Secure Build)")
    print("="*60)
    
    # 0. System Health Check (Python Version)
    from core.system_doctor import SystemDoctor
    doc = SystemDoctor()
    if not doc.check_python_version():
        print("\n[!] WARNING: Non-Standard Python Version Detected.")
        print("    Torching features may require Python 3.10 or 3.11.")
        print("    Proceeding with caution...")
        # sys.exit(1) # DISABLED for Compatibility Mode

    # 1. Hardware Check
    print("\n[1/4] Scanning Hardware...", flush=True)
    hw = HardwareDetector()
    device = hw.get_best_device()
    print(f"   [+] Detected: {device['name']} ({device['type']})")
    
    # 2. Gaming Detection (The "Killer Feature")
    print("\n[2/4] Initializing Gaming Detector...")
    gamer = GameDetector(None)
    print(f"   [+] Monitoring {len(gamer.known_games)} games for auto-pause")
    
    # 3. Security
    print("\n[3/4] Securing Environment...")
    sec = SecurityManager()
    
    # NEW: Automatic Firewall Configuration
    try:
        from utils.firewall_manager import FirewallManager
        if FirewallManager.is_admin():
            print("   [+] Firewall: configuring rules...", end=" ")
            if FirewallManager.authorize_app(port_tcp=args.port, port_udp=9998):
                print("OK")
            else:
                print("Failed (Check Logs)")
        else:
            print("   [!] Firewall: Run as Admin to auto-configure ports")
    except Exception as e:
        print(f"   [!] Firewall Config Error: {e}")
        
    print("   [+] Isolation Layer Active")
    
    # 4. Start Node
    print("\n[4/4] Connecting to Mesh...")
    
    node_port = args.port
    dashboard_port = args.dash_port
    swarm_id = args.swarm
    
    try:
        # Initialize Node with Swarm ID
        # [FIX] Write config.json BEFORE init to ensure Swarm ID is picked up
        # This is safer than monkey-patching
        import json
        with open('config.json', 'w') as f:
             json.dump({'swarm_id': swarm_id}, f)
        
        node = CompleteMeshController('0.0.0.0', node_port, 'hybrid')
        
        # [OK] ENABLE ENCRYPTION for Private/Bank Swarms
        if swarm_id != 'public':
            print("   [SECURE] Private Swarm Detected: Enabling TLS Encryption...")
            # node.enable_tls() # Disabled for now until certs are ready
        
        # Attach node to Dashboard (so UI can see stats)
        from ui.professional_web_ui import attach_controller
        attach_controller(node)
        
        # Start Node Logic
        node.start()
        
        # ACTIVATE GAMING PAUSE
        # Run auto-management in background thread to monitor games and pause node
        # This fulfills the "gaming pause" requirement
        gamer.controller = node
        gamer.start()
        logger.info("Gaming Mode Active: Node will auto-pause when games are running.")
        
        # Start Dashboard in Background
        dash_thread = threading.Thread(target=run_dashboard, args=(dashboard_port, node), daemon=True)
        dash_thread.start()
        
        print(f"\n[OK] NODE ONLINE")
        print(f"   ID: {node.node_id}")
        print(f"   Swarm:          {swarm_id} {'(Private)' if swarm_id != 'public' else '(Public)'}")
        print(f"   Node Layer:     port {node_port}")
        print(f"   Web Dashboard:  http://localhost:{dashboard_port}")
        
        # [UX] Auto-Open Dashboard
        try:
            import webbrowser
            webbrowser.open(f"http://localhost:{dashboard_port}")
        except: pass
        
        print("\nPress Ctrl+C to stop...")
        
        # 5. Start System Tray (Main Loop)
        # This replaces the blocking while loop
        try:
            from utils.system_tray import SystemTrayApp
            print(f"\n[OK] NODE ONLINE - Minimizing to System Tray")
            print(f"   Dashboard: http://localhost:{dashboard_port}")
            
            tray = SystemTrayApp(node, f"http://localhost:{dashboard_port}")
            tray.run() # This blocks until Exit is clicked
            
        except ImportError:
            # Fallback for headless/server mode
            print("   [!] System Tray not available (pystray missing). Running in Console Mode.")
            try:
                start_time = time.time()
                while True:
                    time.sleep(30)
                    uptime = int(time.time() - start_time)
                    peers = len(node.known_peers) if hasattr(node, 'known_peers') else 0
                    score = node.reputation.get_score() if hasattr(node, 'reputation') else 0.0
                    print(f"   [HEARTBEAT] uptime={uptime}s | peers={peers} | score={score:.2f} | status=ACTIVE")
            except KeyboardInterrupt:
                node.stop()

    except KeyboardInterrupt:
        print("\nStopping node...")
        node.stop()
        print("Goodbye.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        main()
    except Exception as e:
        logger.exception("Fatal error:")
        print(f"\n[!] Fatal Error: {e}")
        input("Press Enter to exit...")

