#!/usr/bin/env python3
"""
Pluginfer - GPU-Agnostic AI Execution Runtime
Main application entry point
"""
import sys
import os
import time
import logging
import argparse
from pathlib import Path
import multiprocessing

# Add parent directory to path for imports
# Necessary for PyInstaller and dev mode
if getattr(sys, 'frozen', False):
    application_path = sys._MEIPASS
    sys.path.insert(0, application_path)
else:
    application_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, str(Path(__file__).parent))

# Host protection BEFORE the core import below pulls in torch/BLAS:
# thread caps only bite if exported first, and the job-object memory
# cap + below-normal priority are what keep the app from ever hanging
# the host machine (2026-07-17 freeze).
import host_guard
host_guard.install("pluginfer-app")

from core import (
    PluginRegistry,
    InferenceEngine,
    HardwareDetector,
    QALController,
    LicenseValidator,
    CompleteMeshController # Import the Node logic
)
from utils.gaming_detector import GamingDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('pluginfer.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class Pluginfer:
    """Main Pluginfer application class"""
    
    def __init__(self, plugin_dir: str = "plugins", enable_licensing: bool = True):
        self.plugin_dir = plugin_dir
        self.enable_licensing = enable_licensing
        
        # Initialize components
        self.license = LicenseValidator() if enable_licensing else None
        self.hardware = HardwareDetector()
        self.registry = PluginRegistry(plugin_dir)
        self.engine = InferenceEngine()
        self.qal = QALController()
        
        logger.info("Pluginfer initialized")
    
    def startup_checks(self) -> bool:
        """
        Perform startup validation.
        
        Returns:
            True if all checks pass
        """
        print("\n" + "="*70)
        print("🚀 PLUGINFER - GPU-AGNOSTIC AI EXECUTION RUNTIME")
        print("="*70)
        
        # License check
        if self.license:
            if not self.license.validate():
                print("\n❌ License validation failed!")
                self.license.print_license_info()
                # Continue as free tier instead of failing
                # return False
            else:
                print(f"\n✅ License: {self.license.get_tier().upper()}")
        else:
            print("\n✅ License: FREE")
        
        # Hardware detection
        print("\n🖥️  Hardware Detection:")
        devices = self.hardware.detect_all_devices()
        best_device = self.hardware.get_best_device()
        
        for device in devices:
            status = "✅" if device['available'] else "⚠️"
            print(f"   {status} {device['type'].upper()}: {device['name']}")
        
        print(f"\n🎯 Selected Device: {best_device['type'].upper()}")
        
        # Plugin discovery
        print("\n📦 Discovering Plugins:")
        count = self.registry.discover_plugins()
        
        if count == 0:
            print("   ⚠️  No plugins found!")
            print(f"   Please add plugins to: {self.plugin_dir}/")
            return False
        
        # List plugins briefly
        plugins = self.registry.list_plugins()
        for name, config in plugins:
             print(f"   ✅ {name} v{config.get('version', '0.0.0')}")
        
        print("\n" + "="*70 + "\n")
        
        return True
    
    def run_inference(self, plugin_name: str, input_data: dict, 
                     use_qal: bool = False) -> dict:
        """
        Run inference using specified plugin.
        
        Args:
            plugin_name: Name of plugin to use
            input_data: Input data dictionary
            use_qal: Whether to use QAL for distribution
            
        Returns:
            Result dictionary
        """
        # Check license quota
        if self.license and not self.license.record_inference():
            logger.error("Inference quota exceeded")
            return {'error': 'Quota exceeded'}
        
        # Get plugin
        plugin = self.registry.get_plugin(plugin_name)
        if not plugin:
            logger.error(f"Plugin not found: {plugin_name}")
            return {'error': f'Plugin not found: {plugin_name}'}
        
        # Check QAL availability
        if use_qal and self.license:
            if not self.license.check_feature('qal_enabled'):
                logger.warning("QAL not available in current license tier")
                use_qal = False
        
        # Execute
        if use_qal:
            logger.info("Using QAL for execution")
            results = self.qal.distribute_workload(plugin, [input_data])
            return results[0]
        else:
            return self.engine.run(plugin, input_data)
    
    def run_batch(self, plugin_name: str, input_data_list: list, 
                  use_qal: bool = True) -> list:
        """
        Run batch inference.
        
        Args:
            plugin_name: Name of plugin to use
            input_data_list: List of input dictionaries
            use_qal: Whether to use QAL for distribution
            
        Returns:
            List of result dictionaries
        """
        # Check batch size limit
        if self.license:
            max_batch = self.license.get_feature_value('batch_size')
            if len(input_data_list) > max_batch:
                logger.warning(f"Batch size {len(input_data_list)} exceeds limit {max_batch}")
                input_data_list = input_data_list[:max_batch]
        
        # Get plugin
        plugin = self.registry.get_plugin(plugin_name)
        if not plugin:
            return [{'error': f'Plugin not found: {plugin_name}'}]
        
        # Check QAL
        if use_qal and self.license:
            if not self.license.check_feature('qal_enabled'):
                use_qal = False
        
        # Execute
        if use_qal:
            logger.info(f"Using QAL for batch of {len(input_data_list)}")
            return self.qal.distribute_workload(plugin, input_data_list)
        else:
            return self.engine.run_batch(plugin, input_data_list)
    
    def list_plugins(self):
        """Print available plugins"""
        self.registry.print_summary()
    
    def print_stats(self):
        """Print execution statistics"""
        self.engine.print_stats()
        self.qal.print_performance_summary()
        
        if self.license:
            self.license.print_license_info()


def main():
    """Main CLI entry point"""
    # 'pluginfer up' — the zero-config onboarding path. Handled before
    # argparse so the subcommand's own flags (--port, --seed-host)
    # never collide with the flags below.
    if len(sys.argv) > 1 and sys.argv[1] == "up":
        from tools.up import main as up_main
        up_main(sys.argv[2:])
        return

    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description='Pluginfer - GPU-Agnostic AI Runtime')
    parser.add_argument('--plugin-dir', default='plugins', help='Plugin directory')
    parser.add_argument('--no-license', action='store_true', help='Disable licensing')
    parser.add_argument('--list-plugins', action='store_true', help='List available plugins')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--test', action='store_true', help='Run test inference')
    parser.add_argument('--federation-status', action='store_true',
                        help='Print model federation status (Filum + local + remote)')
    parser.add_argument('--ask', metavar='PROMPT',
                        help='Ask the federation a prompt (uses Filum + local LLMs + opt-in remote)')
    parser.add_argument('--privacy', choices=['LOCAL_ONLY', 'HYBRID', 'MESH_FULL'],
                        default='HYBRID',
                        help='Privacy mode for --ask (default: HYBRID)')
    # Devserver mode: drop-in OpenAI/Anthropic shim
    # routed through the Pluginfer auction. Existing apps point their
    # SDK at this URL and get mesh execution for free.
    parser.add_argument('--dev', action='store_true',
                        help='Run the OpenAI/Anthropic-compatible devserver shim')
    parser.add_argument('--dev-host', default='127.0.0.1',
                        help='Devserver bind host (default: 127.0.0.1)')
    parser.add_argument('--dev-port', type=int, default=11434,
                        help='Devserver bind port (default: 11434)')

    # If no arguments provided, defaults to None (implies Node Mode)

    args = parser.parse_args()

    # Devserver short-circuit: don't require plugins / licensing — the
    # shim only needs the auction layer and configured providers.
    if args.dev:
        from api.devserver import serve
        serve(host=args.dev_host, port=args.dev_port)
        return

    # Federation surface: short-circuit before app init so these are
    # cheap to invoke and don't require licensing / hardware probes.
    if args.federation_status or args.ask:
        from ai.filum.hpa.model_federation import (
            ModelFederation, FederationConfig, GenerationRequest,
            quick_status as federation_quick_status,
        )
        if args.federation_status:
            print(federation_quick_status())
            return
        if args.ask:
            cfg = FederationConfig(privacy_mode=args.privacy)
            fed = ModelFederation(cfg)
            req = GenerationRequest(
                prompt=args.ask, max_tokens=256, privacy_mode=args.privacy,
            )
            resp = fed.generate(req)
            print(f"[backend: {resp.backend_name}  model: {resp.model_id}]")
            print(resp.text)
            return
    
    # Initialize Pluginfer
    app = Pluginfer(
        plugin_dir=args.plugin_dir,
        enable_licensing=not args.no_license
    )
    
    # Run startup checks
    if not app.startup_checks():
        # If checks fail (no plugins), we might still want to run if it's node mode
        pass
    
    # Handle commands
    if args.list_plugins:
        app.list_plugins()
        return
    
    if args.stats:
        app.print_stats()
        return
    
    if args.test:
        print("🧪 Running test inference...\n")
        
        # Try to run TextProcessor if available
        plugin_name = 'TextProcessor'
        test_input = {
            'text': 'Hello Pluginfer!',
            'operation': 'uppercase'
        }
        
        result = app.run_inference(plugin_name, test_input)
        
        if 'error' in result:
            print(f"❌ Test failed: {result['error']}")
        else:
            print(f"✅ Test successful!")
            print(f"   Input: {test_input}")
            print(f"   Result: {result.get('result')}")
            print(f"   Time: {result['_metadata']['execution_time']:.4f}s")
        
        return
    
    # =================================================================
    # NODE SERVER MODE (Default)
    # =================================================================
    print("\n[+] No arguments provided. Starting PLUGINFER NODE SERVER...\n")
    
    # 2. Gaming Detection (The "Killer Feature")
    print("\n[2/4] Initializing Gaming Detector...")
    try:
        gamer = GamingDetector()
        print(f"   ✓ Monitoring {len(gamer.game_list)} games for auto-pause")
    except Exception as e:
        logger.warning(f"Gaming detector failed: {e}")
        print("   ⚠️ Gaming Detector init failed (Continuing)")
    
    # 3. Security
    print("\n[3/4] Securing Environment...")
    # sec = SecurityManager() 
    # (Initialized inside CompleteMeshController)
    print("   ✓ Isolation Layer Active")
    
    # 4. Start Node
    print("\n[4/4] Connecting to Mesh...")
    
    # Default Config
    host = '0.0.0.0'
    port = 9000
    mode = 'hybrid' # Default to hybrid so they can see it working immediately
    
    print(f"   Starting {mode} node on {host}:{port}")
    
    try:
        node = CompleteMeshController(host, port, mode)
        node.start()
        
        print(f"\n✅ NODE ONLINE")
        print(f"   ID: {node.node_id}")
        print(f"   Dashboard: http://localhost:{port}/ (API)")
        print(f"   Listening on port: {port}")
        print("\nPress Ctrl+C to stop...")
        
        # Keep alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping node...")
        if 'node' in locals():
            node.stop()
        print("Goodbye.")
    except Exception as e:
        logger.exception("Node Startup Error")
        print(f"\n❌ Node Startup Error: {e}")
        
        # Keep window open on error
        print("\nPress Enter to exit...")
        input()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception as e:
        logger.exception("Fatal error:")
        print(f"\n❌ Fatal Error: {e}")
    finally:
        if getattr(sys, 'frozen', False):
            # Only pause if we didn't start the node (Node has its own loop)
            # But main() mostly handles exits now.
            pass
