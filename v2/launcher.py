#!/usr/bin/env python3
"""
Pluginfer Launcher
Simple wrapper to launch Pluginfer without needing to know the exact command
"""
import sys
import os
import subprocess

def print_banner():
    print("\n" + "="*70)
    print("  🚀 PLUGINFER LAUNCHER")
    print("="*70 + "\n")

def check_python():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def show_menu():
    """Show interactive menu"""
    print("\n📋 What would you like to do?\n")
    print("  1. Run test inference")
    print("  2. List available plugins")
    print("  3. Show statistics")
    print("  4. Run interactive demo")
    print("  5. Run mesh networking example")
    print("  6. Run basic example")
    print("  7. Exit")
    print()
    
    choice = input("Enter choice (1-7): ").strip()
    return choice

def run_command(cmd, description):
    """Run a command"""
    print(f"\n🚀 {description}...\n")
    print(f"Command: {' '.join(cmd)}\n")
    print("="*70 + "\n")
    
    try:
        result = subprocess.run(cmd)
        print("\n" + "="*70)
        if result.returncode == 0:
            print("✅ Completed successfully")
        else:
            print("⚠️  Command exited with status", result.returncode)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def main():
    print_banner()
    
    # Check Python
    if not check_python():
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"📁 Working directory: {script_dir}")
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            run_command([sys.executable, 'pluginfer.py', '--no-license', '--test'], 
                       "Running test inference")
        
        elif choice == '2':
            run_command([sys.executable, 'pluginfer.py', '--no-license', '--list-plugins'],
                       "Listing plugins")
        
        elif choice == '3':
            run_command([sys.executable, 'pluginfer.py', '--no-license', '--stats'],
                       "Showing statistics")
        
        elif choice == '4':
            run_command([sys.executable, 'demo.py'],
                       "Running interactive demo")
        
        elif choice == '5':
            run_command([sys.executable, 'examples/example_mesh_network.py'],
                       "Running mesh networking example")
        
        elif choice == '6':
            run_command([sys.executable, 'examples/example_basic.py'],
                       "Running basic example")
        
        elif choice == '7':
            print("\n👋 Goodbye!\n")
            break
        
        else:
            print("\n❌ Invalid choice. Please try again.")
        
        if choice in ['1', '2', '3', '4', '5', '6']:
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Launcher error: {e}")
        input("\nPress Enter to exit...")
