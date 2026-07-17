#!/usr/bin/env python3
"""
Pluginfer Setup Script
Handles installation, configuration, and initial setup
"""
import sys
import subprocess
import os
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(text)
    print("="*70 + "\n")

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Try to install PyTorch with basic support
        print("   Installing PyTorch...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "numpy", "--break-system-packages"
        ], check=True, capture_output=True)
        
        print("   ✅ PyTorch installed")
        
        # Install other dependencies
        print("   Installing additional packages...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "py-cpuinfo", "cryptography", "pyjwt", "requests",
            "--break-system-packages"
        ], check=True, capture_output=True)
        
        print("   ✅ Dependencies installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️  Installation warning: {e}")
        print("   You may need to install dependencies manually:")
        print("   pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    dirs = ['plugins', 'logs', 'data']
    for dir_name in dirs:
        path = Path(dir_name)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {dir_name}/")
        else:
            print(f"   ℹ️  Exists: {dir_name}/")
    
    return True

def create_default_license():
    """Create default FREE license"""
    print("\n🔐 Creating default license...")
    
    license_file = Path("license.json")
    if license_file.exists():
        print("   ℹ️  License file already exists")
        return True
    
    import json
    from datetime import datetime, timedelta
    
    license_data = {
        'tier': 'free',
        'key': 'FREE-LICENSE',
        'valid_until': None,
        'device_fingerprint': None,
        'generated_at': datetime.now().isoformat()
    }
    
    try:
        with open(license_file, 'w') as f:
            json.dump(license_data, f, indent=2)
        print("   ✅ Created FREE tier license")
        return True
    except Exception as e:
        print(f"   ⚠️  Could not create license: {e}")
        return False

def run_tests():
    """Run test suite"""
    print("\n🧪 Running tests...")
    
    test_file = Path("tests/test_all.py")
    if not test_file.exists():
        print("   ⚠️  Test file not found")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Print test output
        print(result.stdout)
        
        if result.returncode == 0:
            print("   ✅ All tests passed!")
            return True
        else:
            print("   ⚠️  Some tests failed")
            if result.stderr:
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⚠️  Tests timed out")
        return False
    except Exception as e:
        print(f"   ⚠️  Could not run tests: {e}")
        return False

def print_next_steps():
    """Print next steps for user"""
    print_header("🎉 SETUP COMPLETE!")
    
    print("📚 Next Steps:")
    print()
    print("1. Run the main application:")
    print("   python pluginfer.py --test")
    print()
    print("2. List available plugins:")
    print("   python pluginfer.py --list-plugins")
    print()
    print("3. Run examples:")
    print("   python examples/example_basic.py")
    print("   python examples/example_qal_batch.py")
    print()
    print("4. Create your own plugin:")
    print("   - Copy plugins/text_processor.py as a template")
    print("   - Implement your run() and config() methods")
    print("   - Save in plugins/ directory")
    print()
    print("5. Upgrade to Pro/Enterprise:")
    print("   - Visit: https://pluginfer.ai/pricing")
    print("   - Get GPU support and unlimited inferences")
    print()
    print("📖 Documentation: README.md")
    print("🐛 Issues: GitHub Issues")
    print("💬 Support: support@pluginfer.ai")
    print()

def main():
    print_header("⚙️  PLUGINFER SETUP")
    
    print("This script will:")
    print("  • Check Python version")
    print("  • Install dependencies")
    print("  • Create necessary directories")
    print("  • Set up default license")
    print("  • Run tests")
    print()
    
    input("Press Enter to continue...")
    
    # Run setup steps
    steps = [
        ("Python Version", check_python_version),
        ("Dependencies", install_dependencies),
        ("Directories", create_directories),
        ("License", create_default_license),
    ]
    
    all_passed = True
    for step_name, step_func in steps:
        if not step_func():
            all_passed = False
            print(f"⚠️  {step_name} step had issues (but continuing...)")
    
    # Run tests (optional)
    print()
    run_tests_choice = input("Run test suite? (y/N): ").strip().lower()
    if run_tests_choice == 'y':
        run_tests()
    
    # Print completion
    print_next_steps()
    
    if all_passed:
        print("✨ Setup completed successfully!")
    else:
        print("⚠️  Setup completed with some warnings")
    
    print()

if __name__ == "__main__":
    main()
