"""
Network Configuration Helper
Automatically configures network settings for mesh networking
"""
import socket
import subprocess
import platform
import sys
import requests
from typing import Optional, Dict, List

class NetworkConfigurator:
    """
    Helps configure network settings for mesh networking.
    Detects public IP, checks ports, and provides guidance.
    """
    
    def __init__(self):
        self.system = platform.system()
        self.public_ip = None
        self.local_ip = None
        
    def get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            # Create a socket to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            self.local_ip = local_ip
            return local_ip
        except:
            return "127.0.0.1"
    
    def get_public_ip(self) -> Optional[str]:
        """Get public IP address"""
        try:
            response = requests.get('https://api.ipify.org', timeout=5)
            self.public_ip = response.text
            return self.public_ip
        except:
            try:
                response = requests.get('https://ifconfig.me', timeout=5)
                self.public_ip = response.text
                return self.public_ip
            except:
                return None
    
    def check_port_open(self, port: int) -> bool:
        """Check if port is available locally"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            return result != 0  # Port is free if connection failed
        except:
            return False
    
    def test_external_access(self, port: int) -> bool:
        """Test if port is accessible from internet"""
        if not self.public_ip:
            return False
        
        # This would need an external service to test
        # For now, return False (assume not accessible)
        return False
    
    def configure_firewall_linux(self, port: int) -> bool:
        """Configure Linux firewall (ufw)"""
        try:
            # Check if ufw is installed
            result = subprocess.run(['which', 'ufw'], capture_output=True)
            if result.returncode != 0:
                print("⚠️  ufw not installed")
                return False
            
            # Allow port
            print(f"Configuring firewall for port {port}...")
            subprocess.run(['sudo', 'ufw', 'allow', f'{port}/tcp'], check=True)
            print(f"✅ Port {port} allowed in firewall")
            return True
            
        except subprocess.CalledProcessError:
            print(f"❌ Failed to configure firewall")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def configure_firewall_windows(self, port: int) -> bool:
        """Configure Windows firewall"""
        try:
            rule_name = f"Pluginfer_Port_{port}"
            cmd = [
                'netsh', 'advfirewall', 'firewall', 'add', 'rule',
                f'name={rule_name}',
                'dir=in',
                'action=allow',
                'protocol=TCP',
                f'localport={port}'
            ]
            
            print(f"Configuring Windows firewall for port {port}...")
            subprocess.run(cmd, check=True, shell=True)
            print(f"✅ Port {port} allowed in firewall")
            return True
            
        except subprocess.CalledProcessError:
            print(f"❌ Failed to configure firewall (run as Administrator)")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def configure_firewall(self, port: int) -> bool:
        """Configure firewall based on OS"""
        if self.system == 'Linux':
            return self.configure_firewall_linux(port)
        elif self.system == 'Windows':
            return self.configure_firewall_windows(port)
        elif self.system == 'Darwin':  # macOS
            print("ℹ️  macOS: Use System Preferences → Security → Firewall")
            print(f"   Allow incoming connections on port {port}")
            return False
        else:
            print(f"⚠️  Unsupported OS: {self.system}")
            return False
    
    def print_network_info(self):
        """Print network configuration information"""
        print("\n" + "="*70)
        print("🌐 NETWORK CONFIGURATION")
        print("="*70)
        
        # Get IPs
        local_ip = self.get_local_ip()
        public_ip = self.get_public_ip()
        
        print(f"\n📍 IP Addresses:")
        print(f"   Local IP:  {local_ip}")
        print(f"   Public IP: {public_ip or 'Unable to detect'}")
        
        print(f"\n💻 System: {self.system}")
        
        print("\n" + "="*70 + "\n")
    
    def print_setup_guide(self, port: int = 9999):
        """Print complete setup guide"""
        self.print_network_info()
        
        print("📋 SETUP GUIDE FOR INTERNET ACCESS\n")
        
        print("Option 1: VPN (RECOMMENDED - Easiest & Most Secure)")
        print("-" * 70)
        print("1. Install Tailscale: https://tailscale.com/download")
        print("2. Sign up and run: tailscale up")
        print("3. Get your Tailscale IP: tailscale ip -4")
        print("4. Use Tailscale IP to connect")
        print("   ✅ Secure, encrypted, no port forwarding needed\n")
        
        print("Option 2: Port Forwarding (Advanced)")
        print("-" * 70)
        print(f"1. Configure firewall:")
        if self.system == 'Linux':
            print(f"   sudo ufw allow {port}/tcp")
        elif self.system == 'Windows':
            print(f"   Run as Administrator:")
            print(f"   netsh advfirewall firewall add rule name=Pluginfer dir=in action=allow protocol=TCP localport={port}")
        
        print(f"\n2. Setup port forwarding on your router:")
        print(f"   - Login to router (usually 192.168.1.1)")
        print(f"   - Find 'Port Forwarding' section")
        print(f"   - Add rule:")
        print(f"     External Port: {port}")
        print(f"     Internal Port: {port}")
        print(f"     Internal IP: {self.local_ip}")
        print(f"     Protocol: TCP")
        
        print(f"\n3. Share with workers:")
        print(f"   Coordinator address: {self.public_ip or 'YOUR_PUBLIC_IP'}:{port}\n")
        
        print("Option 3: Cloud Server (Professional)")
        print("-" * 70)
        print("1. Rent a VPS (AWS, DigitalOcean, Linode)")
        print("2. Get public IP from provider")
        print("3. Configure firewall on VPS")
        print("4. Run coordinator on VPS")
        print("   ✅ Always online, reliable, professional\n")
        
        print("Option 4: Local Network Only (Testing)")
        print("-" * 70)
        print("1. Ensure all machines on same WiFi")
        print(f"2. Use local IP: {self.local_ip}:{port}")
        print("3. No internet configuration needed")
        print("   ✅ Simple, fast, secure for local use\n")
        
        print("="*70 + "\n")
    
    def interactive_setup(self):
        """Interactive network setup wizard"""
        print("\n" + "="*70)
        print("🧙 MESH NETWORK SETUP WIZARD")
        print("="*70 + "\n")
        
        # Get network info
        self.print_network_info()
        
        # Ask setup type
        print("How do you want to setup mesh networking?\n")
        print("1. Local Network (same WiFi)")
        print("2. Internet (VPN - Tailscale)")
        print("3. Internet (Port Forwarding)")
        print("4. Cloud Server")
        print("5. Just show me the info")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            print("\n✅ LOCAL NETWORK SETUP")
            print("-" * 70)
            print(f"Your local IP: {self.local_ip}")
            print(f"\nCoordinator command:")
            print(f"  python pluginfer.py --mesh-coordinator --port 9999")
            print(f"\nWorker command (on other machines):")
            print(f"  python pluginfer.py --mesh-worker --connect {self.local_ip}:9999")
            
        elif choice == '2':
            print("\n✅ VPN (TAILSCALE) SETUP")
            print("-" * 70)
            print("1. Install Tailscale:")
            print("   Visit: https://tailscale.com/download")
            print("\n2. On coordinator machine:")
            print("   tailscale up")
            print("   tailscale ip -4")
            print("   # Note the IP (e.g., 100.64.0.1)")
            print("\n3. On worker machines:")
            print("   tailscale up")
            print("   python pluginfer.py --mesh-worker --connect 100.64.0.1:9999")
            
        elif choice == '3':
            print("\n✅ PORT FORWARDING SETUP")
            print("-" * 70)
            port = 9999
            
            # Try to configure firewall
            print("\nAttempting to configure firewall...")
            if self.configure_firewall(port):
                print("✅ Firewall configured")
            else:
                print("⚠️  Please configure firewall manually")
            
            print(f"\nRouter Configuration:")
            print(f"  1. Login to router: http://192.168.1.1")
            print(f"  2. Find 'Port Forwarding' or 'NAT'")
            print(f"  3. Add rule:")
            print(f"     External: {port} → Internal: {port}")
            print(f"     Internal IP: {self.local_ip}")
            print(f"\n  4. Save and reboot router")
            print(f"\nShare this with workers:")
            print(f"  {self.public_ip or 'YOUR_PUBLIC_IP'}:{port}")
            
        elif choice == '4':
            print("\n✅ CLOUD SERVER SETUP")
            print("-" * 70)
            print("Recommended VPS providers:")
            print("  • DigitalOcean: $4-10/month")
            print("  • AWS Lightsail: $3.50-5/month")
            print("  • Linode: $5-10/month")
            print("  • Vultr: $2.50-6/month")
            print("\nSetup steps:")
            print("  1. Create VPS with Ubuntu")
            print("  2. SSH into server")
            print("  3. Install Pluginfer")
            print("  4. Run coordinator")
            print("  5. Configure firewall on VPS")
            print("  6. Share public IP with workers")
            
        else:
            self.print_setup_guide()
        
        print("\n" + "="*70)


def auto_configure(port: int = 9999):
    """Automatically configure network for mesh networking"""
    configurator = NetworkConfigurator()
    
    print("\n🔧 AUTO-CONFIGURATION")
    print("="*70)
    
    # Check if port is available
    if not configurator.check_port_open(port):
        print(f"❌ Port {port} is already in use")
        print(f"   Try a different port or close the application using it")
        return False
    else:
        print(f"✅ Port {port} is available")
    
    # Try to configure firewall
    print(f"\n🔥 Configuring firewall...")
    if configurator.configure_firewall(port):
        print(f"✅ Firewall configured successfully")
    else:
        print(f"⚠️  Manual firewall configuration may be needed")
    
    # Print summary
    configurator.print_network_info()
    
    print("✅ Auto-configuration complete!")
    print(f"\n💡 Next steps:")
    print(f"   1. Start coordinator: python pluginfer.py --mesh-coordinator")
    print(f"   2. For internet access, see setup guide")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        # Auto-configure
        auto_configure()
    else:
        # Interactive wizard
        configurator = NetworkConfigurator()
        configurator.interactive_setup()
