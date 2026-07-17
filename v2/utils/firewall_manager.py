import subprocess
import logging
import ctypes
import sys
import os

logger = logging.getLogger(__name__)

class FirewallManager:
    """
    Manages Windows Defender Firewall rules for the application.
    Automatically creates 'Allow' rules for incoming mesh connections.
    """
    
    RULE_NAME = "Pluginfer Mesh Node"
    
    @staticmethod
    def is_admin():
        """Check if the process has administrator privileges"""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except Exception:
            return False

    @staticmethod
    def authorize_app(port_tcp=9000, port_udp=9998):
        """
        Create firewall rules to allow incoming traffic.
        Returns True if successful.
        """
        if not FirewallManager.is_admin():
            logger.warning("Firewall: Not running as Admin. Cannot modify rules.")
            return False
            
        try:
            # Check if rules exist to avoid redundant calls
            if FirewallManager._rule_exists(FirewallManager.RULE_NAME):
                logger.info("Firewall: Rules already exist.")
                return True
                
            logger.info("Firewall: Creating 'Allow' rules for Pluginfer...")
            
            # Get path to current executable
            if getattr(sys, 'frozen', False):
                app_path = sys.executable
            else:
                app_path = sys.executable  # For python script, it authorizes python.exe (broad but works for dev)
                
            # Create Program Rule (Allows ALL ports for this app)
            # This is critical for UPnP (UDP 1900) and ephemeral return ports
            cmd_program = [
                'netsh', 'advfirewall', 'firewall', 'add', 'rule',
                f'name="{FirewallManager.RULE_NAME} (App)"',
                'dir=in',
                'action=allow',
                'program="{}"'.format(app_path),
                'enable=yes',
                'profile=any'
            ]
            
            # Additional outbound rule just in case (usually allowed by default but good to be sure)
            cmd_out = [
                'netsh', 'advfirewall', 'firewall', 'add', 'rule',
                f'name="{FirewallManager.RULE_NAME} (App Out)"',
                'dir=out',
                'action=allow',
                'program="{}"'.format(app_path),
                'enable=yes',
                'profile=any'
            ]

            # Cleanup old legacy port rules if they exist to avoid clutter
            subprocess.run(f'netsh advfirewall firewall delete rule name="{FirewallManager.RULE_NAME} (TCP)"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(f'netsh advfirewall firewall delete rule name="{FirewallManager.RULE_NAME} (UDP)"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Execute new rules
            subprocess.run(" ".join(cmd_program), shell=True, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(" ".join(cmd_out), shell=True, check=True, stdout=subprocess.DEVNULL)
            
            logger.info(f"Firewall: Successfully authorized application executable: {app_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Firewall: Failed to add rules. Error: {e}")
            return False
        except Exception as e:
            logger.error(f"Firewall: Unexpected error: {e}")
            return False

    @staticmethod
    def _rule_exists(rule_name):
        """Check if a firewall rule exists via netsh"""
        try:
            cmd = f'netsh advfirewall firewall show rule name="{rule_name} (App)"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return "No rules match" not in result.stdout and result.returncode == 0
        except Exception:
            return False

if __name__ == "__main__":
    # Test execution
    if FirewallManager.is_admin():
        print("Running as Admin")
        FirewallManager.authorize_app()
    else:
        print("Not running as Admin")
