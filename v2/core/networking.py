import re
import socket
import logging
import requests
import threading
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class NetworkManager:
    def __init__(self, port=9000):
        self.port = port
        self.public_ip = None
        self.local_ip = self._get_local_ip()
        self.upnp_success = False

    def _get_local_ip(self):
        """
        Get the IP address of the primary network interface.

        Previously did `connect(("8.8.8.8", 80))`, leaking every node's
        startup to Google DNS — a privacy issue. Now uses an RFC 5737
        documentation-range address that never gets routed to anyone;
        the OS still picks the local interface that *would* route there.
        Falls back to `gethostbyname(gethostname())`, then `127.0.0.1`.
        """
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(2)
            try:
                s.connect(("198.51.100.1", 1))     # RFC 5737, never routable
                ip = s.getsockname()[0]
            finally:
                s.close()
            return ip
        except OSError:
            try:
                return socket.gethostbyname(socket.gethostname())
            except OSError:
                return "127.0.0.1"

    def get_public_ip(self):
        """Fetch public IP from external service with retries"""
        try:
            services = [
                'https://api.ipify.org',
                'https://ifconfig.me/ip',
                'https://icanhazip.com'
            ]
            for svc in services:
                try:
                    response = requests.get(svc, timeout=3)
                    if response.status_code == 200:
                        self.public_ip = response.text.strip()
                        logger.info(f"[Network] Public IP Discovered: {self.public_ip}")
                        return self.public_ip
                except:
                    continue
        except Exception as e:
            logger.error(f"Failed to get public IP: {e}")
        return None

    def enable_upnp(self):
        """
        Attempt to map port using UPnP (IGD Protocol).
        Robust implementation with XML parsing and retries.
        """
        try:
            logger.info("Attempting UPnP Port Mapping...")
            
            # 1. SSDP Discovery
            ssdp_response = self._ssdp_discover()
            if not ssdp_response:
                logger.warning("UPnP Discovery Timed Out (No Router Found)")
                return False

            # 2. Extract Location URL
            location_url = self._parse_ssdp_location(ssdp_response)
            if not location_url:
                logger.warning("UPnP: Could not find location in SSDP response")
                return False
                
            logger.info(f"UPnP: Router found at {location_url}")

            # 3. Get Device Description XML
            control_url = self._get_control_url(location_url)
            if not control_url:
                logger.warning("UPnP: Could not find WANIPConnection control URL")
                return False

            # 4. Add Port Mapping via SOAP
            return self._add_port_mapping(control_url)

        except Exception as e:
            logger.error(f"UPnP functionality failed: {e}")
            return False

    def _ssdp_discover(self, retries=3, timeout=2):
        """Send SSDP broadcast to find gateway"""
        SSDP_ADDR = "239.255.255.250"
        SSDP_PORT = 1900
        SSDP_MX = 2
        SSDP_ST = "urn:schemas-upnp-org:device:InternetGatewayDevice:1"

        ssdpRequest = "M-SEARCH * HTTP/1.1\r\n" + \
                    "HOST: {}:{}\r\n".format(SSDP_ADDR, SSDP_PORT) + \
                    "MAN: \"ssdp:discover\"\r\n" + \
                    "MX: {}\r\n".format(SSDP_MX) + \
                    "ST: {}\r\n".format(SSDP_ST) + "\r\n"

        for i in range(retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(timeout)
                
                # Bind to the specific local IP to ensure we use the right interface
                try:
                    sock.bind((self.local_ip, 0))
                except:
                    sock.bind(('', 0)) # Fallback
                
                sock.sendto(ssdpRequest.encode(), (SSDP_ADDR, SSDP_PORT))
                
                while True:
                    try:
                        data, addr = sock.recvfrom(4096)
                        return data.decode('utf-8', errors='ignore')
                    except socket.timeout:
                        break
            except Exception as e:
                logger.debug(f"SSDP Attempt {i+1} failed: {e}")
            finally:
                sock.close()
                
        return None

    def _parse_ssdp_location(self, data):
        """Extract LOCATION header from SSDP response"""
        import re
        # Case insensitive search
        lines = data.split('\r\n')
        for line in lines:
            if line.upper().startswith('LOCATION:'):
                return line.split(':', 1)[1].strip()
        return None

    def _get_control_url(self, location_url):
        """Parse XML to find the Control URL for WANIPConnection"""
        try:
            resp = requests.get(location_url, timeout=3)
            if resp.status_code != 200:
                return None
                
            # Parse XML
            xml_data = resp.text
            
            # We want the service: WANIPConnection or WANPPPConnection
            # Simple approach: Search for serviceType containing WANIPConnection, then get controlURL sibling
            
            # Using basic string finding or Regex is sometimes safer than ET for malformed XML from cheap routers,
            # but let's try ET first for correctness.
            
            # Removing namespaces for easier parsing
            xml_data =  re.sub(r' xmlns="[^"]+"', '', xml_data, count=1) 
            
            # Simpler Regex approach which is often more robust against namespace hell
            import re
            
            # Find the service block for WANIPConnection:1
            # Patterns to try
            service_types = [
                "urn:schemas-upnp-org:service:WANIPConnection:1",
                "urn:schemas-upnp-org:service:WANPPPConnection:1"
            ]
            
            for st in service_types:
                # Regex to find the <service> block containing this serviceType
                # This is a bit complex, let's just find the serviceType and look ahead for controlURL
                pattern = re.compile(f"<serviceType>{st}</serviceType>.*?<controlURL>(.*?)</controlURL>", re.DOTALL)
                match = pattern.search(xml_data)
                
                if match:
                    control_url = match.group(1).strip()
                    parsed_loc = urlparse(location_url)
                    
                    if not control_url.startswith('http'):
                        if control_url.startswith('/'):
                            return f"{parsed_loc.scheme}://{parsed_loc.netloc}{control_url}"
                        else:
                            # Relative to base? Technically yes, but usually relative to root
                            return f"{parsed_loc.scheme}://{parsed_loc.netloc}/{control_url}"
                    return control_url
                    
            return None
            
        except Exception as e:
            logger.error(f"Error parsing UPnP XML: {e}")
            return None

    def _add_port_mapping(self, control_url):
        """Send SOAP request to map port"""
        soap_body = f"""<?xml version="1.0"?>
        <s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
        <s:Body>
        <u:AddPortMapping xmlns:u="urn:schemas-upnp-org:service:WANIPConnection:1">
        <NewRemoteHost></NewRemoteHost>
        <NewExternalPort>{self.port}</NewExternalPort>
        <NewProtocol>TCP</NewProtocol>
        <NewInternalPort>{self.port}</NewInternalPort>
        <NewInternalClient>{self.local_ip}</NewInternalClient>
        <NewEnabled>1</NewEnabled>
        <NewPortMappingDescription>PluginferNode</NewPortMappingDescription>
        <NewLeaseDuration>0</NewLeaseDuration>
        </u:AddPortMapping>
        </s:Body>
        </s:Envelope>
        """
        
        headers = {
            'Content-Type': 'text/xml',
            'SOAPAction': '"urn:schemas-upnp-org:service:WANIPConnection:1#AddPortMapping"'
        }
        
        try:
            resp = requests.post(control_url, data=soap_body, headers=headers, timeout=3)
            if resp.status_code == 200:
                logger.info(f"[UPnP] Success! Port {self.port} mapped to {self.local_ip}:{self.port}")
                self.upnp_success = True
                return True
            else:
                logger.error(f"UPnP Failed: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            logger.error(f"UPnP SOAP Error: {e}")
            return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    nm = NetworkManager(9000)
    nm.enable_upnp()
