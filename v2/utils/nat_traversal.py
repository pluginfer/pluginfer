"""
Zero-Config NAT Traversal (UPnP)
Attempts to forward ports automatically on home routers so this machine can act as a Server/Coordinator.
"""
import socket
import re
from urllib.parse import urlparse
import urllib.request
import logging

logger = logging.getLogger(__name__)

def get_local_ip():
    """Get the local network IP of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Doesn't actually connect, just determines route
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def get_public_ip():
    """
    Get the external (Public) IP address of this network.
    Useful for telling others where to connect.
    """
    services = [
        'https://api.ipify.org',
        'https://ifconfig.me/ip',
        'https://icanhazip.com'
    ]
    
    for service in services:
        try:
            with urllib.request.urlopen(service, timeout=3) as response:
                ip = response.read().decode('utf-8').strip()
                # Basic validation for IPv4
                if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', ip):
                    return ip
        except Exception:
            continue
            
    return None

def add_port_mapping(internal_ip, external_port, internal_port, protocol='TCP', description='Pluginfer Node'):
    """
    Robust UPnP implementation to map a port on the router to this machine.
    
    Steps:
    1. Discover Router (SSDP)
    2. Get Control URL from XML
    3. Send SOAP Request to AddPortMapping
    """
    try:
        # 1. Discover Gateway (SSDP)
        SSDP_ADDR = "239.255.255.250"
        SSDP_PORT = 1900
        SSDP_MX = 2
        SSDP_ST = "urn:schemas-upnp-org:device:InternetGatewayDevice:1"

        ssdpRequest = "M-SEARCH * HTTP/1.1\r\n" + \
                    "HOST: %s:%d\r\n" % (SSDP_ADDR, SSDP_PORT) + \
                    "MAN: \"ssdp:discover\"\r\n" + \
                    "MX: %d\r\n" % (SSDP_MX) + \
                    "ST: %s\r\n" + \
                    "\r\n"

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(3)
        sock.sendto(ssdpRequest.encode(), (SSDP_ADDR, SSDP_PORT))
        
        try:
            resp, address = sock.recvfrom(1024)
            resp = resp.decode()
        except socket.timeout:
            return False
            
        # 2. Extract Location URL
        location_match = re.search(r'LOCATION: (.*)', resp, re.IGNORECASE)
        if not location_match:
            return False
        
        location_url = location_match.group(1).strip()
        
        # 3. Fetch Service Description (XML) to find Control URL
        try:
            with urllib.request.urlopen(location_url, timeout=3) as response:
                xml_content = response.read().decode('utf-8')
        except Exception as e:
            return False
            
        # Find the Control URL for WANIPConnection or WANPPPConnection
        service_types = [
            "urn:schemas-upnp-org:service:WANIPConnection:1",
            "urn:schemas-upnp-org:service:WANPPPConnection:1"
        ]
        
        control_url = None
        service_type_used = None
        
        for st in service_types:
            if st in xml_content:
                parts = xml_content.split(st)
                if len(parts) > 1:
                    after = parts[1]
                    control_match = re.search(r'<controlURL>(.*?)</controlURL>', after, re.IGNORECASE)
                    if control_match:
                        control_url = control_match.group(1).strip()
                        service_type_used = st
                        break
        
        if not control_url:
            control_match = re.search(r'<controlURL>(.*?)</controlURL>', xml_content, re.IGNORECASE)
            if control_match:
                control_url = control_match.group(1).strip()
                service_type_used = "urn:schemas-upnp-org:service:WANIPConnection:1"
            else:
                return False
                
        # Fix relative URL
        if not control_url.startswith('http'):
            parsed = urlparse(location_url)
            if control_url.startswith('/'):
               control_url = f"{parsed.scheme}://{parsed.netloc}{control_url}"
            else:
               base_path = '/'.join(parsed.path.split('/')[:-1])
               control_url = f"{parsed.scheme}://{parsed.netloc}{base_path}/{control_url}"

        # 4. Perform SOAP Request: AddPortMapping
        soap_body = f"""<?xml version="1.0"?>
        <s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
        <s:Body>
            <u:AddPortMapping xmlns:u="{service_type_used}">
            <NewRemoteHost></NewRemoteHost>
            <NewExternalPort>{external_port}</NewExternalPort>
            <NewProtocol>{protocol}</NewProtocol>
            <NewInternalPort>{internal_port}</NewInternalPort>
            <NewInternalClient>{internal_ip}</NewInternalClient>
            <NewEnabled>1</NewEnabled>
            <NewPortMappingDescription>{description}</NewPortMappingDescription>
            <NewLeaseDuration>0</NewLeaseDuration>
            </u:AddPortMapping>
        </s:Body>
        </s:Envelope>"""
        
        headers = {
            'SOAPAction': f'"{service_type_used}#AddPortMapping"',
            'Content-Type': 'text/xml',
            'Connection': 'Close',
            'Content-Length': len(soap_body)
        }
        
        req = urllib.request.Request(control_url, data=soap_body.encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status == 200:
                logger.info(f"UPnP: Port {external_port} mapped successfully.")
                return True
                
        return False
        
    except Exception as e:
        logger.error(f"UPnP Failed: {e}")
        return False

def setup_internet_access(port: int, description: str = 'Pluginfer Node') -> dict:
    """
    Attempt to setup internet access via UPnP.
    Returns dict with status and details.
    """
    logger.info(f"Attempting to setup internet access on port {port}...")
    
    # 1. Get Local IP
    local_ip = get_local_ip()
    
    # 2. Try UPnP Port Mapping
    # We attempt it even if public IP check fails, as it might fix routing.
    success = add_port_mapping(local_ip, port, port, description=description)
    
    # 3. Get Public IP
    public_ip = get_public_ip()
    
    result = {
        'success': success,
        'local_ip': local_ip,
        'public_ip': public_ip,
        'port': port,
        'connection_string': f"{public_ip}:{port}" if public_ip else None
    }
    
    if success and public_ip:
        logger.info(f"Internet access configured! Connect via: {result['connection_string']}")
    elif public_ip:
        logger.warning(f"UPnP failed, but public IP found: {public_ip}. You may need to forward port {port} manually.")
    else:
        logger.error("Could not detect public IP or configure UPnP.")
        
    return result
