"""
Mesh Network Discovery
Enables auto-discovery of mesh nodes on the local network using UDP broadcasts.
"""
import socket
import json
import threading
import time
import logging
import uuid
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

# Standard constants
DISCOVERY_PORT = 9998
DISCOVERY_INTERVAL = 5.0  # Seconds between broadcasts
BROADCAST_ADDR = '<broadcast>'

class MeshDiscovery:
    """
    Handles UDP broadcasting to discover other mesh nodes on the local network.
    
    Protocol:
    - UDP Broadcast on port 9998
    - JSON Payload
    """
    
    def __init__(self, node_id: str, port: int, capabilities: Dict[str, Any], swarm_id: str = "public"):
        self.node_id = node_id
        self.tcp_port = port  # The port where our main TCP server is listening
        self.capabilities = capabilities
        self.swarm_id = swarm_id  # UNIQUE SWARM IDENTIFIER (For Private Meshes)
        
        self.running = False
        self.socket = None
        self.broadcast_thread = None
        self.listener_thread = None
        self.found_coordinator = None
        self.role = 'worker'
        
        # Callbacks
        self.on_peer_found = None  # Function(peer_info)
        
    def start(self, role: str = 'worker'):
        """
        Start the discovery service.
        role: 'coordinator', 'worker', or 'hybrid'
        """
        self.role = role
        self.running = True
        
        # Setup UDP socket
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Windows specific: Allow multiple sockets to bind to same port
            if hasattr(socket, 'SO_REUSEPORT'):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            
            # Windows/Linux difference in binding
            try:
                # Bind to all interfaces for receiving
                self.socket.bind(('', DISCOVERY_PORT))
            except Exception:
                try:
                    self.socket.bind(('0.0.0.0', DISCOVERY_PORT))
                except:
                    logger.warning("Could not bind discovery port. Auto-discovery disabled.")
                    return
                
            logger.info(f"Discovery service started on UDP port {DISCOVERY_PORT} (Swarm: {self.swarm_id})")
            
            # Determine Broadcast Address
            try:
                # Find local IP to calculate subnet broadcast
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    # 1. Try Internet
                    s.connect(("8.8.8.8", 80))
                    self.local_ip = s.getsockname()[0]
                except:
                    # 2. Try Local Broadcast (Finding primary LAN interface)
                    try:
                        s.connect(("10.255.255.255", 1))
                        self.local_ip = s.getsockname()[0]
                    except:
                         self.local_ip = socket.gethostbyname(socket.gethostname())
                s.close()
                
                # Simple Class C broadcast assumption (safe fallback)
                # 192.168.1.x -> 192.168.1.255
                parts = self.local_ip.split('.')
                self.broadcast_ip = f"{parts[0]}.{parts[1]}.{parts[2]}.255"
                logger.info(f"Discovery: Broadcasting to {self.broadcast_ip} (Local IP: {self.local_ip})")
            except Exception as e:
                logger.warning(f"Discovery: Failed to calculate broadcast addr: {e}")
                self.broadcast_ip = '<broadcast>'
                self.local_ip = '127.0.0.1'
            
            # Start listener
            self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.listener_thread.start()
            
            # Start broadcaster (only if we want to be found automatically)
            self.broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True)
            self.broadcast_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to start discovery: {e}")
            self.running = False

    def stop(self):
        """Stop the discovery service"""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass

    def search_for_coordinator(self, timeout: int = 5) -> Optional[Dict[str, Any]]:
        """
        Actively search for a coordinator on the network.
        Returns coordinator info if found, else None.
        """
        msg = {
            'type': 'SEARCH',
            'role': 'worker_looking_for_coordinator',
            'node_id': self.node_id,
            'tcp_port': self.tcp_port,
            'swarm_id': self.swarm_id
        }
        
        # Let's send the broadcast
        self._send_broadcast(msg)
        
        start_time = time.time()
        self.found_coordinator = None
        
        # Wait for listener to find something
        while time.time() - start_time < timeout:
            if self.found_coordinator:
                return self.found_coordinator
            time.sleep(0.1)
            # Send search pulses every second
            if int((time.time() - start_time) * 10) % 10 == 0:
                self._send_broadcast(msg)
                
        return None

    def _broadcast_loop(self):
        """Periodically broadcast presence"""
        while self.running:
            if self.role in ['coordinator', 'hybrid']:
                # I am a coordinator, I announce myself
                msg = {
                    'type': 'ANNOUNCE',
                    'role': self.role,
                    'node_id': self.node_id,
                    'tcp_port': self.tcp_port,
                    'host_name': socket.gethostname(),
                    'capabilities': self.capabilities,
                    'swarm_id': self.swarm_id
                }
                self._send_broadcast(msg)
            
            time.sleep(DISCOVERY_INTERVAL)

    def _send_broadcast(self, msg: Dict[str, Any]):
        """Send a UDP broadcast message"""
        try:
            if not self.socket: return
            msg['swarm_id'] = self.swarm_id # Ensure Swarm ID is attached
            data = json.dumps(msg).encode('utf-8')
            
            # Send to generic broadcast AND specific subnet broadcast
            # This covers both bases (some OS/routers prefer one over the other)
            try:
                self.socket.sendto(data, ('<broadcast>', DISCOVERY_PORT))
            except:
                pass
                
            if hasattr(self, 'broadcast_ip') and self.broadcast_ip != '<broadcast>':
                self.socket.sendto(data, (self.broadcast_ip, DISCOVERY_PORT))
                
        except Exception as e:
            # Broadcast might fail if network is down or other issues, just ignore
            pass

    def _listen_loop(self):
        """Listen for incoming UDP broadcasts"""
        while self.running:
            try:
                if not self.socket: 
                    time.sleep(0.1)
                    continue
                    
                data, addr = self.socket.recvfrom(4096)
                message = json.loads(data.decode('utf-8'))
                
                # Ignore our own messages
                if message.get('node_id') == self.node_id:
                    continue
                    
                # [SECURITY] SWARM CHECK
                # If the Swarm ID doesn't match ours, we ignore this node entirely.
                # This creates the "Private Network" effect.
                remote_swarm = message.get('swarm_id', 'public')
                if remote_swarm != self.swarm_id:
                    # Ignore alien swarm
                    if remote_swarm == 'public': 
                         pass # Quietly ignore public noise if private
                    else:
                         pass # Ignore different private swarm
                    continue
                
                msg_type = message.get('type')
                remote_role = message.get('role')
                
                # Handle different message types
                if msg_type == 'SEARCH' and self.role in ['coordinator', 'hybrid']:
                    # Someone is looking for a coordinator, and that's me!
                    response = {
                        'type': 'RESPONSE',
                        'role': self.role,
                        'node_id': self.node_id,
                        'tcp_port': self.tcp_port,
                        'host_name': socket.gethostname(),
                        'swarm_id': self.swarm_id
                    }
                    # Send back to the sender
                    self.socket.sendto(json.dumps(response).encode('utf-8'), addr)
                    
                    logger.info(f"Answered discovery search from {addr[0]} (Swarm: {self.swarm_id})")
                    
                elif msg_type in ['ANNOUNCE', 'RESPONSE']:
                    # Use the sender's IP address from the socket, not the message
                    sender_ip = addr[0]
                    
                    info = {
                        'ip': sender_ip,
                        'port': message.get('tcp_port'),
                        'node_id': message.get('node_id'),
                        'role': remote_role,
                        'hostname': message.get('host_name')
                    }
                    
                    # If we are looking for a coordinator
                    if remote_role in ['coordinator', 'hybrid'] and not self.role in ['coordinator', 'hybrid']:
                         self.found_coordinator = info
                    
                    # Call callback if set
                    if self.on_peer_found:
                        self.on_peer_found(info)
                        
            except socket.error:
                time.sleep(1)
            except Exception as e:
                logger.error(f"Discovery listener error: {e}")
                time.sleep(1)
