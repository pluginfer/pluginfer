
import sys
import os
print(f"Python: {sys.version}")
print(f"Path: {sys.path}")
try:
    import socket
    print("✅ Socket imported")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("✅ Socket created")
except Exception as e:
    print(f"❌ Socket failed: {e}")
