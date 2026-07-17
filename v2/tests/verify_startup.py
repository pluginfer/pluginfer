import winreg
import sys
import os

print("[-] Verifying Startup Registry Access...", flush=True)
try:
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE)
    
    # simulate writing the key
    dummy_cmd = '"C:\\Test\\Path\\PluginferNode.exe"'
    winreg.SetValueEx(key, "PluginferTest", 0, winreg.REG_SZ, dummy_cmd)
    print("   [+] Write Permission: OK")
    
    # read it back
    val, _ = winreg.QueryValueEx(key, "PluginferTest")
    if val == dummy_cmd:
        print("   [+] Read Verification: OK")
    else:
        print(f"   [!] Read Mismatch: {val}")
        
    # clean up
    winreg.DeleteValue(key, "PluginferTest")
    print("   [+] Cleanup: OK")
    winreg.CloseKey(key)
    print("   [OK] Registry Startup Mechanism Verified.")
    
except Exception as e:
    print(f"   [!] Registry Error: {e}")

print("\n[-] Verifying System Tray Dependencies...")
try:
    import pystray
    import PIL
    print(f"   [+] pystray: INSTALLED", flush=True)
    print(f"   [+] Pillow: {PIL.__version__}")
    print("   [OK] Tray Dependencies ready.")
except ImportError as e:
    print(f"   [!] Missing Dependency: {e}")
