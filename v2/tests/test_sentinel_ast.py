
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.ai_sentinel import AISentinel

sentinel = AISentinel()

# 1. Test Clean Code
clean_code = """
def main():
    x = 10 + 20
    return x
"""
print(f"Checking Clean Code... {'✅ PASS' if sentinel.scan_code(clean_code) else '❌ FAIL'}")

# 2. Test Malicious Import (OS)
mal_code_1 = """
import os
def hack():
    os.remove("critical_file.txt")
"""
print(f"Checking Malicious Import (os)... {'✅ BLOCKED' if not sentinel.scan_code(mal_code_1) else '❌ ALLOWED (DANGER)'}")

# 3. Test Malicious Function (Eval)
mal_code_2 = """
def sneaky():
    eval("print('hack')")
"""
print(f"Checking Malicious Function (eval)... {'✅ BLOCKED' if not sentinel.scan_code(mal_code_2) else '❌ ALLOWED (DANGER)'}")

# 4. Test Subprocess
mal_code_3 = """
from subprocess import call
call(["ls", "-l"])
"""
print(f"Checking Subprocess... {'✅ BLOCKED' if not sentinel.scan_code(mal_code_3) else '❌ ALLOWED (DANGER)'}")
