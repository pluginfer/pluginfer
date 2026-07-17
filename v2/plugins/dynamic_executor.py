"""
Dynamic Executor Plugin (Raw Compute).
Executes arbitrary client-provided Python code in a controlled environment.
"""
from typing import Dict, Any, List
import logging
import sys
import os
import base64
import traceback
from pathlib import Path

# Robust import strategy
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
parent_dir = str(Path(current_dir).parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
grandparent_dir = str(Path(current_dir).parent.parent.absolute())
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

try:
    from core.plugin_base import PluginBase
except ImportError:
    try:
        sys.path.append(os.path.join(current_dir, '..'))
        from core.plugin_base import PluginBase
    except ImportError:
         if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
             sys.path.append(sys._MEIPASS)
             from core.plugin_base import PluginBase
         else:
             raise

logger = logging.getLogger(__name__)

class DynamicExecutor(PluginBase):
    def config(self) -> Dict[str, Any]:
        return {
            "name": "dynamic_executor",
            "version": "1.0.0",
            "description": "Executes user-provided Python code (Raw Compute).",
            "category": "compute",
            "tags": ['compute', 'raw', 'python'],
            "cost_per_exec": 0.02, # Higher cost for raw compute
            "inputs": {
                'code': 'str',          # Base64 encoded Python script
                'function_name': 'str', # Entry point function name
                'args': 'dict'          # Arguments for the function
            },
            "outputs": {'result': 'any'}
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 1. Extract inputs
            b64_code = input_data.get('code')
            func_name = input_data.get('function_name', 'main')
            func_args = input_data.get('args', {})

            if not b64_code:
                return {"error": "Missing 'code' in input_data"}

            # 2. Decode Script
            try:
                script_code = base64.b64decode(b64_code).decode('utf-8')
            except Exception as e:
                return {"error": f"Failed to decode base64 code: {e}"}

            # 3. Security: route through SecureSandbox (AST static analysis +
            #    multiprocessing isolation + timeout). Any code with banned imports
            #    (os, sys, subprocess, socket, shutil, pickle, importlib, ctypes) or
            #    banned builtins (open, exec, eval, compile, __import__, globals,
            #    locals) is rejected before execution. Process isolation contains
            #    any escape attempt to a disposable subprocess.
            from core.secure_sandbox import SecureSandbox, SecurityViolation

            # The sandbox protocol: code defines a `main(*args)` function or assigns
            # to a variable named `result`. We pass func_args as positional args.
            try:
                positional_args = list(func_args.values()) if isinstance(func_args, dict) else list(func_args)
            except Exception:
                positional_args = []

            timeout_sec = float(input_data.get('timeout', 30.0))
            timeout_sec = max(1.0, min(timeout_sec, 300.0))  # clamp [1s, 5min]

            try:
                result = SecureSandbox.run(
                    code=script_code,
                    args=positional_args,
                    timeout=timeout_sec,
                )
            except SecurityViolation as sv:
                return {"error": f"Code rejected by sandbox: {sv}", "code": "SECURITY_VIOLATION"}
            except TimeoutError as te:
                return {"error": f"Execution timeout: {te}", "code": "TIMEOUT"}
            except (ValueError, RuntimeError) as ex:
                return {"error": str(ex), "code": "EXECUTION_ERROR"}

            return {
                "status": "success",
                "message": "Dynamic code executed successfully (sandboxed)",
                "result": result,
                "sandbox": "ast+process_isolated",
                "timeout_sec": timeout_sec,
            }

        except Exception as e:
            logger.error(f"Plugin dynamic_executor failed: {e}")
            return {"error": str(e), "traceback": traceback.format_exc()}
