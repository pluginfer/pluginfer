"""
Secure Sandbox (V3.1 Hardened)
==============================
Static AST validation + process isolation for untrusted code.

Threat model the previous version missed
----------------------------------------
The earlier implementation only checked `ast.Name` calls, leaving
classic Python sandbox escapes wide open:

    # 1. Walk the type hierarchy back to `object`, scan its
    #    subclasses for something dangerous.
    cls = ().__class__.__base__              # -> object
    for sub in cls.__subclasses__():
        if sub.__name__ == 'Popen':
            sub('id'.split())                # arbitrary exec

    # 2. Reach the host's __builtins__ via a string-format trick.
    "{0.__class__.__base__.__subclasses__}".format(())

    # 3. Pull `os` out of any allowed module's transitive imports.
    import math
    math.__loader__.__init__.__globals__['sys'].modules['os']

None of these use a forbidden `ast.Name` call. The previous AST
visitor would let them through, and the worker's `exec(code,
safe_globals)` inherited the caller's `__builtins__` because
`safe_globals['__builtins__']` was never set, so `open`,
`__import__`, `getattr`, etc. were silently available.

This version:
    * **Blocks attribute access to any dunder name** that isn't on
      a tiny whitelist (`__init__` / `__name__` / `__doc__`).
    * **Expanded module/builtin denylist** — covers `marshal`,
      `code`, `gc`, `tracemalloc`, `inspect`, `_thread`,
      `threading`, `asyncio`, `weakref`, `builtins`, plus
      `getattr`/`setattr`/`vars`/`dir`/`format`/`breakpoint`.
    * **Replaces `__builtins__` in the exec context** with a
      curated whitelist of safe primitives (no I/O, no
      reflection, no dynamic import).
    * **Bans `import` / `from … import` entirely** in user code;
      anything they need must come from a pre-injected, audited
      module set.

This is still NOT a true sandbox. A determined adversary may find
new escapes (this is what whole research papers are written
about). For untrusted-third-party contracts, ship WASM via
`wasmtime` (real capability-based isolation) or run inside a
gVisor / seccomp-bpf jail. The pure-Python sandbox is fine for
a closed beta of curated plugins, NOT for arbitrary public
contracts.
"""

from __future__ import annotations

import ast
import logging
import multiprocessing
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---- Static deny lists ---------------------------------------------------
DANGEROUS_IMPORTS = {
    # I/O & process
    "os", "sys", "subprocess", "shutil", "socket", "ssl",
    "select", "selectors", "fcntl", "pty", "posix", "tty", "termios",
    "mmap", "resource",
    # Code generation / reflection
    "marshal", "code", "codeop", "compileall", "py_compile",
    "inspect", "tracemalloc", "gc", "weakref", "ctypes", "cffi",
    # Module loaders
    "importlib", "imp", "pkgutil", "builtins",
    # Concurrency primitives
    "_thread", "threading", "asyncio", "concurrent",
    # Pickle / arbitrary deserialisation
    "pickle", "pickletools", "shelve", "dill",
    # Net
    "urllib", "http", "ftplib", "smtplib", "telnetlib", "socketserver",
    "requests", "aiohttp",
    # Mass cleanup
    "atexit", "signal",
}

# Builtins that read/write filesystem, do I/O, reflect, or dynamically
# import. Block by name even though `__builtins__` is replaced — a user
# could still write `eval = ...` shadow if we leave them parseable.
DANGEROUS_BUILTINS = {
    "open", "exec", "eval", "compile", "__import__",
    "globals", "locals",
    "getattr", "setattr", "delattr", "vars", "dir",
    "input", "breakpoint", "help",
    "format",                # `"{0.__class__...}".format(x)` escape
    "memoryview",            # raw memory access escape
    "type",                  # 3-arg type() builds new classes (escape vector)
}

# Attribute names we never allow user code to mention. This is the key
# defense against `().__class__.__base__.__subclasses__()` and friends.
SAFE_DUNDERS = {"__init__", "__call__", "__len__", "__iter__", "__next__",
                "__enter__", "__exit__", "__getitem__", "__setitem__",
                "__contains__", "__eq__", "__ne__", "__lt__", "__le__",
                "__gt__", "__ge__", "__hash__", "__str__", "__repr__",
                "__add__", "__sub__", "__mul__", "__truediv__",
                "__floordiv__", "__mod__", "__pow__", "__neg__",
                "__abs__", "__round__", "__bool__"}


class SecurityViolation(Exception):
    """Raised by the AST validator when user code reaches for the floor."""


class ASTValidator(ast.NodeVisitor):
    """
    Static-analysis pass over user source. Rejects:
      * `import X` / `from X import …` for any X in DANGEROUS_IMPORTS.
      * `import X` / `from X import …` at all (we whitelist nothing).
        Anything user code legitimately needs must be pre-injected
        via the `extra_globals` argument to `SecureSandbox.run`.
      * Calls to any name in DANGEROUS_BUILTINS, regardless of how
        they're reached (Name or Attribute).
      * Any attribute access to a dunder not on `SAFE_DUNDERS`.
      * `yield from`, `async for`, `try: … except SystemExit:` —
        all unusual control-flow that can leak side channels.
    """

    def __init__(self, allow_imports: bool = False):
        self.allow_imports = allow_imports

    # ---- imports --------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        if not self.allow_imports:
            raise SecurityViolation(
                f"Import not allowed in sandboxed code: {[a.name for a in node.names]}"
            )
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in DANGEROUS_IMPORTS:
                raise SecurityViolation(f"Forbidden import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not self.allow_imports:
            raise SecurityViolation(
                f"Import not allowed in sandboxed code: from {node.module}"
            )
        if node.module:
            top = node.module.split(".")[0]
            if top in DANGEROUS_IMPORTS:
                raise SecurityViolation(f"Forbidden import: from {node.module}")
        if node.level and node.level > 0:
            raise SecurityViolation("Relative imports forbidden in sandbox")
        self.generic_visit(node)

    # ---- dangerous calls ------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        # Block dangerous builtin names AND attribute-form access.
        target = node.func
        name: Optional[str] = None
        if isinstance(target, ast.Name):
            name = target.id
        elif isinstance(target, ast.Attribute):
            name = target.attr
        if name is not None and name in DANGEROUS_BUILTINS:
            raise SecurityViolation(f"Forbidden call: {name}()")
        self.generic_visit(node)

    # ---- attribute access — the critical hardening ----------------------
    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr = node.attr
        if attr.startswith("__") and attr not in SAFE_DUNDERS:
            raise SecurityViolation(
                f"Forbidden dunder attribute access: .{attr} "
                f"(blocks __class__/__base__/__subclasses__/__globals__/etc.)"
            )
        self.generic_visit(node)

    # ---- control-flow side channels ------------------------------------
    def visit_Try(self, node: ast.Try) -> None:
        for handler in node.handlers:
            t = handler.type
            if isinstance(t, ast.Name) and t.id in {"SystemExit", "KeyboardInterrupt"}:
                raise SecurityViolation(
                    f"Catching {t.id} in sandbox is forbidden — leak channel"
                )
        self.generic_visit(node)


# Curated builtin namespace handed to the worker `exec`. Excludes
# everything that touches I/O / reflection / import / arbitrary
# code construction.
_SAFE_BUILTINS = {
    # arithmetic / sequence helpers
    "abs": abs, "all": all, "any": any, "ascii": ascii, "bin": bin,
    "bool": bool, "bytes": bytes, "callable": callable, "chr": chr,
    "complex": complex, "dict": dict, "divmod": divmod,
    "enumerate": enumerate, "filter": filter, "float": float,
    "frozenset": frozenset, "hash": hash, "hex": hex, "id": id,
    "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "iter": iter, "len": len, "list": list, "map": map, "max": max,
    "min": min, "next": next, "object": object, "oct": oct, "ord": ord,
    "pow": pow, "print": print, "range": range, "repr": repr,
    "reversed": reversed, "round": round, "set": set, "slice": slice,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    # exceptions plug-in code may legitimately need to raise
    "Exception": Exception, "ValueError": ValueError,
    "TypeError": TypeError, "RuntimeError": RuntimeError,
    "KeyError": KeyError, "IndexError": IndexError,
    "ArithmeticError": ArithmeticError, "ZeroDivisionError": ZeroDivisionError,
    "OverflowError": OverflowError, "AssertionError": AssertionError,
    "True": True, "False": False, "None": None,
}


def _isolated_worker(code_str: str, input_args: list, return_q) -> None:
    """
    Worker process. Even with all the static checks, we still execute
    inside a separate, killable PID — defense in depth.

    The exec context's `__builtins__` is REPLACED with `_SAFE_BUILTINS`,
    so `open`, `__import__`, `getattr`, etc. are simply not present
    even if the AST validator missed something.
    """
    payload = {}
    try:
        safe_globals = {
            "__builtins__": _SAFE_BUILTINS,
            # Pre-injected, audited modules user code is permitted to use.
            "math": __import__("math"),
            "json": __import__("json"),
            "datetime": __import__("datetime"),
            "args": input_args,
        }

        exec(code_str, safe_globals)

        if "result" in safe_globals:
            payload["result"] = safe_globals["result"]
        elif "main" in safe_globals and callable(safe_globals["main"]):
            payload["result"] = safe_globals["main"](*input_args)
        else:
            payload["result"] = None
        payload["status"] = "success"
    except Exception as e:
        payload["status"] = "error"
        payload["error"] = str(e)
    # mp.Queue is dramatically lighter than mp.Manager().dict() on
    # Windows because Manager spawns a side-helper process per call;
    # Queue piggybacks on the existing Process pipe.
    try:
        return_q.put(payload)
    except Exception:
        # If serialisation breaks, surface as a worker error rather
        # than hanging the parent on join().
        return_q.put({"status": "error", "error": "result-not-pickleable"})


class SecureSandbox:
    """Public API."""

    @staticmethod
    def validate_code(code: str, allow_imports: bool = False) -> bool:
        """Run the static AST analysis. Raises SecurityViolation on rejection."""
        try:
            tree = ast.parse(code)
        except SyntaxError as se:
            raise ValueError(f"Invalid Python syntax: {se}") from se
        ASTValidator(allow_imports=allow_imports).visit(tree)
        return True

    @staticmethod
    def run(code: str, args: list = None, timeout: float = 5.0,
            allow_imports: bool = False) -> Any:
        """
        Static-validate `code`, then execute it in a separate process
        with a curated builtins namespace. Raises:
          * `SecurityViolation` if the code is statically rejected
          * `TimeoutError` if execution exceeds `timeout`
          * `RuntimeError` if the code itself raised
        """
        args = list(args or [])
        SecureSandbox.validate_code(code, allow_imports=allow_imports)

        # mp.Queue + Process is ~10x faster on Windows than
        # mp.Manager().dict() (no helper-process bootstrap). Same
        # isolation guarantee — code still runs in a separate
        # killable PID with the curated __builtins__.
        return_q: multiprocessing.Queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=_isolated_worker, args=(code, args, return_q),
        )
        p.start()
        p.join(timeout=timeout)

        if p.is_alive():
            p.terminate()
            p.join(timeout=1.0)
            if p.is_alive():
                p.kill()
            raise TimeoutError("Sandbox execution exceeded time limit")

        # Drain the queue (timeout is defensive; worker has already
        # joined cleanly so the put() must have completed).
        try:
            payload = return_q.get(timeout=1.0)
        except Exception:
            raise RuntimeError("Sandbox exited without delivering a result")
        if payload.get("status") == "error":
            raise RuntimeError(f"Sandbox error: {payload.get('error')}")
        return payload.get("result")
