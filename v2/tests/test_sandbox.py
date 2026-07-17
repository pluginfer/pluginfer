"""
SecureSandbox hardening regression test
========================================
Proves the classic Python sandbox escapes are now blocked at the
AST validation step (i.e. before any code runs).

Cases:
  1. Safe code (sum) executes and returns the right answer.
  2. `import os` raises SecurityViolation.
  3. `open(...)` raises SecurityViolation.
  4. `getattr(...)` raises SecurityViolation.
  5. `().__class__.__base__.__subclasses__()` -- the canonical escape --
     raises SecurityViolation (dunder-attribute access is forbidden).
  6. `"{0.__class__}".format(())` raises SecurityViolation
     (format() is in DANGEROUS_BUILTINS).
  7. `import math; math.__loader__` raises SecurityViolation
     (dunder-attribute access).
  8. `try: ... except SystemExit:` raises SecurityViolation.
  9. The worker context's `__builtins__` lacks `open` / `__import__`
     even if the static check were somehow bypassed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for parent in [_HERE.parents[1], _HERE.parents[2]]:
    if (parent / "core").is_dir():
        sys.path.insert(0, str(parent))
        break

from core.secure_sandbox import (                                    # noqa: E402
    SecureSandbox, SecurityViolation, _SAFE_BUILTINS,
)


def _expect_violation(label, code):
    try:
        SecureSandbox.validate_code(code)
    except SecurityViolation as e:
        print(f"  [{label}] BLOCKED OK: {e}")
        return
    raise AssertionError(f"[{label}] sandbox FAILED to block: {code!r}")


def test_safe_code_runs():
    print("\n[1] SAFE CODE EXECUTES")
    print("-" * 60)
    out = SecureSandbox.run("result = sum(args)", args=[1, 2, 3, 4])
    assert out == 10, f"expected 10 got {out}"
    print("  sum([1,2,3,4]) -> 10 OK")
    print("  PASS")


def test_blocks_import_os():
    print("\n[2] BLOCKS 'import os'")
    print("-" * 60)
    _expect_violation("import os", "import os")
    _expect_violation("from os import path", "from os import path")
    _expect_violation("from subprocess import run", "from subprocess import run")
    print("  PASS")


def test_blocks_dangerous_builtins():
    print("\n[3] BLOCKS open / eval / exec / __import__ / getattr")
    print("-" * 60)
    _expect_violation("open()", "open('x')")
    _expect_violation("eval()", "eval('1+1')")
    _expect_violation("exec()", "exec('x=1')")
    _expect_violation("__import__()", "__import__('os')")
    _expect_violation("getattr()", "getattr(x, 'foo')")
    _expect_violation("vars()", "vars(x)")
    _expect_violation("format()", "'{0}'.format(x)")
    print("  PASS")


def test_blocks_dunder_attribute_access():
    print("\n[4] BLOCKS __class__/__bases__/__subclasses__/__globals__")
    print("-" * 60)
    _expect_violation("__class__",
                      "x = ().__class__")
    _expect_violation("__base__ chain",
                      "x = ().__class__.__base__")
    _expect_violation("__subclasses__",
                      "for s in ().__class__.__base__.__subclasses__(): pass")
    _expect_violation("__globals__",
                      "x = print.__globals__")
    _expect_violation("__builtins__ via attr",
                      "x = print.__builtins__")
    _expect_violation("import + dunder",
                      "import math\nx = math.__loader__")
    print("  PASS")


def test_blocks_systemexit_handler():
    print("\n[5] BLOCKS catching SystemExit/KeyboardInterrupt")
    print("-" * 60)
    _expect_violation("except SystemExit",
                      "try:\n  pass\nexcept SystemExit:\n  pass")
    print("  PASS")


def test_safe_dunders_still_allowed():
    print("\n[6] SAFE DUNDERS (__init__/__str__/...) STILL OK")
    print("-" * 60)
    SecureSandbox.validate_code(
        "class C:\n  def __init__(self): self.x = 1\nc = C(); result = str(c)"
    )
    print("  __init__ + __str__ allowed OK")
    print("  PASS")


def test_safe_builtins_lacks_dangerous():
    print("\n[7] _SAFE_BUILTINS DOES NOT EXPOSE open/__import__/getattr")
    print("-" * 60)
    for forbidden in ("open", "__import__", "getattr", "setattr",
                      "exec", "eval", "compile", "vars", "dir"):
        assert forbidden not in _SAFE_BUILTINS, \
            f"_SAFE_BUILTINS exposes {forbidden}"
    print(f"  Whitelist size: {len(_SAFE_BUILTINS)} entries")
    print("  PASS")


def test_blocks_classic_subclasses_escape():
    print("\n[8] CLASSIC ESCAPE BLOCKED end-to-end")
    print("-" * 60)
    classic = (
        "for sub in ().__class__.__base__.__subclasses__():\n"
        "    if sub.__name__ == 'Popen':\n"
        "        sub('id'.split())"
    )
    try:
        SecureSandbox.run(classic)
    except SecurityViolation as e:
        print(f"  CLASSIC ESCAPE BLOCKED: {e}")
        print("  PASS")
        return
    raise AssertionError("Classic sandbox escape was NOT blocked!")


if __name__ == "__main__":
    print("=" * 60)
    print("SECURE SANDBOX HARDENING TEST")
    print("=" * 60)
    import time
    t0 = time.time()
    test_safe_code_runs()
    test_blocks_import_os()
    test_blocks_dangerous_builtins()
    test_blocks_dunder_attribute_access()
    test_blocks_systemexit_handler()
    test_safe_dunders_still_allowed()
    test_safe_builtins_lacks_dangerous()
    test_blocks_classic_subclasses_escape()
    print("\n" + "=" * 60)
    print(f"ALL SANDBOX HARDENING TESTS PASSED in {time.time() - t0:.1f}s")
    print("=" * 60)
