"""Test-time path shim for the ai/ package.

Background: the project ships a vendored `v2/torch/` directory (left
over from a PyInstaller dump). Its DLLs are broken on this machine
(see WORKLOG 9876524 / W18). When pytest is run with cwd=v2/, pytest
puts v2/ on sys.path[0], so `import torch` finds the broken vendored
copy first and OSErrors before any test code runs. The chain-only
modules (core/) work around it with `try: import torch except ImportError`,
but ai/ requires torch -- so we must guarantee the SYSTEM torch loads.

Fix: this conftest runs once before any ai/ test imports its module,
filters v2/ off sys.path, imports torch (so it caches under the system
location in sys.modules), then restores sys.path so test files can
still do `sys.path.insert(0, ROOT)` to import the `ai` package.

Side effect of being in conftest.py at v2/ai/: pytest evaluates this
on collection of any test under ai/, before exec_module of the test
file itself.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Host-guard FIRST — thread caps must be exported before the torch
# import below; see tests/conftest.py for the full rationale.
if "host_guard" not in sys.modules:
    _hg_spec = importlib.util.spec_from_file_location(
        "host_guard",
        Path(__file__).resolve().parents[1] / "host_guard.py",
    )
    _hg = importlib.util.module_from_spec(_hg_spec)
    try:
        _hg_spec.loader.exec_module(_hg)
        sys.modules["host_guard"] = _hg
    except Exception:
        _hg = None
if sys.modules.get("host_guard") is not None:
    sys.modules["host_guard"].install("pytest")

if "torch" not in sys.modules:
    v2 = Path(__file__).resolve().parents[1]
    saved = sys.path[:]
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != v2]
    try:
        import torch  # noqa: F401  - cached in sys.modules
    finally:
        sys.path[:] = saved
