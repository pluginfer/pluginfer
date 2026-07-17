"""
Pluginfer Smart Launcher
========================

Auto-detects the right Python interpreter on this machine and re-execs
the actual node entrypoint under it.  This solves three real-world pains:

  1. The user's default `python` may be too new (e.g. 3.14) to have
     torch + CUDA wheels yet.  The smart launcher walks every Python
     installation it can find via `py -0`, tests each for working
     torch, and picks the one with the strongest GPU support.
  2. Required runtime deps (torch / cryptography / flask / psutil /
     keyring) may be missing on the chosen interpreter.  The launcher
     `pip install`s them on demand, with a confirmation prompt only on
     interactive runs.
  3. Re-spawning the actual node under the right interpreter so users
     don't have to know about Python versions at all.

Selection priority (high → low):
  * Python 3.11 with torch built and `torch.cuda.is_available() == True`
  * Python 3.10 with torch + CUDA
  * Python 3.12 with torch + CUDA
  * Any Python 3.11/3.10/3.12 with torch DirectML / CPU only
  * Anything else with cryptography (pure-chain mode, no inference)

The hard floor is "can import core/" — chain features only.  Inference
features additionally need torch.

Designed for Windows, macOS, and Linux; uses the `py` launcher when
available, and falls back to scanning `PATH` and well-known install
locations.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

HERE = Path(__file__).resolve().parent
NODE_ENTRY = HERE / "pluginfer_node.py"

REQUIRED_HARD = [
    # (import name, pip-installable name)
    ("cryptography", "cryptography>=41.0.0"),
    ("flask", "flask>=3.0.0"),
    ("psutil", "psutil>=5.9.0"),
    ("requests", "requests>=2.31.0"),
]
REQUIRED_INFERENCE = [
    ("torch", "torch>=2.0.0"),
    ("numpy", "numpy>=1.24.0"),
]
OPTIONAL_NICE = [
    ("keyring", "keyring>=24.0.0"),
    ("pystray", "pystray>=0.19.0"),
    ("PIL", "Pillow>=10.0.0"),
    ("cpuinfo", "py-cpuinfo>=9.0.0"),
]
PREFERRED_VERSION_ORDER = ["3.11", "3.12", "3.10", "3.13"]


def _which_pythons() -> List[str]:
    """Enumerate Python interpreters on this machine.  Returns a
    deduplicated list of absolute paths.  Robust to: Windows `py`
    launcher, PATH-listed `python` / `python3`, and well-known install
    locations under %LOCALAPPDATA%."""
    candidates: List[str] = []

    # 1. Windows `py` launcher: `py -0p` lists all installs with paths.
    try:
        out = subprocess.run(
            ["py", "-0p"], capture_output=True, text=True, timeout=10,
        )
        for line in out.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("--") or line.startswith("Available"):
                continue
            # format: " -V:3.11 *        C:\\Path\\To\\python.exe"
            parts = line.split()
            for p in parts:
                if p.lower().endswith("python.exe") or p.lower().endswith("python3"):
                    candidates.append(p)
                    break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 2. PATH lookups
    for name in ("python3.11", "python3.12", "python3.10",
                 "python3.13", "python3", "python"):
        which = shutil.which(name)
        if which:
            candidates.append(which)

    # 3. Well-known Windows install dirs.
    if os.name == "nt":
        for ver in ("311", "312", "310", "313", "314"):
            for base in (Path(os.environ.get("LOCALAPPDATA", "")) /
                         "Programs" / "Python" / f"Python{ver}",
                         Path(f"C:/Python{ver}"),
                         Path(f"C:/Program Files/Python{ver}")):
                exe = base / ("python.exe" if os.name == "nt" else "bin/python")
                if exe.exists():
                    candidates.append(str(exe))

    # Dedupe by resolved path; keep insertion order.
    seen = set()
    uniq: List[str] = []
    for c in candidates:
        try:
            r = str(Path(c).resolve())
        except Exception:
            r = c
        if r not in seen:
            seen.add(r)
            uniq.append(c)
    return uniq


_PROBE_SCRIPT = r"""
import sys, json, importlib
rep = {
    'ok': sys.version_info[0] == 3 and sys.version_info[1] >= 9,
    'version': '%d.%d.%d' % sys.version_info[:3],
    'torch': None, 'cuda': False, 'directml': False, 'missing': [],
}
for m in ('cryptography', 'flask', 'psutil', 'requests'):
    try:
        importlib.import_module(m)
    except Exception:
        rep['missing'].append(m)
try:
    import torch
    rep['torch'] = torch.__version__
    rep['cuda'] = bool(torch.cuda.is_available())
except Exception:
    rep['missing'].append('torch')
try:
    import torch_directml  # noqa: F401
    rep['directml'] = True
except Exception:
    pass
print(json.dumps(rep))
"""


def _probe_interpreter(exe: str) -> dict:
    """Return a capability report for `exe`:
       {ok, version, torch, cuda, has_hard_deps, missing}."""
    try:
        out = subprocess.run(
            [exe, "-c", _PROBE_SCRIPT], capture_output=True,
            text=True, timeout=15,
        )
        if out.returncode != 0:
            return {"ok": False, "exe": exe,
                    "error": out.stderr.strip()[:200]}
        line = out.stdout.strip().splitlines()[-1]
        rep = json.loads(line)
        rep["exe"] = exe
        rep["ok"] = rep.get("ok", False)
        return rep
    except Exception as e:
        return {"ok": False, "exe": exe, "error": str(e)[:200]}


def _score(rep: dict) -> Tuple[int, ...]:
    """Higher tuple = better. Compared lexicographically.

    Tier 1: torch CUDA available.
    Tier 2: torch DirectML available.
    Tier 3: torch CPU only.
    Tier 4: no torch but the chain stack works.
    Within tier, prefer 3.11 > 3.12 > 3.10 > 3.13 > 3.14 (because torch
    wheels lag Python release by ~6 months).
    """
    if not rep.get("ok"):
        return (-1,)
    has_cuda = 1 if rep.get("cuda") else 0
    has_dml = 1 if rep.get("directml") else 0
    has_torch = 1 if rep.get("torch") else 0
    has_hard = 0 if rep.get("missing") else 1
    ver = rep.get("version", "0.0")
    minor = int(ver.split(".")[1]) if "." in ver else 0
    # Map preferred minor to a positive score; 3.11 best.
    pref = {11: 5, 12: 4, 10: 3, 13: 2, 14: 1}.get(minor, 0)
    return (has_cuda, has_dml, has_torch, has_hard, pref)


def _pip_install(exe: str, packages: List[str]) -> bool:
    """`<exe> -m pip install <pkgs>`.  Returns True on success."""
    if not packages:
        return True
    print(f"[smart-launcher] pip install ({len(packages)} pkgs) "
          f"under {exe}")
    args = [exe, "-m", "pip", "install", "--upgrade"] + packages
    try:
        out = subprocess.run(args, capture_output=False, text=True)
        return out.returncode == 0
    except Exception as e:
        print(f"[smart-launcher] pip install failed: {e}")
        return False


def _ensure_deps(exe: str, rep: dict, *, install: bool) -> dict:
    """Install missing hard + inference deps if `install`.  Returns
    a refreshed report."""
    missing = list(rep.get("missing") or [])
    if not missing:
        return rep
    if not install:
        print(f"[smart-launcher] {exe} is missing: {missing}")
        return rep
    pkgs = []
    for short, pkg in REQUIRED_HARD + REQUIRED_INFERENCE:
        if short in missing:
            pkgs.append(pkg)
    if pkgs:
        ok = _pip_install(exe, pkgs)
        if ok:
            return _probe_interpreter(exe)
    return rep


def _print_report(rep: dict) -> None:
    cuda = "CUDA" if rep.get("cuda") else (
        "DirectML" if rep.get("directml") else
        ("CPU-only" if rep.get("torch") else "no-torch"))
    miss = ",".join(rep.get("missing") or []) or "-"
    print(f"  {rep.get('version', '?'):8s} {cuda:9s} torch={rep.get('torch') or '-':12s} "
          f"missing={miss:30s} {rep.get('exe', '?')}")


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])

    # Plumbing flags the smart launcher consumes; everything else is
    # forwarded to pluginfer_node.py.
    install_missing = "--no-install" not in argv
    if "--no-install" in argv:
        argv.remove("--no-install")
    only_chain = "--no-torch" in argv
    if "--no-torch" in argv:
        argv.remove("--no-torch")
    show_only = "--probe" in argv
    if "--probe" in argv:
        argv.remove("--probe")

    print("=" * 70)
    print("  PLUGINFER SMART LAUNCHER")
    print("=" * 70)
    print(f"  scanning Python interpreters on {sys.platform}...\n")

    pythons = _which_pythons()
    if not pythons:
        print("  no Python interpreters found.")
        print("  install Python 3.11 from https://www.python.org/downloads/")
        return 2

    reports = [_probe_interpreter(p) for p in pythons]
    print("  capability survey:")
    for r in reports:
        _print_report(r)

    # Rank.
    ranked = sorted(reports, key=_score, reverse=True)
    best = ranked[0] if ranked else None
    if best is None or not best.get("ok"):
        print("  no usable Python found.")
        return 2

    print(f"\n  selected: {best.get('exe')} (Python {best.get('version')})")

    # Install missing on the chosen one (only if user didn't disable).
    refreshed = _ensure_deps(best["exe"], best, install=install_missing)
    if refreshed.get("missing"):
        if only_chain:
            # User asked for chain-only mode — don't fail on missing torch.
            non_torch_missing = [m for m in refreshed["missing"]
                                  if m != "torch"]
            if non_torch_missing:
                print(f"  ERROR: still missing {non_torch_missing}")
                return 3
        else:
            print(f"  ERROR: still missing {refreshed['missing']}")
            return 3

    if show_only:
        print("\n  --probe set; not launching node.")
        return 0

    # Re-exec the real node under the chosen interpreter.
    cmd = [refreshed["exe"], str(NODE_ENTRY)] + argv
    print(f"\n  spawning: {' '.join(shlex.quote(c) for c in cmd)}")
    print("=" * 70)
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
