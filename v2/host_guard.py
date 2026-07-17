"""host_guard — Pluginfer must NEVER hang the machine it runs on.

Motivation (2026-07-17): running the full test suite (torch saturating
all 12 logical cores at normal priority) while Ollama held qwen2.5:14b
(~10 GB resident, keep_alive=60m) on a 16 GB laptop swap-stormed the
host into a multi-minute freeze — terminal included. That is an
architecture bug, not an operator error: a mesh node's first duty to
its operator is to be a polite guest on their hardware.

This module is the single choke point for host protection. Every
entrypoint (auto_mesh node, run_node supervisor, the desktop app,
pytest via conftest) calls `install()` FIRST — before torch or any
BLAS library loads — and every load-bearing consumer asks it before
taking on work:

  * `should_accept_work()`  — JobsService.submit gate,
  * `fits_model(bytes)`     — Ollama model negotiation gate,
  * `headroom_bytes()`      — anyone sizing a buffer/model/batch.

What install() enforces, in order of importance:

  1. **Windows Job Object** wrapping this process AND every child it
     spawns (children inherit membership):
       - `JOB_OBJECT_LIMIT_JOB_MEMORY`: committed memory of the whole
         tree is hard-capped at (total RAM − host reserve). Exceeding
         it makes *our* allocations fail (MemoryError → a visible test
         failure / job failure) instead of pushing the OS into swap.
       - `JOB_OBJECT_LIMIT_PRIORITY_CLASS` = BELOW_NORMAL: the
         operator's shell, browser and Explorer always outrank us for
         CPU. A saturated suite slows itself, not the host.
       - `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`: when the parent dies,
         every stray node server / test subprocess dies with it — no
         orphaned uvicorns quietly eating RAM after a crashed run.
     POSIX equivalent: `os.nice(10)` + `RLIMIT_AS` at the same cap.

  2. **Thread caps before the first BLAS/torch import**: OMP / MKL /
     OpenBLAS / numexpr default to one thread per logical core each;
     torch on 12 threads at normal priority is indistinguishable from
     a hang for the user. We default every library to half the logical
     cores (env-overridable), leaving the OS interactive.

  3. **Memory watchdog** (daemon thread, ~5 s cadence): tracks
     system-wide available RAM. Below the soft floor the node stops
     accepting new work and fires registered shed callbacks; below the
     critical floor it additionally gc-collects and trims its own
     working set. Recovery is automatic with hysteresis.

Deliberate non-goals: we never kill the host's other processes, never
raise from install(), and never lie — a rejected job carries an honest
`host_guard:` detail string, and a model that doesn't fit is refused,
not silently swapped for the echo.

Import cost is nil (stdlib + optional psutil); nothing starts until
`install()` is called. `PLUGINFER_HOST_GUARD=0` is the operator escape
hatch that turns the whole module into no-ops.

Env knobs (all optional):
  PLUGINFER_HOST_GUARD        "0" disables everything (default on)
  PLUGINFER_HOST_RESERVE_MB   RAM the tree must leave the OS (3072)
  PLUGINFER_JOB_MEM_MB        absolute tree cap, overrides reserve math
  PLUGINFER_MEM_SOFT_MB       stop accepting new work below (2048)
  PLUGINFER_MEM_CRIT_MB       shed + trim below (1024)
  PLUGINFER_THREAD_CAP        BLAS/torch threads (logical cores // 2)
  PLUGINFER_GUARD_INTERVAL_S  watchdog cadence (5)
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import sys
import threading
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("pluginfer.host_guard")

_GB = 1024 * 1024 * 1024
_MB = 1024 * 1024

# ---------------------------------------------------------------------------
# Module state — one instance per process, guarded by _LOCK.
# ---------------------------------------------------------------------------

# Reentrant on purpose: install() returns status() while holding the
# lock on the already-installed / disabled paths.
_LOCK = threading.RLock()
_STATE: Dict[str, Any] = {
    "installed": False,
    "enabled": True,
    "role": None,
    "pressure": "ok",          # ok | soft | critical
    "accepting": True,
    "job_handle": None,        # keep-alive: closing it kills the tree
    "job_assigned": False,
    "priority_set": False,
    "thread_cap": None,
    "job_mem_limit_bytes": None,
    "watchdog": None,
}
_SHED_CALLBACKS: List[Callable[[str], None]] = []


def _env_mb(name: str, default_mb: int) -> int:
    try:
        return int(os.environ.get(name, "") or default_mb)
    except ValueError:
        return default_mb


def _enabled() -> bool:
    return os.environ.get("PLUGINFER_HOST_GUARD", "1") != "0"


# ---------------------------------------------------------------------------
# Memory sampling — psutil when present, GlobalMemoryStatusEx fallback.
# ---------------------------------------------------------------------------

class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def _mem_sample() -> "tuple[int, int]":
    """(total_bytes, available_bytes) for the whole machine."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:
        pass
    if sys.platform == "win32":
        st = _MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullTotalPhys), int(st.ullAvailPhys)
    # Last resort: pretend 8 GB total / 4 GB free so gates stay sane
    # rather than dividing by zero. Logged once at install().
    return 8 * _GB, 4 * _GB


def total_bytes() -> int:
    return _mem_sample()[0]


def headroom_bytes() -> int:
    """System-wide available physical RAM right now."""
    return _mem_sample()[1]


# ---------------------------------------------------------------------------
# Windows Job Object plumbing (ctypes; no pywin32 dependency).
# ---------------------------------------------------------------------------

_JOB_OBJECT_LIMIT_PRIORITY_CLASS = 0x00000020
_JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
_JobObjectExtendedLimitInformation = 9


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _install_job_object(mem_limit_bytes: int) -> "tuple[Optional[int], bool]":
    """Create + configure a job object and assign ourselves to it.
    Returns (handle, assigned). Children inherit membership, so one
    call at the entrypoint covers the whole future process tree.
    Never raises — a machine where this fails (exotic sandboxing)
    still gets the priority/thread/watchdog layers."""
    if sys.platform != "win32":
        return None, False
    try:
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        # 64-bit HANDLEs truncate through ctypes' default c_int
        # restype — prototype everything explicitly or assignment
        # fails silently with a garbage handle.
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [
            wintypes.LPVOID, wintypes.LPCWSTR]
        k32.SetInformationJobObject.restype = wintypes.BOOL
        k32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int,
            wintypes.LPVOID, wintypes.DWORD]
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        k32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE, wintypes.HANDLE]
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.GetCurrentProcess.argtypes = []
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = k32.CreateJobObjectW(None, None)
        if not handle:
            return None, False
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_JOB_MEMORY
            | _JOB_OBJECT_LIMIT_PRIORITY_CLASS
            | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        info.BasicLimitInformation.PriorityClass = (
            _BELOW_NORMAL_PRIORITY_CLASS
        )
        info.JobMemoryLimit = int(mem_limit_bytes)
        ok = k32.SetInformationJobObject(
            handle, _JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info),
        )
        if not ok:
            k32.CloseHandle(handle)
            return None, False
        assigned = bool(k32.AssignProcessToJobObject(
            handle, k32.GetCurrentProcess(),
        ))
        if not assigned:
            # Already in an unnestable job (pre-Win8 only) or access
            # denied. Keep the handle closed; other layers still apply.
            k32.CloseHandle(handle)
            return None, False
        return handle, True
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("job object setup failed: %s", e)
        return None, False


def in_job() -> bool:
    """True when this process runs inside ANY Windows job object."""
    if sys.platform != "win32":
        return False
    try:
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.IsProcessInJob.restype = wintypes.BOOL
        k32.IsProcessInJob.argtypes = [
            wintypes.HANDLE, wintypes.HANDLE,
            ctypes.POINTER(wintypes.BOOL)]
        res = wintypes.BOOL(0)
        k32.IsProcessInJob(k32.GetCurrentProcess(), None, ctypes.byref(res))
        return bool(res.value)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Priority + thread caps
# ---------------------------------------------------------------------------

def _lower_priority() -> bool:
    """Belt-and-braces: set our own priority class even when the job
    object already enforces it (covers the assignment-failed path)."""
    try:
        if sys.platform == "win32":
            import psutil
            p = psutil.Process()
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            try:
                p.ionice(psutil.IOPRIO_LOW)
            except Exception:
                pass
        else:
            os.nice(10)
        return True
    except Exception as e:
        logger.debug("priority lowering unavailable: %s", e)
        return False


def _cap_threads() -> int:
    """Cap BLAS/torch parallelism BEFORE those libraries initialise.
    setdefault only — an operator's explicit OMP_NUM_THREADS wins."""
    cpu = os.cpu_count() or 4
    cap = _env_mb("PLUGINFER_THREAD_CAP", max(1, cpu // 2))
    cap = max(1, cap)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(cap))
    if "torch" in sys.modules:
        # Too late for interop threads, but intra-op still applies.
        try:
            sys.modules["torch"].set_num_threads(cap)
        except Exception:
            pass
    return cap


def _posix_mem_limit(mem_limit_bytes: int) -> None:
    if sys.platform == "win32":
        return
    try:
        import resource
        resource.setrlimit(
            resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes),
        )
    except Exception as e:
        logger.debug("RLIMIT_AS unavailable: %s", e)


# ---------------------------------------------------------------------------
# Watchdog — pressure classification is a pure function so tests can
# drive transitions without threads or real memory pressure.
# ---------------------------------------------------------------------------

def _floors() -> "tuple[int, int]":
    soft = _env_mb("PLUGINFER_MEM_SOFT_MB", 2048) * _MB
    crit = _env_mb("PLUGINFER_MEM_CRIT_MB", 1024) * _MB
    return soft, min(crit, soft)


def _classify(avail_bytes: int) -> str:
    soft, crit = _floors()
    if avail_bytes < crit:
        return "critical"
    if avail_bytes < soft:
        return "soft"
    # Hysteresis: only fully recover once we clear the soft floor with
    # 25% margin, so we don't flap on the boundary.
    if _STATE["pressure"] != "ok" and avail_bytes < int(soft * 1.25):
        return _STATE["pressure"]
    return "ok"


def _trim_self() -> None:
    """Give RAM back NOW: gc + (on Windows) empty our working set."""
    try:
        gc.collect()
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            ctypes.windll.psapi.EmptyWorkingSet(
                ctypes.windll.kernel32.GetCurrentProcess(),
            )
        except Exception:
            pass


def _apply_state(new: str) -> None:
    with _LOCK:
        old = _STATE["pressure"]
        if new == old:
            return
        _STATE["pressure"] = new
        _STATE["accepting"] = new == "ok"
    if new == "ok":
        logger.info("host memory pressure cleared — accepting work again")
        return
    logger.warning(
        "host memory pressure %s (available=%.1f GB) — %s",
        new.upper(), headroom_bytes() / _GB,
        "shedding + trimming" if new == "critical"
        else "pausing new work",
    )
    if new == "critical":
        _trim_self()
    for cb in list(_SHED_CALLBACKS):
        try:
            cb(new)
        except Exception as e:
            logger.warning("shed callback %s failed: %s", cb, e)


def _watchdog_loop(interval_s: float) -> None:
    while True:
        try:
            _apply_state(_classify(headroom_bytes()))
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("watchdog tick failed: %s", e)
        threading.Event().wait(interval_s)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def install(role: str = "node") -> Dict[str, Any]:
    """Apply every host-protection layer. Idempotent; never raises.
    Call FIRST at every entrypoint — before importing core/torch."""
    with _LOCK:
        if _STATE["installed"]:
            return status()
        _STATE["installed"] = True
        _STATE["role"] = role
        if not _enabled():
            _STATE["enabled"] = False
            logger.warning(
                "PLUGINFER_HOST_GUARD=0 — host protection DISABLED by "
                "operator. The host may hang under load.",
            )
            return status()

    total = total_bytes()
    reserve = _env_mb("PLUGINFER_HOST_RESERVE_MB", 3072) * _MB
    override = _env_mb("PLUGINFER_JOB_MEM_MB", 0) * _MB
    mem_limit = override or max(_GB, total - reserve)

    _STATE["thread_cap"] = _cap_threads()
    _STATE["job_mem_limit_bytes"] = mem_limit
    handle, assigned = _install_job_object(mem_limit)
    _STATE["job_handle"] = handle       # keep alive for process lifetime
    _STATE["job_assigned"] = assigned
    _posix_mem_limit(mem_limit)
    _STATE["priority_set"] = _lower_priority()

    interval = float(os.environ.get("PLUGINFER_GUARD_INTERVAL_S", "5"))
    t = threading.Thread(
        target=_watchdog_loop, args=(max(1.0, interval),),
        name="pluginfer-host-guard", daemon=True,
    )
    t.start()
    _STATE["watchdog"] = t

    logger.info(
        "host_guard installed (role=%s): tree mem cap %.1f GB, "
        "priority below-normal (job=%s), threads capped at %d, "
        "watchdog every %.0fs (soft %.1f GB / crit %.1f GB free)",
        role, mem_limit / _GB, assigned, _STATE["thread_cap"],
        interval, _floors()[0] / _GB, _floors()[1] / _GB,
    )
    return status()


def status() -> Dict[str, Any]:
    """Observability snapshot — surfaced via /v1/hardware."""
    with _LOCK:
        return {
            "installed": _STATE["installed"],
            "enabled": _STATE["enabled"],
            "role": _STATE["role"],
            "pressure": _STATE["pressure"],
            "accepting": _STATE["accepting"],
            "job_assigned": _STATE["job_assigned"],
            "priority_set": _STATE["priority_set"],
            "thread_cap": _STATE["thread_cap"],
            "job_mem_limit_bytes": _STATE["job_mem_limit_bytes"],
        }


def should_accept_work() -> bool:
    """False while the host is under memory pressure. JobsService
    rejects (honestly, terminally) instead of queueing work the
    machine cannot take."""
    if not _enabled():
        return True
    with _LOCK:
        return bool(_STATE["accepting"])


def register_shed_callback(cb: Callable[[str], None]) -> None:
    """cb(pressure_level) fires on every ok→soft/critical transition.
    Consumers use it to cancel queued work / drop caches."""
    _SHED_CALLBACKS.append(cb)


def estimate_model_footprint(disk_bytes: int) -> int:
    """Runtime RSS of a quantized GGUF ≈ its disk size plus KV-cache
    and runtime buffers. 15% + 1.5 GB is deliberately conservative:
    under-estimating is how a 14B model froze this laptop."""
    return int(disk_bytes * 1.15) + int(1.5 * _GB)


def fits_model(disk_bytes: int) -> bool:
    """Can the host load a model of this on-disk size AND stay above
    the soft floor? Unknown sizes (0) pass — we can't judge them, and
    refusing everything unknown would break remote/custom runtimes."""
    if not _enabled() or not disk_bytes:
        return True
    soft, _ = _floors()
    return estimate_model_footprint(disk_bytes) <= headroom_bytes() - soft
