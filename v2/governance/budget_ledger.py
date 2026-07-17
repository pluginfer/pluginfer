"""BudgetLedger — Budget-as-Contract (§RFC-3 / HG13).

The enterprise problem this solves (research/04_cost_governance.md):
73% of enterprises overrun AI budgets because token spend has no hard
rate limit, no attribution, and no forecastability. Every existing tool
OBSERVES spend after the fact; this ledger ENFORCES it before the job
runs — the host_guard idea applied to money: a fail-closed gate at one
choke point.

Model:

  * An **envelope** is a spend cap on a slash-separated path, e.g.
    ``acme``, ``acme/support``, ``acme/support/chatbot``. Enforcement
    walks EVERY configured prefix of a job's envelope path — the org
    cap and the team cap both bind, so a runaway agent exhausts its own
    envelope without eating the whole org's.
  * Caps are per **period**: 'day' | 'week' | 'month' (30 d) | 'total'.
    Windows are fixed-length, anchored at envelope creation, and roll
    over automatically (spent resets; the journal keeps history).
  * The job lifecycle is **reserve → settle | release**:
      - ``reserve(job_id, path, ceiling)`` at submit, BEFORE the
        auction — holds the worst-case amount. Returns a reason string
        on refusal (fail-closed), None on success.
      - ``settle(job_id, actual)`` at terminal completed — converts
        the hold into recorded spend at the auction-cleared price.
      - ``release(job_id)`` at terminal failed — drops the hold.
    All three are idempotent per job_id, mirroring BuyerLedger's
    escrow semantics. Reservations self-expire after RESERVATION_TTL_S
    so a crashed path can never permanently wedge an envelope.
  * Every event is journalled (JSONL when a state dir is given) with
    the caller's metadata — chargeback ("the $455M went WHERE?") is
    ``report()``, a GROUP BY over that journal.

Deliberate semantics:
  * No envelopes configured on a path → allow (governance is opt-in;
    a bare node behaves exactly as before). ``require_envelope=True``
    flips that: a path covered by NO envelope is refused — the
    gateway's strict mode for orgs that want zero ungoverned spend.
  * Never raises out of the public API; IO failures degrade to
    in-memory operation with a logged warning. Enforcement decisions
    never depend on the disk.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("pluginfer.budget_ledger")

PERIOD_SECONDS = {
    "day": 86_400.0,
    "week": 7 * 86_400.0,
    "month": 30 * 86_400.0,
    "total": None,          # never rolls over
}

RESERVATION_TTL_S = 3600.0


@dataclass
class Envelope:
    path: str
    cap_usd: float
    period: str = "month"
    spent_usd: float = 0.0
    window_start: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path, "cap_usd": self.cap_usd,
            "period": self.period, "spent_usd": self.spent_usd,
            "window_start": self.window_start,
        }


@dataclass
class _Reservation:
    job_id: str
    path: str
    amount_usd: float
    created_at: float
    meta: Dict[str, Any] = field(default_factory=dict)


def _prefixes(path: str) -> List[str]:
    """'a/b/c' -> ['a', 'a/b', 'a/b/c']."""
    parts = [p for p in path.split("/") if p]
    return ["/".join(parts[: i + 1]) for i in range(len(parts))]


class BudgetLedger:
    """Thread-safe. `clock` is injectable so tests drive rollover
    deterministically without sleeping."""

    def __init__(self, state_dir: Optional[str] = None, *,
                 require_envelope: bool = False,
                 clock: Callable[[], float] = time.time):
        self._lock = threading.RLock()
        self._clock = clock
        self.require_envelope = require_envelope
        self._envelopes: Dict[str, Envelope] = {}
        self._reservations: Dict[str, _Reservation] = {}
        self._settled_job_ids: set = set()
        self._journal_mem: List[Dict[str, Any]] = []
        self._state_dir: Optional[Path] = None
        if state_dir:
            try:
                d = Path(state_dir)
                d.mkdir(parents=True, exist_ok=True)
                self._state_dir = d
                self._load_state()
            except Exception as e:
                logger.warning(
                    "budget state dir unusable (%s) — running in-memory", e)
                self._state_dir = None

    # ------------------------------------------------------------------
    # Envelope admin
    # ------------------------------------------------------------------

    def set_envelope(self, path: str, cap_usd: float,
                     period: str = "month") -> Envelope:
        """Create or update. Updating the cap keeps current spend —
        raising a cap mid-period unblocks immediately; lowering below
        spent blocks further work until rollover (that's the point)."""
        if period not in PERIOD_SECONDS:
            raise ValueError(f"unknown period {period!r}; "
                             f"one of {sorted(PERIOD_SECONDS)}")
        path = "/".join(p for p in path.split("/") if p)
        if not path:
            raise ValueError("empty envelope path")
        if cap_usd < 0:
            raise ValueError("cap_usd must be >= 0")
        with self._lock:
            env = self._envelopes.get(path)
            if env is None:
                env = Envelope(path=path, cap_usd=float(cap_usd),
                               period=period, window_start=self._clock())
                self._envelopes[path] = env
            else:
                env.cap_usd = float(cap_usd)
                env.period = period
            self._save_state()
            self._journal({"event": "envelope_set", "path": path,
                           "cap_usd": cap_usd, "period": period})
            return env

    def envelopes(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._rollover_all()
            out = []
            for env in sorted(self._envelopes.values(),
                              key=lambda e: e.path):
                d = env.to_dict()
                d["reserved_usd"] = self._reserved_against(env.path)
                d["remaining_usd"] = max(
                    0.0, env.cap_usd - env.spent_usd - d["reserved_usd"])
                out.append(d)
            return out

    # ------------------------------------------------------------------
    # The gate: reserve → settle | release
    # ------------------------------------------------------------------

    def reserve(self, job_id: str, path: str, amount_usd: float,
                meta: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Hold `amount_usd` against every envelope on `path`.
        Returns None on success, or an honest human-readable refusal
        reason (fail-closed). Idempotent: re-reserving an existing
        job_id succeeds without double-holding."""
        path = "/".join(p for p in path.split("/") if p) or "default"
        with self._lock:
            if job_id in self._reservations or job_id in self._settled_job_ids:
                return None
            self._expire_stale()
            self._rollover_all()
            matched = [self._envelopes[p] for p in _prefixes(path)
                       if p in self._envelopes]
            if not matched:
                if self.require_envelope:
                    return (f"budget_ledger: no envelope covers "
                            f"'{path}' and require_envelope is on")
                # Ungoverned path — allowed, but journalled so it shows
                # up in reports instead of vanishing.
                self._reservations[job_id] = _Reservation(
                    job_id=job_id, path=path,
                    amount_usd=float(amount_usd),
                    created_at=self._clock(), meta=dict(meta or {}))
                self._journal({"event": "reserve", "job_id": job_id,
                               "path": path, "amount_usd": amount_usd,
                               "governed": False, **(meta or {})})
                return None
            for env in matched:
                headroom = (env.cap_usd - env.spent_usd
                            - self._reserved_against(env.path))
                if amount_usd > headroom:
                    return (
                        f"budget_ledger: envelope '{env.path}' "
                        f"({env.period} cap ${env.cap_usd:.2f}) has "
                        f"${max(0.0, headroom):.4f} headroom; job needs "
                        f"${amount_usd:.4f}. Raise the cap or wait for "
                        f"rollover."
                    )
            self._reservations[job_id] = _Reservation(
                job_id=job_id, path=path, amount_usd=float(amount_usd),
                created_at=self._clock(), meta=dict(meta or {}))
            self._journal({"event": "reserve", "job_id": job_id,
                           "path": path, "amount_usd": amount_usd,
                           "governed": True, **(meta or {})})
            return None

    def settle(self, job_id: str, actual_usd: float,
               meta: Optional[Dict[str, Any]] = None) -> None:
        """Convert the hold into recorded spend at the ACTUAL cleared
        price (≤ the reserved ceiling in the normal case; a higher
        actual is still recorded — the ledger never lies about money
        already spent). Idempotent."""
        with self._lock:
            if job_id in self._settled_job_ids:
                return
            res = self._reservations.pop(job_id, None)
            if res is None:
                return
            self._settled_job_ids.add(job_id)
            self._rollover_all()
            for p in _prefixes(res.path):
                env = self._envelopes.get(p)
                if env is not None:
                    env.spent_usd += float(actual_usd)
            self._save_state()
            self._journal({"event": "settle", "job_id": job_id,
                           "path": res.path,
                           "amount_usd": float(actual_usd),
                           **res.meta, **(meta or {})})

    def release(self, job_id: str) -> None:
        """Drop the hold (failed/cancelled job). Idempotent."""
        with self._lock:
            res = self._reservations.pop(job_id, None)
            if res is None:
                return
            self._journal({"event": "release", "job_id": job_id,
                           "path": res.path,
                           "amount_usd": res.amount_usd})

    # ------------------------------------------------------------------
    # Chargeback / observability
    # ------------------------------------------------------------------

    def report(self, *, prefix: str = "",
               since_unix: float = 0.0) -> Dict[str, Any]:
        """Aggregated settled spend grouped by envelope path — the
        chargeback query. Includes ungoverned spend explicitly so
        nothing hides off-book."""
        with self._lock:
            by_path: Dict[str, Dict[str, Any]] = {}
            ungoverned = 0.0
            total = 0.0
            for ev in self._journal_mem:
                if ev.get("event") != "settle":
                    continue
                if ev.get("ts", 0.0) < since_unix:
                    continue
                path = str(ev.get("path", ""))
                if prefix and not path.startswith(prefix):
                    continue
                amt = float(ev.get("amount_usd", 0.0))
                total += amt
                slot = by_path.setdefault(
                    path, {"spend_usd": 0.0, "jobs": 0})
                slot["spend_usd"] += amt
                slot["jobs"] += 1
                if not any(p in self._envelopes for p in _prefixes(path)):
                    ungoverned += amt
            return {
                "total_spend_usd": total,
                "ungoverned_spend_usd": ungoverned,
                "by_envelope": dict(sorted(by_path.items())),
                "envelopes": self.envelopes(),
            }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reserved_against(self, env_path: str) -> float:
        return sum(
            r.amount_usd for r in self._reservations.values()
            if env_path in _prefixes(r.path)
        )

    def _rollover_all(self) -> None:
        now = self._clock()
        changed = False
        for env in self._envelopes.values():
            span = PERIOD_SECONDS[env.period]
            if span is None:
                continue
            if now - env.window_start >= span:
                # Advance in whole windows so the anchor stays stable.
                windows = int((now - env.window_start) // span)
                env.window_start += windows * span
                if env.spent_usd:
                    self._journal({"event": "rollover", "path": env.path,
                                   "spent_was_usd": env.spent_usd})
                env.spent_usd = 0.0
                changed = True
        if changed:
            self._save_state()

    def _expire_stale(self) -> None:
        now = self._clock()
        for jid in [j for j, r in self._reservations.items()
                    if now - r.created_at > RESERVATION_TTL_S]:
            logger.warning("budget reservation %s expired unsettled", jid)
            res = self._reservations.pop(jid)
            self._journal({"event": "expire", "job_id": jid,
                           "path": res.path,
                           "amount_usd": res.amount_usd})

    def _journal(self, ev: Dict[str, Any]) -> None:
        ev = {"ts": self._clock(), **ev}
        self._journal_mem.append(ev)
        if self._state_dir is not None:
            try:
                with open(self._state_dir / "journal.jsonl", "a",
                          encoding="utf-8") as f:
                    f.write(json.dumps(ev, sort_keys=True) + "\n")
            except Exception as e:
                logger.warning("budget journal write failed: %s", e)

    def _save_state(self) -> None:
        if self._state_dir is None:
            return
        try:
            body = {"envelopes": [e.to_dict()
                                  for e in self._envelopes.values()]}
            tmp = self._state_dir / "state.json.tmp"
            tmp.write_text(json.dumps(body, indent=2), encoding="utf-8")
            tmp.replace(self._state_dir / "state.json")
        except Exception as e:
            logger.warning("budget state write failed: %s", e)

    def _load_state(self) -> None:
        f = self._state_dir / "state.json"
        if not f.exists():
            return
        try:
            body = json.loads(f.read_text(encoding="utf-8"))
            for d in body.get("envelopes", []):
                self._envelopes[d["path"]] = Envelope(
                    path=d["path"], cap_usd=float(d["cap_usd"]),
                    period=d.get("period", "month"),
                    spent_usd=float(d.get("spent_usd", 0.0)),
                    window_start=float(d.get("window_start", 0.0)),
                )
        except Exception as e:
            logger.warning("budget state unreadable (%s) — starting "
                           "with no envelopes", e)
        j = self._state_dir / "journal.jsonl"
        if j.exists():
            try:
                for line in j.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        self._journal_mem.append(json.loads(line))
            except Exception as e:
                logger.warning("budget journal unreadable: %s", e)
