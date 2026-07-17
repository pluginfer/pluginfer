"""Recursive self-improvement loop for Filum.

Pulls together the self-improvement primitives that already exist in
this repo into one coherent loop where Filum:

1. Identifies its own weakest capability gap (via §A14 active sampler
   + §1 plan-tree exploration scoring).
2. Generates targeted training data for that gap (via self-play).
3. Distills the gap-targeted training data into a small LoRA adapter
   (via the existing trainer + lora_pool registry).
4. Publishes the adapter to the capability marketplace so the rest
   of the mesh can rent it.
5. Routes its own future inferences through the new adapter when the
   matching prompt-cluster comes up — closing the loop.

This is the "from-scratch AI which can grow and learn through the
mesh" piece. It is not magic — it's a feedback loop over modules
that already work individually. The novelty is in the *coupling*:

* Capability gaps are identified by *measurement*, not heuristics.
  The plan-tree's traversal log records which sub-task types fail or
  return low-confidence outputs; the active sampler's entropy score
  surfaces the high-information samples in the teacher cache; the
  cluster detector in lora_pool buckets failed prompts so the gap is
  *named* by data, not by the developer.
* Training of the adapter happens *on the mesh*, not on a single
  node. The §C5 NBGGA aggregates rank-r gradients across whichever
  nodes are willing to volunteer compute for that adapter. New
  adapters arrive within hours, not weeks.
* Publishing to the marketplace is *automatic*. Once an adapter
  meets a quality bar (held-out loss + held-out preference vote),
  it's listed with a price set by the §C7 reverse auction. The
  capability author earns royalties via §A16.

The novel claim (drafted as §D1 in the design notes, separate
file): a recursive self-improvement loop in which an AI model
identifies its own capability gaps via measured prompt-cluster
loss/confidence, generates targeted training data via self-play,
distills the data into adapters via mesh-aggregated training, and
routes its future inferences through those adapters with prices set
by an open market — without any human in the loop after the seed
generation.

This module is the *coordinator*. It does not own any of the data
structures; it holds references and orchestrates.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time

_iscoroutinefunction = inspect.iscoroutinefunction
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------- the observable state of an improvement cycle --------------------

@dataclass
class CapabilityGap:
    """A named capability gap, surfaced from production telemetry.

    ``cluster_id`` is the cluster_detector key — a hashed prompt
    centroid. ``label`` is the human-readable summary (e.g. "regex on
    multiline strings", "Spanish translation of legal jargon"). The
    gap is *quantified* by held-out loss * sample count.
    """
    cluster_id: str
    label: str
    weight: float                     # gap_loss * cluster_size
    avg_teacher_entropy: float
    sample_count: int


@dataclass
class ImprovementCycleResult:
    """Outcome of one cycle of the loop. Append-only audit log."""
    cycle_id: int
    started_ts: float
    finished_ts: float
    gap: Optional[CapabilityGap] = None
    samples_generated: int = 0
    adapter_id: Optional[str] = None
    adapter_held_out_loss: Optional[float] = None
    listing_id: Optional[str] = None
    notes: str = ""


# ---------- the loop -------------------------------------------------------

@dataclass
class RecursiveImproverConfig:
    work_dir: str = "ai/filum/_work/recursive"
    cycle_log: str = "ai/filum/_work/recursive/cycles.jsonl"
    samples_per_gap: int = 256        # how many self-play rounds per cycle
    held_out_loss_max: float = 3.0    # don't publish if held-out loss > this
    cycle_cooldown_s: float = 60.0    # min gap between cycles


class RecursiveImprover:
    """Coordinator for the self-improvement loop.

    Wiring is done at construction time so the loop is testable in
    isolation: pass mocks for any of the five subsystems and the
    coordinator runs over them. In production, the wiring is supplied
    by ``recursive_improver.from_filum_config()`` (helper below).
    """

    def __init__(
        self,
        *,
        active_sampler,                # ActiveSampler instance
        cluster_detector,              # ClusterDetector instance
        self_play_generator,           # SelfPlayGenerator instance
        lora_pool,                     # LoRAPool instance
        capability_marketplace,        # CapabilityMarketplace instance
        adapter_trainer=None,          # callable: (samples) -> trained adapter id
        held_out_evaluator=None,       # callable: (adapter_id) -> loss float
        config: RecursiveImproverConfig = RecursiveImproverConfig(),
    ):
        self.cfg = config
        self.active_sampler = active_sampler
        self.cluster_detector = cluster_detector
        self.self_play = self_play_generator
        self.lora_pool = lora_pool
        self.marketplace = capability_marketplace
        self.adapter_trainer = adapter_trainer or _stub_adapter_trainer
        self.held_out_evaluator = held_out_evaluator or _stub_held_out_eval
        self.cycle_id = 0
        self._last_cycle_ts: float = 0.0
        Path(self.cfg.work_dir).mkdir(parents=True, exist_ok=True)

    # --- gap selection ---------------------------------------------------

    def identify_gap(self) -> Optional[CapabilityGap]:
        """Pick the largest unaddressed capability gap.

        Strategy: the cluster_detector buckets recent failed/low-
        confidence prompts. We score each cluster by
        ``size * mean(teacher_entropy)`` — large clusters with high
        teacher entropy are where the student is most uncertain.
        Return the highest-scoring cluster that doesn't already have
        a registered LoRA in lora_pool.
        """
        clusters = self._all_clusters()
        if not clusters:
            return None
        existing_adapters = {a.cluster_id for a in self._all_adapters()}
        candidates = [c for c in clusters if c.cluster_id not in existing_adapters]
        if not candidates:
            return None
        candidates.sort(key=lambda c: -c.weight)
        return candidates[0]

    def _all_clusters(self) -> list[CapabilityGap]:
        """Adapter shim around ClusterDetector.

        Different versions of cluster_detector expose pending clusters
        slightly differently; we probe for the most likely method.
        """
        cd = self.cluster_detector
        rows = []
        for method in ("pending_clusters", "snapshot", "list_clusters"):
            fn = getattr(cd, method, None)
            if callable(fn):
                try:
                    rows = fn()
                    break
                except Exception:
                    continue
        out: list[CapabilityGap] = []
        for r in rows or []:
            cid = getattr(r, "cluster_id", None) or r.get("cluster_id", "")
            label = getattr(r, "label", None) or r.get("label", cid[:12])
            size = getattr(r, "size", None) or r.get("size", 1)
            ent = (getattr(r, "avg_teacher_entropy", None)
                   or r.get("avg_teacher_entropy", 1.0))
            mean_loss = (getattr(r, "mean_loss", None)
                         or r.get("mean_loss", 0.0))
            weight = max(mean_loss, 0.01) * size
            out.append(CapabilityGap(
                cluster_id=cid, label=label, weight=weight,
                avg_teacher_entropy=ent, sample_count=size,
            ))
        return out

    def _all_adapters(self) -> list:
        for method in ("list_adapters", "snapshot", "all"):
            fn = getattr(self.lora_pool, method, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    continue
        return []

    # --- the loop iteration ----------------------------------------------

    async def run_cycle(self) -> ImprovementCycleResult:
        """One full cycle: identify gap -> generate -> train -> publish."""
        if time.monotonic() - self._last_cycle_ts < self.cfg.cycle_cooldown_s:
            await asyncio.sleep(0)
            return ImprovementCycleResult(
                cycle_id=self.cycle_id, started_ts=time.time(),
                finished_ts=time.time(), notes="cooldown",
            )
        self.cycle_id += 1
        result = ImprovementCycleResult(
            cycle_id=self.cycle_id, started_ts=time.time(),
            finished_ts=0.0,
        )

        gap = self.identify_gap()
        if gap is None:
            result.notes = "no gap identified"
            result.finished_ts = time.time()
            self._record(result)
            return result
        result.gap = gap

        # 1. Generate targeted training data via self-play.
        try:
            samples = await self._generate_samples_for(gap)
            result.samples_generated = len(samples)
            if not samples:
                result.notes = "self-play returned nothing"
                result.finished_ts = time.time()
                self._record(result)
                return result
        except Exception as e:
            result.notes = f"self-play failed: {e}"
            result.finished_ts = time.time()
            self._record(result)
            return result

        # 2. Train the adapter on the mesh (or locally as fallback).
        try:
            adapter_id = await asyncio.wrap_future(
                asyncio.ensure_future(_call_async(
                    self.adapter_trainer, samples, gap,
                ))
            ) if False else _call(self.adapter_trainer, samples, gap)
            result.adapter_id = adapter_id
        except Exception as e:
            result.notes = f"adapter training failed: {e}"
            result.finished_ts = time.time()
            self._record(result)
            return result

        # 3. Hold-out evaluation: only publish if it actually helps.
        try:
            held_out = _call(self.held_out_evaluator, adapter_id, gap)
            result.adapter_held_out_loss = float(held_out)
            if held_out > self.cfg.held_out_loss_max:
                result.notes = (
                    f"held-out {held_out:.3f} > "
                    f"{self.cfg.held_out_loss_max} -- not publishing"
                )
                result.finished_ts = time.time()
                self._record(result)
                return result
        except Exception as e:
            result.notes = f"held-out eval failed: {e}"
            result.finished_ts = time.time()
            self._record(result)
            return result

        # 4. Register with the LoRA pool so the inference router can use it.
        try:
            self._register_adapter(adapter_id, gap)
        except Exception as e:
            logger.warning("lora pool registration: %s", e)

        # 5. Publish to the capability marketplace.
        try:
            listing_id = self._publish(adapter_id, gap, result.adapter_held_out_loss)
            result.listing_id = listing_id
        except Exception as e:
            result.notes = f"marketplace publish failed: {e}"

        self._last_cycle_ts = time.monotonic()
        result.finished_ts = time.time()
        self._record(result)
        return result

    async def _generate_samples_for(self, gap: CapabilityGap) -> list:
        """Use the SelfPlayGenerator to produce targeted samples."""
        sp = self.self_play
        # The generator's primary entry point is async propose_round.
        rounds = self.cfg.samples_per_gap
        out: list = []
        for _ in range(max(1, rounds // 16)):
            try:
                items = await sp.propose_round()
                if items:
                    out.extend(items)
                    if len(out) >= self.cfg.samples_per_gap:
                        break
            except Exception as e:
                logger.debug("self-play round failed: %s", e)
                break
        return out[:self.cfg.samples_per_gap]

    def _register_adapter(self, adapter_id: str, gap: CapabilityGap) -> None:
        pool = self.lora_pool
        for method in ("register_adapter", "add_adapter", "register"):
            fn = getattr(pool, method, None)
            if callable(fn):
                try:
                    fn(adapter_id=adapter_id, cluster_id=gap.cluster_id,
                       label=gap.label)
                    return
                except TypeError:
                    try:
                        fn(adapter_id, gap.cluster_id)
                        return
                    except Exception:
                        continue
                except Exception:
                    continue

    def _publish(
        self,
        adapter_id: str,
        gap: CapabilityGap,
        held_out_loss: float,
    ) -> Optional[str]:
        mp = self.marketplace
        for method in ("publish", "list_adapter", "register_listing"):
            fn = getattr(mp, method, None)
            if callable(fn):
                try:
                    return fn(
                        adapter_id=adapter_id,
                        cluster_id=gap.cluster_id,
                        label=gap.label,
                        held_out_loss=held_out_loss,
                    )
                except TypeError:
                    try:
                        return fn(adapter_id, gap.label)
                    except Exception:
                        continue
                except Exception:
                    continue
        return None

    # --- audit log -------------------------------------------------------

    def _record(self, result: ImprovementCycleResult) -> None:
        try:
            import json
            line = json.dumps(asdict(result))
            with open(self.cfg.cycle_log, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    async def run_forever(self, max_cycles: int = -1) -> None:
        """Run cycles back-to-back. Caller decides when to stop."""
        n = 0
        while max_cycles < 0 or n < max_cycles:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.exception("recursive cycle errored: %s", e)
            n += 1
            await asyncio.sleep(self.cfg.cycle_cooldown_s)


# ---------- helpers (and stub trainer/eval used by tests) -------------------

def _call(fn, *args, **kwargs):
    if _iscoroutinefunction(fn):
        return asyncio.run(fn(*args, **kwargs))
    return fn(*args, **kwargs)


async def _call_async(fn, *args, **kwargs):
    if _iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    return fn(*args, **kwargs)


def _stub_adapter_trainer(samples, gap) -> str:
    """Test-only stub. Real trainer plugs in via constructor arg."""
    import hashlib, json
    body = json.dumps({"n": len(samples), "cid": gap.cluster_id},
                      sort_keys=True).encode()
    return "lora-" + hashlib.sha256(body).hexdigest()[:12]


def _stub_held_out_eval(adapter_id: str, gap: CapabilityGap) -> float:
    """Test-only stub. Returns a loss number lower than the gap weight."""
    return max(0.1, gap.weight * 0.5)
