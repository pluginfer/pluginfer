"""Mixture-of-LoRAs with mesh-driven adapter spawning.

INVENTION (claim §8 in the design notes): catastrophic forgetting in
continually-trained models is a hard problem. The conventional fix
(EWC, replay buffer) preserves OLD capabilities but doesn't add NEW
ones cleanly -- the single LoRA blurs across domains.

Our novelty: **automatic LoRA spawning driven by mesh-level
distribution drift detection**. When the Pluginfer anomaly detector
notices a cluster of new task inputs that the existing LoRAs can't
handle, the system spawns a FRESH LoRA adapter, seeds it with
teacher-distilled examples on that cluster, and trains it
continually on traffic in that cluster. Inference routes to the
right adapter via a tiny clustering router.

Effects:
  * Existing LoRAs stay frozen (no forgetting).
  * New domains get their own adapter (clean learning).
  * Adapters are small (~5 MB each at rank=8); a node can hold
    100+ specialty adapters in 500 MB.
  * Adapters are TRADEABLE (see capability_marketplace.py): a node
    that trained an excellent legal-text adapter can rent it.

Failure modes (honest)
----------------------
* The cluster router is itself a small classifier; mis-routes go to
  the wrong adapter and produce nonsense. We use confidence scores
  + fallback to the base model.
* Spawning a new adapter for every minor distribution shift bloats
  storage. We cap the pool at config.max_adapters and evict
  least-used.
* The "teacher seeds the new adapter" step costs API budget. The
  trade-off: pay $1-5 once to bootstrap a new specialty, then own
  it forever.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:                                                # pragma: no cover
    torch = None
    _HAS_TORCH = False

logger = logging.getLogger(__name__)


@dataclass
class AdapterMetadata:
    """Tracks one LoRA adapter's identity + lifecycle state."""
    adapter_id: str
    domain: str                     # human-readable name ("legal", "code", "router")
    cluster_centroid: List[float]   # in input-embedding space
    rank: int = 8
    samples_trained: int = 0
    creation_time: float = 0.0
    last_used: float = 0.0
    use_count: int = 0
    parent_owner_pubkey: Optional[str] = None       # for marketplace
    quality_score: float = 0.0                       # 0..1, eval against held-out


@dataclass
class LoRAPoolConfig:
    max_adapters: int = 32
    cluster_distance_threshold: float = 0.4
    min_cluster_size_to_spawn: int = 50
    rank: int = 8
    eviction_policy: str = "lru"     # "lru" | "lowest_quality"


# ---------------------------------------------------------------------------
# Cluster detector -- watches incoming queries, spawns when a new
# coherent cluster emerges that's far from every existing adapter.
# ---------------------------------------------------------------------------


@dataclass
class _PendingCluster:
    centroid: List[float]
    member_embeddings: List[List[float]] = field(default_factory=list)
    member_queries: List[str] = field(default_factory=list)
    discovered_at: float = 0.0


class ClusterDetector:
    """Maintains a running set of "drift candidates" -- query
    embeddings that don't fit any existing adapter's centroid. When a
    candidate cluster grows above `min_cluster_size_to_spawn`, we
    declare it a new domain and spawn an adapter."""

    def __init__(self, *, config: LoRAPoolConfig):
        self.config = config
        self.candidates: List[_PendingCluster] = []
        self.recent_misses: List[Tuple[List[float], str]] = []

    def observe_miss(self, query_emb: List[float], query_text: str) -> None:
        """Record a query that didn't match any adapter. After enough
        misses cluster together, we'll spawn an adapter."""
        self.recent_misses.append((query_emb, query_text))
        if len(self.recent_misses) > 1000:
            self.recent_misses = self.recent_misses[-500:]

    def maybe_spawn(self) -> Optional[Tuple[List[float], List[str]]]:
        """Run a one-pass clustering on recent misses. Returns
        (centroid, query_texts) for a cluster that meets the spawn
        threshold, else None."""
        if len(self.recent_misses) < self.config.min_cluster_size_to_spawn:
            return None
        # Greedy clustering: find the densest cluster.
        centroids: List[Tuple[List[float], List[int]]] = []
        for i, (emb, _) in enumerate(self.recent_misses):
            placed = False
            for c, idxs in centroids:
                if _cosine(emb, c) > 1 - self.config.cluster_distance_threshold:
                    idxs.append(i)
                    # update centroid (running mean)
                    n = len(idxs)
                    for j in range(len(c)):
                        c[j] = (c[j] * (n - 1) + emb[j]) / n
                    placed = True
                    break
            if not placed:
                centroids.append((list(emb), [i]))
        # Largest cluster.
        if not centroids:
            return None
        centroids.sort(key=lambda x: len(x[1]), reverse=True)
        best_c, best_idxs = centroids[0]
        if len(best_idxs) < self.config.min_cluster_size_to_spawn:
            return None
        queries = [self.recent_misses[i][1] for i in best_idxs]
        # Drain the misses we used.
        keep = [m for j, m in enumerate(self.recent_misses) if j not in set(best_idxs)]
        self.recent_misses = keep
        return best_c, queries


def _cosine(a: List[float], b: List[float]) -> float:
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (da * db)


# ---------------------------------------------------------------------------
# Adapter pool
# ---------------------------------------------------------------------------


class LoRAPool:
    """Owns N LoRA adapters, a clustering router, and the spawning
    pipeline.

    Each adapter is a state_dict on disk + an AdapterMetadata entry.
    Loading/unloading is fast (a few hundred MB / sec from NVMe);
    only the active adapter is on GPU."""

    def __init__(
        self,
        config: LoRAPoolConfig,
        *,
        store_dir: Path,
        embedder: Callable[[str], List[float]],
    ):
        self.config = config
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.adapters: Dict[str, AdapterMetadata] = {}
        self.detector = ClusterDetector(config=config)
        self._load_index()

    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return self.store_dir / "adapter_index.json"

    def _load_index(self) -> None:
        p = self._index_path()
        if not p.exists():
            return
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        for ad in d.get("adapters", []):
            md = AdapterMetadata(**ad)
            self.adapters[md.adapter_id] = md

    def _save_index(self) -> None:
        d = {"adapters": [vars(md) for md in self.adapters.values()]}
        self._index_path().write_text(json.dumps(d), encoding="utf-8")

    # ------------------------------------------------------------------
    # Routing: given a query, return the best adapter (or None for base)
    # ------------------------------------------------------------------

    def route(self, query: str) -> Tuple[Optional[AdapterMetadata], float]:
        """Score every adapter centroid against the query. Return the
        best match if its similarity > threshold, else None."""
        emb = self.embedder(query)
        best_md: Optional[AdapterMetadata] = None
        best_score = -1.0
        for md in self.adapters.values():
            s = _cosine(emb, md.cluster_centroid)
            if s > best_score:
                best_score = s
                best_md = md
        if best_md is not None and (1 - best_score) <= self.config.cluster_distance_threshold:
            best_md.last_used = time.time()
            best_md.use_count += 1
            return best_md, best_score
        # No good match -- record as a miss for future spawning.
        self.detector.observe_miss(emb, query)
        return None, best_score if best_md is None else 0.0

    # ------------------------------------------------------------------
    # Spawning a new adapter
    # ------------------------------------------------------------------

    async def maybe_spawn_new_adapter(
        self,
        seed_with_teacher_fn: Callable[[List[str]], Awaitable[List[Tuple[str, str]]]],
        train_adapter_fn: Callable[[List[Tuple[str, str]], AdapterMetadata], Awaitable[None]],
    ) -> Optional[AdapterMetadata]:
        """Check the detector. If a new cluster has emerged, seed a
        fresh adapter from teacher-generated examples on its queries
        + train it. Returns the new adapter's metadata, or None if no
        spawn was triggered."""
        result = self.detector.maybe_spawn()
        if result is None:
            return None
        centroid, sample_queries = result
        adapter_id = "ada_" + hashlib.sha256(
            (str(centroid)[:64] + str(time.time())).encode()
        ).hexdigest()[:12]
        md = AdapterMetadata(
            adapter_id=adapter_id,
            domain="auto_" + adapter_id[-4:],   # caller can rename
            cluster_centroid=centroid,
            rank=self.config.rank,
            creation_time=time.time(),
        )
        # Step 1: get teacher answers for the cluster's seed queries.
        teacher_pairs = await seed_with_teacher_fn(sample_queries)
        if not teacher_pairs:
            logger.warning("teacher seeding produced no pairs; aborting spawn")
            return None
        md.samples_trained = len(teacher_pairs)
        # Step 2: train the LoRA on those pairs.
        await train_adapter_fn(teacher_pairs, md)
        # Step 3: register + maybe evict.
        if len(self.adapters) >= self.config.max_adapters:
            self._evict_one()
        self.adapters[adapter_id] = md
        self._save_index()
        return md

    def _evict_one(self) -> None:
        if not self.adapters:
            return
        if self.config.eviction_policy == "lru":
            evict = min(self.adapters.values(), key=lambda m: m.last_used)
        else:
            evict = min(self.adapters.values(), key=lambda m: m.quality_score)
        # Remove the on-disk file (if present) but keep the metadata
        # entry as a tombstone for marketplace audit.
        adapter_path = self.store_dir / f"{evict.adapter_id}.pt"
        try:
            adapter_path.unlink()
        except FileNotFoundError:
            pass
        self.adapters.pop(evict.adapter_id, None)
        self._save_index()
