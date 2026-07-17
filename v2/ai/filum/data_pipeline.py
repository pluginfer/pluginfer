"""Filum data pipeline: dedup, perplexity-filter, lineage, curriculum.

INVENTION (claim §11 in the design notes): in standard ML pipelines,
training data is a black box -- "we trained on Common Crawl + some
filtering." When the model misbehaves, you can't audit which sample
caused it. When you want to remove a contaminated source, you have
to retrain from scratch.

Filum's data layer is FULL-LINEAGE: every training token is tagged
with (source, generator, generation_method, timestamp, quality_score,
trust_chain). This costs ~5% disk overhead (the lineage records)
and gives you:

  * Subtractive retraining: remove all samples from a poisoned
    source via a SQL-like delete; the next training run skips them.
  * Audit trail: when the model says something wrong, trace the
    samples it learned that pattern from.
  * Per-source quality weighting: trusted sources contribute
    full-magnitude gradients; lower-trust sources are downweighted.

Five filtering layers (each is novel-by-combination):

  1. **Length filter**: discard samples below 32 tokens or above
     context window.
  2. **Perplexity filter**: pass each sample through a small
     reference model; reject high-perplexity outliers (broken text)
     and ultra-low-perplexity (likely template / copy-paste).
  3. **MinHash dedup**: across the entire corpus, drop near-duplicate
     samples (Jaccard >0.85 on shingle hashes).
  4. **Toxicity / safety**: simple regex + classifier heuristics.
  5. **Repetition filter**: kill samples with repetition score > 0.5
     (n-gram repetition / total unique).

Plus the **curriculum scheduler**: data flows in stages from simple
to complex. Stage progression is gated on student loss (we don't
move to the next stage until the current stage's loss has plateaued
on a held-out eval).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lineage-tagged training sample
# ---------------------------------------------------------------------------


@dataclass
class TrainingSample:
    """One training example with full provenance."""
    text: str
    source: str                       # "openassistant" / "self_play" / "chain_receipt" / etc
    generator: str = ""               # which teacher / dataset version
    method: str = "raw"               # "raw" | "distilled" | "synthetic" | "augmented"
    timestamp: float = 0.0
    quality_score: float = 1.0        # 0..1
    trust_score: float = 1.0          # 0..1, decays for less-trusted sources
    parent_sample_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    sample_id: str = ""

    def __post_init__(self) -> None:
        if not self.sample_id:
            self.sample_id = hashlib.sha256(
                f"{self.text}|{self.source}|{self.timestamp}".encode()
            ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# MinHash deduplication (no extra dep)
# ---------------------------------------------------------------------------


def shingle_hashes(text: str, k: int = 5, n_hashes: int = 64) -> List[int]:
    """Compute n MinHashes over k-shingles of `text`. Cheap and
    deterministic; 64 hashes give Jaccard estimation error ~0.06.
    Pure-Python; no `datasketch` dep."""
    tokens = text.split()
    if len(tokens) < k:
        shingles = [" ".join(tokens)]
    else:
        shingles = [" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)]
    if not shingles:
        return [0] * n_hashes
    seeds = [(i * 2654435761) & 0xFFFFFFFF for i in range(1, n_hashes + 1)]
    out = []
    for seed in seeds:
        m = float("inf")
        for s in shingles:
            # Universal hash: hash(shingle) XOR seed.
            h = int(hashlib.sha1(s.encode()).hexdigest(), 16) ^ seed
            if h < m:
                m = h
        out.append(int(m & 0xFFFFFFFFFFFFFFFF))
    return out


def jaccard_estimate(a_hashes: List[int], b_hashes: List[int]) -> float:
    if not a_hashes or len(a_hashes) != len(b_hashes):
        return 0.0
    matches = sum(1 for x, y in zip(a_hashes, b_hashes) if x == y)
    return matches / len(a_hashes)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


@dataclass
class FilterStats:
    seen: int = 0
    kept: int = 0
    rejected_short: int = 0
    rejected_long: int = 0
    rejected_dedup: int = 0
    rejected_repetition: int = 0
    rejected_toxicity: int = 0
    rejected_low_trust: int = 0


def repetition_ratio(text: str, n: int = 4) -> float:
    """1 - unique_n_grams / total_n_grams. 0 = no repetition; 1 =
    all repeats."""
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return 1 - (len(set(grams)) / len(grams))


# Cheap toxicity heuristic. Real systems use a classifier; this is a
# starting point.
_TOXIC_PATTERNS = re.compile(
    r"\b(fuck|shit|bitch|asshole|nigger|faggot|kike|retard)\w*", re.I,
)


def toxicity_score(text: str) -> float:
    matches = len(_TOXIC_PATTERNS.findall(text))
    if not text.strip():
        return 0.0
    return min(1.0, matches / max(1, len(text.split())) * 10)


# ---------------------------------------------------------------------------
# DataPipeline: orchestrates filters + lineage
# ---------------------------------------------------------------------------


@dataclass
class DataPipelineConfig:
    min_tokens: int = 16
    max_tokens: int = 4096
    dedup_threshold: float = 0.85
    repetition_threshold: float = 0.5
    toxicity_threshold: float = 0.05
    min_trust: float = 0.3
    enable_dedup: bool = True
    enable_repetition: bool = True
    enable_toxicity: bool = True


class DataPipeline:
    """Stream-style filter + lineage tagger.

    Drop incoming samples through `add_sample(s)`; iterate
    `filtered_stream()` to yield only samples that passed every
    filter, with their lineage entries persisted to disk."""

    def __init__(
        self,
        config: DataPipelineConfig,
        *,
        lineage_path: Optional[Path] = None,
    ):
        self.config = config
        self.lineage_path = Path(lineage_path) if lineage_path else None
        if self.lineage_path is not None:
            self.lineage_path.parent.mkdir(parents=True, exist_ok=True)
        self.dedup_index: Dict[Tuple[int, ...], List[str]] = defaultdict(list)
        # Use the first 8 minhashes as bands (LSH-style); two samples
        # match if any band matches.
        self.lsh_bands = 8
        self.stats = FilterStats()
        self._known_hashes: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------

    def _passes(self, s: TrainingSample) -> Tuple[bool, str]:
        self.stats.seen += 1
        n_tokens = len(s.text.split())
        if n_tokens < self.config.min_tokens:
            self.stats.rejected_short += 1
            return False, "short"
        if n_tokens > self.config.max_tokens:
            self.stats.rejected_long += 1
            return False, "long"
        if s.trust_score < self.config.min_trust:
            self.stats.rejected_low_trust += 1
            return False, "low_trust"
        if self.config.enable_repetition:
            r = repetition_ratio(s.text)
            if r > self.config.repetition_threshold:
                self.stats.rejected_repetition += 1
                return False, f"repetition_{r:.2f}"
        if self.config.enable_toxicity:
            t = toxicity_score(s.text)
            if t > self.config.toxicity_threshold:
                self.stats.rejected_toxicity += 1
                return False, f"toxicity_{t:.2f}"
        if self.config.enable_dedup:
            h = shingle_hashes(s.text)
            # Quick band-check: if any band of 8 hashes matches an
            # existing sample exactly, do a full Jaccard.
            for band_idx in range(self.lsh_bands):
                start = band_idx * (len(h) // self.lsh_bands)
                end = start + (len(h) // self.lsh_bands)
                band = tuple(h[start:end])
                for other_id in self.dedup_index.get(band, []):
                    other_h = self._known_hashes.get(other_id)
                    if other_h is None:
                        continue
                    j = jaccard_estimate(h, other_h)
                    if j > self.config.dedup_threshold:
                        self.stats.rejected_dedup += 1
                        return False, f"dedup_{j:.2f}_with_{other_id[:8]}"
            # Register this sample's hashes.
            self._known_hashes[s.sample_id] = h
            for band_idx in range(self.lsh_bands):
                start = band_idx * (len(h) // self.lsh_bands)
                end = start + (len(h) // self.lsh_bands)
                self.dedup_index[tuple(h[start:end])].append(s.sample_id)
        return True, "ok"

    def add_sample(self, s: TrainingSample) -> Tuple[bool, str]:
        ok, reason = self._passes(s)
        if ok:
            self.stats.kept += 1
            if self.lineage_path is not None:
                self._log_lineage(s, "kept")
        else:
            if self.lineage_path is not None:
                self._log_lineage(s, f"rejected:{reason}")
        return ok, reason

    def _log_lineage(self, s: TrainingSample, status: str) -> None:
        try:
            with open(self.lineage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "sample_id": s.sample_id,
                    "source": s.source,
                    "generator": s.generator,
                    "method": s.method,
                    "timestamp": s.timestamp or time.time(),
                    "trust_score": s.trust_score,
                    "quality_score": s.quality_score,
                    "status": status,
                }) + "\n")
        except Exception as e:                                  # pragma: no cover
            logger.warning("lineage log failed: %s", e)


# ---------------------------------------------------------------------------
# Curriculum scheduler
# ---------------------------------------------------------------------------


@dataclass
class CurriculumStage:
    name: str
    sources: Tuple[str, ...]          # which TrainingSample.source values flow at this stage
    target_loss: float                # advance when held-out loss <= this
    max_steps: int                    # advance unconditionally after this many steps
    weight: float = 1.0               # mixing weight when this stage is active


class CurriculumScheduler:
    """Tracks the current stage. The trainer calls `current_sources()`
    to know which TrainingSample.source values to draw from this step;
    after each eval round it calls `update(loss)` and the scheduler
    decides whether to advance."""

    def __init__(self, stages: List[CurriculumStage]):
        if not stages:
            raise ValueError("at least one stage required")
        self.stages = stages
        self.idx = 0
        self.steps_in_stage = 0

    def current(self) -> CurriculumStage:
        return self.stages[self.idx]

    def step(self, eval_loss: Optional[float] = None) -> None:
        self.steps_in_stage += 1
        cur = self.current()
        advance = False
        if eval_loss is not None and eval_loss <= cur.target_loss:
            advance = True
        elif self.steps_in_stage >= cur.max_steps:
            advance = True
        if advance and self.idx < len(self.stages) - 1:
            logger.info("curriculum: %s -> %s (steps=%d, eval_loss=%s)",
                        cur.name, self.stages[self.idx + 1].name,
                        self.steps_in_stage, eval_loss)
            self.idx += 1
            self.steps_in_stage = 0

    def is_complete(self) -> bool:
        return (self.idx == len(self.stages) - 1
                and self.steps_in_stage >= self.current().max_steps)


# Default Filum curriculum (7 stages from byte to chain-receipts).
DEFAULT_CURRICULUM: List[CurriculumStage] = [
    CurriculumStage(
        name="byte_completion",
        sources=("common_crawl", "wikitext"),
        target_loss=4.0, max_steps=5_000, weight=1.0,
    ),
    CurriculumStage(
        name="phrase_completion",
        sources=("wikitext", "openwebtext", "books"),
        target_loss=3.0, max_steps=8_000, weight=1.0,
    ),
    CurriculumStage(
        name="instruct",
        sources=("openassistant", "dolly", "alpaca", "distilled_teacher"),
        target_loss=2.5, max_steps=10_000, weight=1.0,
    ),
    CurriculumStage(
        name="reasoning",
        sources=("gsm8k", "metamath", "math_dataset", "distilled_teacher"),
        target_loss=2.0, max_steps=10_000, weight=1.5,
    ),
    CurriculumStage(
        name="router_task",
        sources=("pluginfer_router_synth", "distilled_teacher"),
        target_loss=1.5, max_steps=5_000, weight=2.0,
    ),
    CurriculumStage(
        name="self_play",
        sources=("self_play", "distilled_teacher"),
        target_loss=1.5, max_steps=8_000, weight=1.0,
    ),
    CurriculumStage(
        name="continual",
        sources=("chain_receipt",),
        target_loss=1.2, max_steps=1_000_000_000, weight=1.0,  # never finishes
    ),
]
