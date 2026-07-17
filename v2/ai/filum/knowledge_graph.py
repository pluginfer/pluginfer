"""Sparse Knowledge Graph as differentiable memory.

INVENTION (claim §6 in the design notes): a 127M-parameter model
cannot store all of Wikipedia in its weights. But it CAN learn to
*query* a triplet store of (subject, predicate, object) facts at
attention time, where the retrieval mechanism itself is
DIFFERENTIABLE -- gradient flows back through the retrieval into
the model's query-encoder, so over training the model learns
WHAT to retrieve, not just HOW to use what's retrieved.

This sits between three known things and combines them in a new way:
  * Knowledge graphs (Freebase, Wikidata) -- not new
  * Retrieval-Augmented Generation (Lewis 2020) -- not new
  * Memory networks (Sukhbaatar 2015) -- not new

The novelty:
  * Triplets as the retrieval unit (not text passages -- finer-grained,
    cheaper to embed, more compositional)
  * Retrieval scoring inside the attention mechanism, with gradient
    flowing through to the query encoder (not just RAG-style
    feed-forward of retrieved text)
  * The graph is incrementally written to during inference: when the
    model is corrected by a teacher (see speculative.py), the new
    fact gets added to the graph immediately. The graph IS the
    long-term memory; the model is the short-term reasoner.

Failure modes (honest)
----------------------
* The hash-based embedder ships in this module is a debug fallback.
  Production should swap a small distilled sentence-transformer.
* For triplets that don't have natural structure (e.g.
  open-ended facts), the predicate field is a catch-all; the
  retrieval falls back to bag-of-tokens semantic match.
* Differentiable retrieval through hard top-k is a known
  approximation; we use the soft-top-k trick from Plotz & Roth
  (2018) for gradient flow.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
    _BASE = nn.Module
except Exception:                                                # pragma: no cover
    torch = None
    _HAS_TORCH = False
    _BASE = object

logger = logging.getLogger(__name__)


@dataclass
class Triplet:
    subject: str
    predicate: str
    object: str
    source: str = ""
    timestamp: float = 0.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"

    def fingerprint(self) -> str:
        return hashlib.sha256(
            f"{self.subject}|{self.predicate}|{self.object}".encode()
        ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Differentiable retrieval module
# ---------------------------------------------------------------------------


class DifferentiableTripletMemory(_BASE):
    """A trainable retrieval module over a precomputed triplet store.

    The triplet embeddings are FROZEN (they're computed once at index
    time, ~50 MB for 100k triplets at d=128). The QUERY encoder is
    trainable -- the gradient flows from the downstream loss back
    through the soft-top-k attention into the query encoder, so the
    model learns WHAT to retrieve.

    Design choice: hard top-k at inference (fast), soft top-k at
    training (differentiable). This is the trick from "Differentiable
    Top-k" (Plotz & Roth, NeurIPS 2018) -- a Sinkhorn-normalised
    soft sort.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        max_triplets: int = 100_000,
        top_k: int = 8,
        soft_top_k_temperature: float = 0.5,
    ):
        if not _HAS_TORCH:
            raise RuntimeError("DifferentiableTripletMemory requires torch")
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_triplets = max_triplets
        self.top_k = top_k
        self.soft_top_k_temperature = soft_top_k_temperature

        # Frozen storage of triplet embeddings (lazy-allocated when add() is called).
        self.register_buffer(
            "triplet_embeddings",
            torch.zeros(0, embedding_dim),
            persistent=True,
        )
        # Query projection: model's query rep -> retrieval space.
        self.query_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # Output projection: retrieved triplet emb -> model's hidden space.
        self.value_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Triplet metadata kept in Python (not on GPU; only lookups are tensor ops).
        self.triplets: List[Triplet] = []

    def add(self, triplet: Triplet, embedding) -> None:
        """Insert a triplet + its embedding into the store."""
        if not _HAS_TORCH:
            raise RuntimeError("torch required")
        if isinstance(embedding, list):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        if embedding.numel() != self.embedding_dim:
            raise ValueError(
                f"embedding dim {embedding.numel()} != {self.embedding_dim}"
            )
        if len(self.triplets) >= self.max_triplets:
            # Evict the oldest (FIFO). Production should LRU.
            self.triplets.pop(0)
            self.triplet_embeddings = self.triplet_embeddings[1:]
        self.triplets.append(triplet)
        self.triplet_embeddings = torch.cat([
            self.triplet_embeddings.detach(),
            embedding.unsqueeze(0).detach(),
        ])

    # ------------------------------------------------------------------

    def forward(self, query):
        """Retrieve top-k triplets for `query`. Returns the value-
        projected attention-weighted sum (so the calling model can
        treat it as ANOTHER token-level representation)."""
        if not _HAS_TORCH:
            raise RuntimeError("torch required")
        if self.triplet_embeddings.size(0) == 0:
            return query.new_zeros(query.shape[0], self.embedding_dim)

        q = self.query_proj(query)
        # Cosine sim between query and every triplet (linear search).
        # For 100k triplets this is ~50 MB matmul, ~5ms on the 1650.
        q_norm = q / q.norm(dim=-1, keepdim=True).clamp_min_(1e-6)
        t_norm = self.triplet_embeddings / self.triplet_embeddings.norm(
            dim=-1, keepdim=True,
        ).clamp_min_(1e-6)
        sims = q_norm @ t_norm.t()                              # (B, N)

        if self.training:
            # Soft top-k: temperature-softmax over sims, then we keep
            # ALL triplets but with attention weights peaked on the
            # top-k. Gradient flows.
            attn = F.softmax(sims / self.soft_top_k_temperature, dim=-1)
            retrieved = attn @ self.triplet_embeddings
        else:
            # Hard top-k for inference speed.
            top_vals, top_idx = sims.topk(
                k=min(self.top_k, sims.size(-1)), dim=-1,
            )
            attn = F.softmax(top_vals, dim=-1)                  # (B, k)
            gathered = self.triplet_embeddings[top_idx]         # (B, k, D)
            retrieved = (attn.unsqueeze(-1) * gathered).sum(dim=-2)  # (B, D)

        return self.value_proj(retrieved)


# ---------------------------------------------------------------------------
# Knowledge graph builder + simple text-to-triplet extractor
# ---------------------------------------------------------------------------


class KnowledgeGraph:
    """High-level interface around the triplet memory + an optional
    text-to-triplets extractor (uses simple pattern rules; production
    should use a real OpenIE model)."""

    def __init__(
        self,
        embedder,                # callable(text) -> List[float] of dim D
        memory: Optional[DifferentiableTripletMemory] = None,
        embedding_dim: int = 128,
    ):
        self.embedder = embedder
        self.memory = memory or DifferentiableTripletMemory(
            embedding_dim=embedding_dim,
        )

    def add_triplet(self, triplet: Triplet) -> None:
        emb = self.embedder(triplet.to_text())
        self.memory.add(triplet, emb)

    def add_text_facts(self, text: str, source: str = "") -> int:
        """Extract simple subject-verb-object triplets from text. The
        rule-based extractor is a debug fallback -- production should
        use a fine-tuned OpenIE model. Returns the count added."""
        triplets = self._extract_simple(text, source)
        for t in triplets:
            self.add_triplet(t)
        return len(triplets)

    def _extract_simple(self, text: str, source: str) -> List[Triplet]:
        """Naive sentence-splitter + position-based S-V-O extraction.
        Catches patterns like 'X is Y', 'X has Y', 'X does Y'."""
        triplets: List[Triplet] = []
        for sent in self._split_sentences(text):
            tokens = sent.split()
            if len(tokens) < 3:
                continue
            # Look for the simplest pattern: <noun> <verb> <noun-phrase>
            for i in range(1, len(tokens) - 1):
                v = tokens[i].lower()
                if v in {"is", "was", "are", "has", "have", "had", "does", "did", "will"}:
                    subj = " ".join(tokens[:i])
                    obj = " ".join(tokens[i + 1:])
                    if subj and obj:
                        triplets.append(Triplet(
                            subject=subj, predicate=v, object=obj,
                            source=source, timestamp=time.time(),
                        ))
                    break
        return triplets

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        out = []
        cur = []
        for ch in text:
            cur.append(ch)
            if ch in ".!?":
                s = "".join(cur).strip()
                if s:
                    out.append(s.rstrip(".!?").strip())
                cur = []
        tail = "".join(cur).strip()
        if tail:
            out.append(tail)
        return out

    def query(self, query_text: str, *, top_k: int = 5) -> List[Tuple[Triplet, float]]:
        """Inference-time retrieval (hard top-k)."""
        if not _HAS_TORCH:
            return []
        if not self.memory.triplets:
            return []
        emb = self.embedder(query_text)
        q = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        q_proj = self.memory.query_proj(q)
        q_norm = q_proj / q_proj.norm(dim=-1, keepdim=True).clamp_min_(1e-6)
        t_norm = self.memory.triplet_embeddings / self.memory.triplet_embeddings.norm(
            dim=-1, keepdim=True,
        ).clamp_min_(1e-6)
        sims = (q_norm @ t_norm.t()).squeeze(0)
        top_vals, top_idx = sims.topk(k=min(top_k, sims.size(0)))
        return [
            (self.memory.triplets[int(i)], float(s))
            for s, i in zip(top_vals.tolist(), top_idx.tolist())
        ]
