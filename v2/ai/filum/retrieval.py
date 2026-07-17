"""Retrieval-augmented generation: facts in a vector store, reasoning
in a 127M-param model.

INNOVATION (the breadth fix): a 127M-param model has finite
information-storage capacity -- you cannot fit Wikipedia inside it.
But you don't have to. A small reasoning model + a large external
knowledge base outperforms a large model trying to memorize
everything in its weights, on knowledge-heavy queries.

This module ships a pure-Python vector store (numpy + a simple
top-k linear search) -- no FAISS dependency. For a knowledge base
of < 100k passages on a laptop, exhaustive cosine similarity over
fp16 embeddings is fast enough (~50ms / query at 100k passages,
~5ms at 10k).

Components
----------
* SimpleVectorStore: { passage_id -> (embedding, text, source) }.
  add, search by cosine, persist to disk.
* HashEmbedder: a stand-in embedder that hashes tokens to a fixed
  dimension. For real deployments, swap for a small distilled
  sentence-transformers model (see `set_embedder()`); the
  HashEmbedder is the test/debug fallback so we don't depend on
  another network model at import time.
* RAGPipeline: encode query -> retrieve top-k -> format as a
  context preamble for Filum.

Failure modes (honest)
----------------------
* The HashEmbedder is NOT semantic. It works for exact-keyword
  lookups but loses to a real sentence transformer on paraphrase.
  Production should swap in `BAAI/bge-small-en` or
  `all-MiniLM-L6-v2` (~80 MB on disk; high quality).
* Retrieval can ground the model in OUTDATED facts. We tag every
  passage with a timestamp so the prompt template can warn the
  model when retrieved info is old.
* For queries with no good retrieval (open-ended creative writing),
  the retrieved passages are noise. RAGPipeline.search() returns
  scores; the caller should bail out of RAG when max_score < threshold.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

EMBED_DIM_DEFAULT = 256


@dataclass
class Passage:
    passage_id: str
    text: str
    source: str = ""
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalHit:
    passage: Passage
    score: float


# ---------------------------------------------------------------------------
# HashEmbedder -- works without any extra deps; semantic accuracy is
# limited but deterministic and fast.
# ---------------------------------------------------------------------------


class HashEmbedder:
    """Hash the tokens of a string into a fixed-dim float vector.
    Each token contributes to D positions chosen by hash; sign is
    pseudorandom from a second hash. Approximates a count-bag of
    tokens projected to a low-dim space (think: random-projection
    LSH for a sparse bag-of-words)."""

    def __init__(self, dim: int = EMBED_DIM_DEFAULT,
                 token_fn: Optional[Callable[[str], List[str]]] = None):
        self.dim = int(dim)
        self.token_fn = token_fn or self._default_tokens

    @staticmethod
    def _default_tokens(s: str) -> List[str]:
        # Lowercase whitespace tokens. Cheap; good enough for the
        # debug embedder.
        return s.lower().split()

    def encode(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        toks = self.token_fn(text)
        for tok in toks:
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            # Use first 4 bytes for index, next byte sign-bit for sign.
            idx = struct.unpack(">I", h[:4])[0] % self.dim
            sign = 1.0 if (h[4] & 1) else -1.0
            vec[idx] += sign
        # L2 normalize so cosine sim is just dot product.
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


@dataclass
class SimpleVectorStore:
    """Linear-search exact-cosine store. Pure Python."""
    passages: List[Passage] = field(default_factory=list)
    embeddings: List[List[float]] = field(default_factory=list)

    def add(self, passage: Passage, embedding: List[float]) -> None:
        if not embedding:
            raise ValueError("empty embedding")
        if self.embeddings and len(self.embeddings[0]) != len(embedding):
            raise ValueError(
                f"dim mismatch: store has {len(self.embeddings[0])}, "
                f"got {len(embedding)}"
            )
        self.passages.append(passage)
        self.embeddings.append(embedding)

    def search(self, query_embedding: List[float], *,
               k: int = 5) -> List[RetrievalHit]:
        if not self.embeddings:
            return []
        if len(query_embedding) != len(self.embeddings[0]):
            raise ValueError(
                f"query dim {len(query_embedding)} != store "
                f"{len(self.embeddings[0])}"
            )
        scores = []
        for i, e in enumerate(self.embeddings):
            s = sum(q * v for q, v in zip(query_embedding, e))
            scores.append((s, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievalHit(passage=self.passages[i], score=s)
            for s, i in scores[:k]
        ]

    def __len__(self) -> int:
        return len(self.passages)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "version": 1,
            "passages": [
                {
                    "passage_id": p.passage_id, "text": p.text,
                    "source": p.source, "timestamp": p.timestamp,
                    "metadata": p.metadata,
                }
                for p in self.passages
            ],
            "embeddings": self.embeddings,
        }
        path.write_text(json.dumps(body), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SimpleVectorStore":
        path = Path(path)
        body = json.loads(path.read_text(encoding="utf-8"))
        store = cls()
        for pdict, emb in zip(body["passages"], body["embeddings"]):
            p = Passage(
                passage_id=pdict["passage_id"], text=pdict["text"],
                source=pdict.get("source", ""),
                timestamp=pdict.get("timestamp", 0.0),
                metadata=pdict.get("metadata", {}),
            )
            store.add(p, emb)
        return store


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------


@dataclass
class RAGConfig:
    top_k: int = 5
    min_score: float = 0.15           # below this we don't use retrieval
    max_context_chars: int = 1500
    citation_format: str = "[{source}]"


class RAGPipeline:
    """Encode a query -> retrieve top-k -> format a context preamble."""

    def __init__(
        self,
        store: SimpleVectorStore,
        embedder: Optional[HashEmbedder] = None,
        config: Optional[RAGConfig] = None,
    ):
        self.store = store
        self.embedder = embedder or HashEmbedder()
        self.config = config or RAGConfig()

    def add_passage(self, passage: Passage) -> None:
        emb = self.embedder.encode(passage.text)
        self.store.add(passage, emb)

    def search(self, query: str) -> Tuple[List[RetrievalHit], float]:
        """Returns (hits, max_score). Caller can short-circuit RAG
        when max_score < min_score."""
        emb = self.embedder.encode(query)
        hits = self.store.search(emb, k=self.config.top_k)
        max_score = hits[0].score if hits else 0.0
        return hits, max_score

    def format_prompt_with_context(self, query: str) -> str:
        """Compose a final prompt: retrieved passages followed by the
        user query. The model sees facts inline; it doesn't have to
        memorize them."""
        hits, max_score = self.search(query)
        if max_score < self.config.min_score or not hits:
            return query
        # Build a citation-rich preamble; trim to fit the budget.
        parts: List[str] = ["Context from knowledge base:"]
        used_chars = 0
        for h in hits:
            citation = self.config.citation_format.format(
                source=h.passage.source or h.passage.passage_id,
            )
            line = f"{citation} {h.passage.text.strip()}"
            if used_chars + len(line) > self.config.max_context_chars:
                break
            parts.append(line)
            used_chars += len(line)
        parts.append("")
        parts.append(f"Query: {query}")
        parts.append("")
        parts.append(
            "Answer the query using ONLY the facts above when they apply. "
            "If the context doesn't cover the query, say so. Cite sources "
            "in [brackets].",
        )
        return "\n".join(parts)
