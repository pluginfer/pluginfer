"""Multi-teacher distillation: sample-efficient training from
free-tier LLM APIs.

Why
---
Pretraining 1.13B on raw web text needs ~100B tokens to converge.
At GeForce-class throughput (~5k tokens/sec on a 1660), that's
~250 days of wall-clock training. Even with all the other
optimizations in this directory, you do not finish.

Distillation gives 10-50× sample efficiency. The student learns
from a TEACHER's output distribution rather than from one-hot
target tokens. Each token's gradient carries far more information:
the teacher's full softmax over the vocab is N values vs the
single ground-truth token. Plus the teacher has already learned
the linguistic structure -- the student is just being shaped to
match it.

Free-tier teachers (as of 2026 -- we use availability adapters):
  * Google Gemini 1.5 Flash / 2.0 Flash : 1500 req/day free
  * Anthropic Claude Haiku              : ~$0.25 / 1M input tokens
  * OpenAI GPT-4o-mini                  : $0.15 / 1M input tokens
  * Mistral Codestral / Small           : free dev tier
  * Cohere Command R                    : free trial 1k req/day

Using a TRIO (or quartet) gives:
  * ~10k-100k high-quality (prompt, response) pairs / day for free
  * Disagreement-resolution: average teacher logprobs to reduce
    individual-teacher hallucinations.
  * Curriculum: start with simple prompts (token completion,
    cloze tasks); ramp to instruction following.

What's in this module
---------------------
* TeacherClient ABC with concrete adapters (mocked for tests; real
  ones use the env-var-keyed clients each provider ships).
* DistillationDatasetGenerator: produces (prompt, response_tokens,
  response_logprobs) tuples on a background asyncio task pool,
  rate-limit-aware.
* DistillationLoss: KL-divergence between the student's logits and
  the teacher's logprobs, optionally with a hard-target cross-
  entropy mixed in (alpha-blend).
* run_distillation(model, generator, loss, optimizer, ...) trains
  the student on a stream of teacher samples.

Failure modes (honest)
----------------------
* Teachers can output the same wrong answer (cohort collapse).
  Mitigation: rejection sampling + at least one teacher with
  different training data (Gemini + Claude + GPT-4o is the
  recommended trio: independent base models, independent training
  corpora).
* Free-tier rate limits halt training when burned. We round-robin
  + cache aggressively; the cache is the second source of training
  data for the rest of the run.
* The student inherits the teacher's BIASES + REFUSALS. A medical
  Q without context might trigger every teacher's refusal layer;
  the student then learns "don't answer" rather than the actual
  knowledge. Filter refusal outputs out at generation time.

References
----------
* "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)
* "MiniLLM: Knowledge Distillation of Large Language Models" (Gu et al., 2024)
* "DistilQwen / TinyLlama" -- well-known small-model distillation runs.
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import json
import logging
import math
import os
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
    nn = None
    F = None
    _HAS_TORCH = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Teacher abstraction
# ---------------------------------------------------------------------------


@dataclass
class TeacherSample:
    """One distillation sample. The student is trained to match
    `top_k_logprobs` (NOT the full vocab -- that's never available
    from APIs; we use the top-k that the API exposes)."""
    prompt: str
    response_text: str
    # Per-token (token_id, top_k_token_ids, top_k_logprobs).
    # token_id is the actual sampled token; top_k_* is the teacher's
    # top-k alternatives at that position.
    per_token: List[Tuple[int, List[int], List[float]]]
    teacher_id: str
    cost_usd: float = 0.0
    cached: bool = False


class TeacherClient(abc.ABC):
    """One LLM API. Concrete subclasses wrap the provider SDK."""

    @property
    @abc.abstractmethod
    def teacher_id(self) -> str: ...

    @abc.abstractmethod
    async def generate(self, prompt: str, *,
                       max_tokens: int, top_k_logprobs: int = 20) -> TeacherSample:
        ...


class MockTeacher(TeacherClient):
    """Deterministic test teacher. Returns a canned response with
    flat top-k logprobs. Used by tests so no network is involved."""

    def __init__(self, teacher_id: str = "mock", canned: str = "the cat sat",
                 vocab_size: int = 256):
        self._id = teacher_id
        self.canned = canned
        self.vocab_size = vocab_size

    @property
    def teacher_id(self) -> str:
        return self._id

    async def generate(self, prompt: str, *, max_tokens: int,
                       top_k_logprobs: int = 20) -> TeacherSample:
        # Tokenize the canned response by bytes (the BPE tokenizer
        # would do better but tests don't need that). Truncate.
        toks = list(self.canned.encode("utf-8"))[:max_tokens]
        per_token = []
        for t in toks:
            top_ids = [(t + i) % self.vocab_size for i in range(top_k_logprobs)]
            # Synthetic top-k logprobs: peak on `t`, decaying for the
            # rest. Sums to ~1 in probability space.
            logprobs = [-0.1] + [-2.0 - 0.5 * i for i in range(1, top_k_logprobs)]
            per_token.append((t, top_ids, logprobs))
        return TeacherSample(
            prompt=prompt,
            response_text=self.canned,
            per_token=per_token,
            teacher_id=self._id,
        )


class AnthropicTeacher(TeacherClient):
    """Real Claude teacher. Uses the anthropic SDK if available; the
    SDK exposes per-token logprobs for the top alternatives via the
    `logprobs` parameter (see provider docs)."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001",
                 api_key_env: str = "ANTHROPIC_API_KEY",
                 *, request_per_min: int = 60):
        self.model = model
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"{api_key_env} not set; cannot use Anthropic teacher",
            )
        self.request_per_min = int(request_per_min)
        self._last_request_at = 0.0
        self._client = None

    @property
    def teacher_id(self) -> str:
        return f"anthropic:{self.model}"

    async def _client_lazy(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise RuntimeError(
                    "pip install anthropic to use AnthropicTeacher",
                ) from e
            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def generate(self, prompt: str, *, max_tokens: int,
                       top_k_logprobs: int = 20) -> TeacherSample:
        # Rate limit (per the provider's free-tier defaults).
        now = time.monotonic()
        delta = 60.0 / max(1, self.request_per_min)
        wait = max(0.0, delta - (now - self._last_request_at))
        if wait > 0:
            await asyncio.sleep(wait)
        client = await self._client_lazy()
        msg = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        self._last_request_at = time.monotonic()
        # The Anthropic API does NOT expose token-level logprobs
        # publicly as of 2026-05; we return a TeacherSample with
        # response text only. The caller falls back to hard-target
        # cross-entropy on this sample (which is still useful for
        # supervised fine-tuning, just not full distillation).
        text = "".join(b.text for b in msg.content if hasattr(b, "text"))
        return TeacherSample(
            prompt=prompt, response_text=text, per_token=[],
            teacher_id=self.teacher_id,
            cost_usd=_estimate_cost(self.model, msg),
        )


class GeminiTeacher(TeacherClient):
    """Google Gemini teacher. The 2.0 Flash model exposes
    avgLogprobs via the responseLogprobs config (Vertex AI / GenAI)."""

    def __init__(self, model: str = "gemini-2.5-flash",
                 api_key_env: str = "GOOGLE_API_KEY",
                 *, request_per_min: int = 15):
        self.model = model
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"{api_key_env} not set; cannot use Gemini teacher",
            )
        self.request_per_min = int(request_per_min)
        self._last_request_at = 0.0
        self._client = None

    @property
    def teacher_id(self) -> str:
        return f"google:{self.model}"

    async def _client_lazy(self):
        if self._client is None:
            try:
                import google.generativeai as genai
            except ImportError as e:
                raise RuntimeError(
                    "pip install google-generativeai to use GeminiTeacher",
                ) from e
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
        return self._client

    async def generate(self, prompt: str, *, max_tokens: int,
                       top_k_logprobs: int = 20) -> TeacherSample:
        now = time.monotonic()
        delta = 60.0 / max(1, self.request_per_min)
        wait = max(0.0, delta - (now - self._last_request_at))
        if wait > 0:
            await asyncio.sleep(wait)
        client = await self._client_lazy()
        # Run sync SDK call in a thread (the genai SDK is currently sync).
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: client.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "response_logprobs": True,
                    "logprobs": top_k_logprobs,
                },
            ),
        )
        self._last_request_at = time.monotonic()
        text = resp.text if hasattr(resp, "text") else ""
        # Per-token logprobs may be on resp.candidates[0].logprobs_result.
        per_token = []
        try:
            cand = resp.candidates[0]
            for tok in (cand.logprobs_result.chosen_candidates or []):
                # Each "tok" has `token_id`, `log_probability`, and
                # alternatives. Schema may vary across SDK versions;
                # fall back to empty per_token if shape doesn't match.
                tid = getattr(tok, "token_id", None)
                if tid is None:
                    continue
                top_ids = [getattr(a, "token_id", -1)
                           for a in getattr(tok, "alternative_tokens", [])]
                top_lps = [getattr(a, "log_probability", -100.0)
                           for a in getattr(tok, "alternative_tokens", [])]
                per_token.append((tid, top_ids, top_lps))
        except Exception:
            per_token = []
        return TeacherSample(
            prompt=prompt, response_text=text, per_token=per_token,
            teacher_id=self.teacher_id,
        )


def _estimate_cost(model: str, msg) -> float:
    """Rough $/M tokens estimate -- used for budget telemetry only."""
    table = {
        "claude-haiku-4-5-20251001": (0.80, 4.00),     # in / out per 1M tokens
        "claude-sonnet-4-6": (3.00, 15.00),
        "claude-opus-4-7": (15.00, 75.00),
        "gemini-2.0-flash-exp": (0.0, 0.0),            # free tier
    }
    in_rate, out_rate = table.get(model, (1.0, 5.0))
    try:
        usage = getattr(msg, "usage", None)
        if usage:
            in_tok = getattr(usage, "input_tokens", 0) or 0
            out_tok = getattr(usage, "output_tokens", 0) or 0
            return (in_tok / 1e6) * in_rate + (out_tok / 1e6) * out_rate
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Disk cache so we don't re-burn rate limits between runs
# ---------------------------------------------------------------------------


@dataclass
class TeacherCache:
    """SHA256-keyed disk cache of teacher samples. Survives crashes.
    The same prompt asked twice hits the cache the second time."""
    cache_dir: Path
    hit: int = 0
    miss: int = 0

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, teacher_id: str, prompt: str) -> Path:
        h = hashlib.sha256(f"{teacher_id}|{prompt}".encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    def get(self, teacher_id: str, prompt: str) -> Optional[TeacherSample]:
        p = self._path(teacher_id, prompt)
        if not p.exists():
            self.miss += 1
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            self.hit += 1
            return TeacherSample(
                prompt=d["prompt"],
                response_text=d["response_text"],
                per_token=[tuple(x) for x in d["per_token"]],
                teacher_id=d["teacher_id"],
                cost_usd=float(d.get("cost_usd", 0.0)),
                cached=True,
            )
        except Exception:
            self.miss += 1
            return None

    def put(self, sample: TeacherSample) -> None:
        p = self._path(sample.teacher_id, sample.prompt)
        try:
            p.write_text(json.dumps({
                "prompt": sample.prompt,
                "response_text": sample.response_text,
                "per_token": [list(x) for x in sample.per_token],
                "teacher_id": sample.teacher_id,
                "cost_usd": sample.cost_usd,
            }), encoding="utf-8")
        except Exception as e:                                  # pragma: no cover
            logger.warning("teacher cache put failed: %s", e)


# ---------------------------------------------------------------------------
# Multi-teacher generator
# ---------------------------------------------------------------------------


@dataclass
class DistillationGenerator:
    """Round-robin over a pool of TeacherClients. Caches every sample
    so future runs replay from disk."""
    teachers: List[TeacherClient]
    cache: Optional[TeacherCache] = None
    _next_idx: int = 0

    def __post_init__(self) -> None:
        if not self.teachers:
            raise ValueError("at least one teacher required")

    async def sample(self, prompt: str, *, max_tokens: int = 256,
                     top_k_logprobs: int = 20) -> TeacherSample:
        teacher = self.teachers[self._next_idx % len(self.teachers)]
        self._next_idx += 1
        if self.cache is not None:
            cached = self.cache.get(teacher.teacher_id, prompt)
            if cached is not None:
                return cached
        try:
            sample = await teacher.generate(
                prompt, max_tokens=max_tokens,
                top_k_logprobs=top_k_logprobs,
            )
        except Exception as e:
            logger.warning("teacher %s failed: %s; trying next",
                           teacher.teacher_id, e)
            # Try the rest of the pool.
            for fallback in self.teachers:
                if fallback is teacher:
                    continue
                try:
                    sample = await fallback.generate(
                        prompt, max_tokens=max_tokens,
                        top_k_logprobs=top_k_logprobs,
                    )
                    break
                except Exception:
                    continue
            else:
                raise
        if self.cache is not None and not sample.cached:
            self.cache.put(sample)
        return sample


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------


def distillation_loss(
    student_logits, sample: TeacherSample, *,
    alpha: float = 0.5, temperature: float = 1.0,
):
    """Combined loss:

      L = alpha * KL(student_topk || teacher_topk)        (soft targets)
        + (1 - alpha) * CE(student, hard_target)         (hard targets)

    `student_logits` -- (T, vocab) logits from the student over the
    teacher's response tokens.
    `sample` -- TeacherSample with per_token (token_id, top_ids,
    top_logprobs).

    `alpha=0.5` is the Hinton-paper default.

    If the sample has empty `per_token` (provider didn't expose
    logprobs), falls back to pure hard-target cross-entropy."""
    if not _HAS_TORCH:
        raise RuntimeError("torch required")
    if not sample.per_token:
        # Hard-target only.
        targets = torch.tensor(
            [list(sample.response_text.encode("utf-8"))[:student_logits.size(0)][0]
             if sample.response_text else 0],
            device=student_logits.device, dtype=torch.long,
        )
        # Truncate to the response length the teacher gave us.
        return F.cross_entropy(student_logits[:targets.size(0)], targets)

    # Soft + hard.
    n = min(len(sample.per_token), student_logits.size(0))
    soft_loss = student_logits.new_zeros(())
    hard_targets = []
    for i in range(n):
        token_id, top_ids, top_lps = sample.per_token[i]
        hard_targets.append(token_id)
        # Teacher's distribution over the top-k slots.
        teacher_lps = torch.tensor(top_lps, device=student_logits.device,
                                   dtype=student_logits.dtype) / temperature
        teacher_probs = F.softmax(teacher_lps, dim=-1)
        # Student's logits at the same top-k slots.
        student_top = student_logits[i, top_ids] / temperature
        student_log_probs = F.log_softmax(student_top, dim=-1)
        # KL(teacher || student) -- conventional direction for distillation.
        soft_loss = soft_loss - (teacher_probs * student_log_probs).sum()
    soft_loss = soft_loss / max(1, n)

    hard_target_t = torch.tensor(hard_targets, device=student_logits.device,
                                 dtype=torch.long)
    hard_loss = F.cross_entropy(student_logits[:n], hard_target_t)

    return alpha * soft_loss + (1 - alpha) * hard_loss
