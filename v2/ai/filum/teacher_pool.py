"""Multi-teacher consensus distillation.

INNOVATION (vs single-teacher distillation): we run the same prompt
through K independent teachers (Gemini Flash + Claude Haiku +
optional GPT-4o-mini), measure pairwise Jensen-Shannon divergence
on their token-level distributions, and:

  * If JSD between any pair > threshold -> REJECT the sample. Either
    one teacher is hallucinating or the question is genuinely
    ambiguous; either way we don't want it in the training mix.
  * If they agree -> AVERAGE their top-k logprobs. The averaged
    distribution carries strictly more information than any single
    teacher (variance reduction, bias cancellation across providers
    with different training data).

Why JSD and not KL: JSD is symmetric and bounded in [0, 1] (with
log base 2), so the threshold is interpretable. KL is unbounded
and asymmetric -- thresholds depend on which teacher you put first.

Pluginfer chooses to use teachers from DIFFERENT lineages on purpose:
  * Google Gemini -- trained on different corpora than OpenAI's
  * Anthropic Claude -- different RLHF objective
  * OpenAI GPT-4o-mini -- different again
This means a forged answer from any one provider will disagree
with the other two and get filtered.

Failure modes (honest)
----------------------
* All three teachers might share the SAME bias (e.g. RLHF refusals
  on identical safety topics). Filter is silent on that case --
  consensus on a refusal looks identical to consensus on a fact.
  Mitigation: at least one teacher should be uncensored / less RLHF'd
  (Mistral Codestral, Llama 3 instruct via Together API).
* Free tiers run out. The cache + 24h budget cap protect against
  burning the wallet; daily reset replenishes the queue.
* Per-token logprobs are inconsistent across providers. We
  normalise by re-tokenising teacher output through the student's
  BPE tokenizer; the per-token positions then correspond.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..training.teacher_distill import (
    DistillationGenerator,
    MockTeacher,
    TeacherCache,
    TeacherClient,
    TeacherSample,
)

logger = logging.getLogger(__name__)


@dataclass
class ConsensusSample:
    """The output of a multi-teacher round. Either contains a usable
    averaged sample (`accepted=True`) or an explanation of why the
    teachers couldn't agree."""
    accepted: bool
    averaged: Optional[TeacherSample] = None
    teacher_samples: List[TeacherSample] = field(default_factory=list)
    pairwise_jsd: List[Tuple[str, str, float]] = field(default_factory=list)
    detail: Optional[str] = None


def _jsd(p: List[float], q: List[float]) -> float:
    """Jensen-Shannon divergence between two probability distributions
    over the SAME support. Operates in linear-prob space; returns a
    value in [0, 1] in log-base-2 nats. The caller is responsible for
    aligning supports (we do that in `consensus_filter`)."""
    if len(p) != len(q):
        raise ValueError(f"distribution lengths differ: {len(p)} vs {len(q)}")
    eps = 1e-12
    m = [(pi + qi) / 2 + eps for pi, qi in zip(p, q)]

    def _kl(a, b):
        s = 0.0
        for ai, bi in zip(a, b):
            if ai > eps:
                s += ai * math.log(ai / bi)
        return s

    # Convert to log base 2 by dividing by ln(2). Bounded in [0, 1]
    # in those units when both inputs are valid distributions.
    raw = 0.5 * (_kl(p, m) + _kl(q, m)) / math.log(2)
    return max(0.0, min(1.0, raw))


def _align_topk(samples: List[TeacherSample]) -> Tuple[
    List[List[Dict[int, float]]], int,
]:
    """Realign K teachers' per-token top-k distributions onto the
    UNION of their top-k token ids. Returns (per_token_dist_list, n)
    where n is the number of token positions that all K teachers
    agreed on (we truncate to the shortest sequence)."""
    if not samples:
        return [], 0
    n = min(len(s.per_token) for s in samples)
    if n == 0:
        return [], 0

    aligned: List[List[Dict[int, float]]] = []
    for i in range(n):
        # Per-position: union of all teachers' top-k token ids.
        union_ids: set[int] = set()
        per_teacher_logprobs: List[Dict[int, float]] = []
        for s in samples:
            tid, ids, lps = s.per_token[i]
            d = {int(tid): float(0.0)}  # the chosen token gets prob mass 0 logprob? No -- we read top_lps
            for j, t_id in enumerate(ids):
                d[int(t_id)] = float(lps[j]) if j < len(lps) else -100.0
            per_teacher_logprobs.append(d)
            union_ids.update(d.keys())
        aligned.append(per_teacher_logprobs)
    return aligned, n


def consensus_filter(
    samples: List[TeacherSample],
    *,
    jsd_threshold: float = 0.4,
) -> ConsensusSample:
    """Decide whether the K teachers agree enough to use the sample.

    Computes pairwise JSD on the per-position softmax distributions
    over the UNION of each pair's top-k token ids. If ANY pair has
    JSD > threshold at ANY position, reject. Otherwise return an
    averaged TeacherSample.
    """
    if len(samples) < 2:
        # With one teacher, "consensus" is undefined; accept by default.
        return ConsensusSample(
            accepted=True, averaged=samples[0] if samples else None,
            teacher_samples=samples,
            detail="single_teacher_no_consensus_check",
        )

    aligned, n = _align_topk(samples)
    if n == 0:
        return ConsensusSample(
            accepted=False, teacher_samples=samples,
            detail="no_per_token_logprobs_from_teachers",
        )

    # Pairwise JSD per position; collect the worst across positions
    # and across pairs.
    pairs: List[Tuple[str, str, float]] = []
    worst_jsd = 0.0
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            si, sj = samples[i], samples[j]
            for pos in range(n):
                di, dj = aligned[pos][i], aligned[pos][j]
                # Build aligned dense distributions on the union of
                # token ids.
                union = sorted(set(di.keys()) | set(dj.keys()))

                def _to_probs(d):
                    # Convert log-probs over `union` to a normalized
                    # distribution. Missing ids default to a low
                    # logprob (-15) so the softmax sees them as
                    # near-zero probability.
                    out = []
                    for t in union:
                        out.append(d.get(t, -15.0))
                    # Softmax over the union.
                    m = max(out)
                    exps = [math.exp(x - m) for x in out]
                    z = sum(exps)
                    return [e / z for e in exps]

                p_i = _to_probs(di)
                p_j = _to_probs(dj)
                d_ij = _jsd(p_i, p_j)
                if d_ij > worst_jsd:
                    worst_jsd = d_ij
            pairs.append((si.teacher_id, sj.teacher_id, worst_jsd))

    if worst_jsd > jsd_threshold:
        return ConsensusSample(
            accepted=False, teacher_samples=samples, pairwise_jsd=pairs,
            detail=f"max_pairwise_jsd={worst_jsd:.3f} > {jsd_threshold}",
        )

    # Average: per position, average each teacher's logprob over the
    # union, take the top-k of the averaged distribution.
    avg_per_token: List[Tuple[int, List[int], List[float]]] = []
    K = max(20, max(len(s.per_token[0][1]) for s in samples))
    for pos in range(n):
        # Reuse the union we just built above (recompute is cheap and
        # clearer).
        union: set[int] = set()
        for d in aligned[pos]:
            union.update(d.keys())
        union_list = sorted(union)
        avg = []
        for t in union_list:
            mean = sum(d.get(t, -15.0) for d in aligned[pos]) / len(aligned[pos])
            avg.append((t, mean))
        avg.sort(key=lambda x: x[1], reverse=True)
        top = avg[:K]
        # The chosen token = teachers' chosen token, taking the
        # majority vote (tie -> first teacher's pick).
        votes: Dict[int, int] = {}
        for s in samples:
            tid = int(s.per_token[pos][0])
            votes[tid] = votes.get(tid, 0) + 1
        chosen = max(votes.items(), key=lambda x: x[1])[0]
        avg_per_token.append(
            (chosen, [t for t, _ in top], [v for _, v in top]),
        )

    averaged = TeacherSample(
        prompt=samples[0].prompt,
        response_text=samples[0].response_text,    # use first teacher's text for now
        per_token=avg_per_token,
        teacher_id="consensus:" + "+".join(s.teacher_id for s in samples),
        cost_usd=sum(s.cost_usd for s in samples),
    )
    return ConsensusSample(
        accepted=True, averaged=averaged, teacher_samples=samples,
        pairwise_jsd=pairs, detail=f"max_jsd={worst_jsd:.3f}",
    )


# ---------------------------------------------------------------------------
# TeacherPool: orchestrates K parallel teacher generations + consensus
# ---------------------------------------------------------------------------


@dataclass
class TeacherPool:
    """K teachers, parallel sample generation with consensus filter.

    `cache` shared across teachers so the same prompt isn't re-asked
    on retry.
    """
    teachers: List[TeacherClient]
    cache: Optional[TeacherCache] = None
    jsd_threshold: float = 0.4
    accepted_count: int = 0
    rejected_count: int = 0

    async def sample_with_consensus(
        self, prompt: str, *,
        max_tokens: int = 256,
        top_k_logprobs: int = 20,
    ) -> ConsensusSample:
        """Ask all teachers in parallel, run the consensus filter."""
        if not self.teachers:
            return ConsensusSample(accepted=False, detail="no_teachers")

        async def _ask(t: TeacherClient) -> Optional[TeacherSample]:
            if self.cache is not None:
                cached = self.cache.get(t.teacher_id, prompt)
                if cached is not None:
                    return cached
            try:
                s = await t.generate(prompt, max_tokens=max_tokens,
                                     top_k_logprobs=top_k_logprobs)
            except Exception as e:
                logger.warning("teacher %s failed on prompt: %s",
                               t.teacher_id, e)
                return None
            if self.cache is not None:
                self.cache.put(s)
            return s

        results = await asyncio.gather(*[_ask(t) for t in self.teachers])
        good = [s for s in results if s is not None]
        if len(good) < 2:
            return ConsensusSample(
                accepted=False, teacher_samples=good,
                detail=f"only_{len(good)}_teachers_responded",
            )
        out = consensus_filter(good, jsd_threshold=self.jsd_threshold)
        if out.accepted:
            self.accepted_count += 1
        else:
            self.rejected_count += 1
        return out

    @property
    def acceptance_rate(self) -> float:
        n = self.accepted_count + self.rejected_count
        return self.accepted_count / n if n > 0 else 0.0
