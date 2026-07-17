"""Self-play prompt generation for recursive distillation.

INNOVATION: the student model itself generates new prompts; the
teacher pool answers those prompts; the student trains on the
(generated_prompt, teacher_answer) pair. After the first few
thousand teacher-supplied seed prompts, this loop produces
unlimited fresh training data without burning more API quota
than the answers cost.

Why it works
------------
The student's generator gradually drifts toward prompt distributions
the student is comfortable with -- that's a problem (it stops
exploring) but it's also free signal (these are the prompts the
student would actually see at deploy time). We counter the drift
with two mechanisms:

  * Diversity scoring: new generated prompts are clustered against
    recent prompts; near-duplicates are rejected.
  * Periodic seed injection: every N rounds we mix in fresh seed
    prompts from a curriculum schedule so the student's distribution
    can't collapse onto a single mode.

This is loosely inspired by:
  * Self-Instruct (Wang et al., 2023) -- LLM generates instructions
    for itself, but they used GPT-3.5 as both generator and teacher;
    we split the roles (small student generator, free-tier teacher).
  * Constitutional AI / RLAIF -- self-play on preferences. We do
    the same with knowledge distillation.

Failure modes (honest)
----------------------
* Without the diversity check, generated prompts collapse onto
  whatever the student already does well. We ship the check by
  default.
* The student's generation quality is poor early on -- the first
  few thousand self-play prompts are mostly noise. We start
  self-play AFTER 5000 supervised steps so the generator is at
  least minimally capable.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

logger = logging.getLogger(__name__)


# Seed prompts for a Pluginfer-relevant curriculum. The student
# starts here, then takes over generation once it has internalised
# the structure.
DEFAULT_SEED_PROMPTS: List[str] = [
    # Routing
    "Route this job to a provider type: 'I need a 30-min Llama-3 8B inference at <0.05 USD'",
    "Route: 'Train a 100M-param transformer on synthetic data overnight'",
    "Route: 'Run a quick image classification on 1000 photos'",
    # Pricing
    "What's a fair USD price for a 1024-token Llama 3 inference at 95% quality?",
    "What's a fair PLG price for renting an idle RTX 4090 for an hour at 3am?",
    # NL parsing
    "Parse: 'I want a quick translation of these 10 docs into Hindi by tonight'",
    "Parse: 'Generate 5 product descriptions, conservative tone, max 100 words each'",
    # Quality scoring
    "Rate the result quality 1-10: provider returned 4 out of 5 expected items",
    "Rate quality: provider returned 'I cannot help with that'",
    # Anomaly
    "Is this provider behaviour suspicious: bid is 100x cheaper than the floor price?",
    "Suspicious? Provider's reported eta_ms is 0.001 ms",
    # General reasoning
    "If a provider's stake is 10 PLG and slash penalty is 50%, what's their post-slash balance?",
    "Why might a provider abstain from bidding on a 'confidential' privacy-class job?",
]


@dataclass
class SelfPlayConfig:
    seed_prompts: List[str] = field(default_factory=lambda: list(DEFAULT_SEED_PROMPTS))
    prompts_per_round: int = 64
    fresh_seed_every_n_rounds: int = 4
    diversity_min_distance: float = 0.3
    history_window: int = 256
    start_after_step: int = 5_000
    max_prompt_tokens: int = 96


def _normalised_lev(a: str, b: str) -> float:
    """Cheap normalised edit distance (0=identical, 1=disjoint).
    O(n*m) is fine for prompts < a few hundred tokens."""
    if a == b:
        return 0.0
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 1.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / max(n, m)


@dataclass
class SelfPlayGenerator:
    """Drives the self-play loop. The owner provides:

      generate_fn(seed_prompt) -> Awaitable[str]
            -- the student's current generator. Must produce a
               textual prompt (not a response) given a seed.

    On each `propose_round()` call, returns a list of
    `prompts_per_round` deduplicated, diversity-filtered prompts."""
    config: SelfPlayConfig
    generate_fn: Callable[[str], Awaitable[str]]
    history: List[str] = field(default_factory=list)
    rounds: int = 0

    def _is_diverse(self, prompt: str) -> bool:
        if not prompt or not prompt.strip():
            return False
        for h in self.history[-self.config.history_window:]:
            if _normalised_lev(prompt, h) < self.config.diversity_min_distance:
                return False
        return True

    async def propose_round(self) -> List[str]:
        """One self-play round: generate prompts, filter for diversity,
        return what's left."""
        self.rounds += 1
        out: List[str] = []
        # Seed mix: every N rounds, half are fresh seeds; otherwise
        # one or two are seeded as anchors.
        if self.rounds % self.config.fresh_seed_every_n_rounds == 0:
            n_seed = self.config.prompts_per_round // 2
        else:
            n_seed = max(2, self.config.prompts_per_round // 8)

        # Inject fresh seeds straight into the round (no generation).
        seeds = random.sample(
            self.config.seed_prompts,
            min(n_seed, len(self.config.seed_prompts)),
        )
        for s in seeds:
            if self._is_diverse(s):
                out.append(s)
                self.history.append(s)

        # Generate the rest by asking the student to expand a seed.
        remaining = self.config.prompts_per_round - len(out)
        attempts = 0
        max_attempts = remaining * 4
        while len(out) < self.config.prompts_per_round and attempts < max_attempts:
            attempts += 1
            seed = random.choice(self.config.seed_prompts)
            try:
                generated = await self.generate_fn(seed)
            except Exception as e:
                logger.warning("self-play generator failed: %s", e)
                continue
            generated = (generated or "").strip()
            if not generated:
                continue
            # Truncate to keep prompts compact.
            generated = generated[:self.config.max_prompt_tokens * 4]
            if self._is_diverse(generated):
                out.append(generated)
                self.history.append(generated)

        # Bound the history.
        if len(self.history) > self.config.history_window * 2:
            self.history = self.history[-self.config.history_window:]
        return out
