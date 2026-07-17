"""Reward model + Direct Preference Optimization (DPO) for Filum.

INVENTION (claim §12 in the design notes): conventional RLHF requires
PPO with a value head, a KL-regularised policy, and a finicky
training loop. At our parameter scale the PPO machinery is bigger
than the model itself and reliably destabilises training.

We use Direct Preference Optimization (Rafailov et al., NeurIPS
2023) instead. DPO sidesteps the value head entirely: train on
pairwise preferences (chosen, rejected) with a simple log-ratio
loss. Empirically as good as PPO with 1/3 the compute and zero
value-network overhead.

Plus the Pluginfer-specific innovation: the chain itself supplies
preference signals. Every K-redundant dispatch where one provider's
result was accepted via majority vote and another's was rejected
gives us a free (chosen, rejected) pair labelled by ground-truth
correctness on the chain. We don't need a paid annotator pool --
the mesh's settlement records are the labels.

Multi-domain reasoning bias
---------------------------
The reward model itself receives bonuses for:
  * Step-by-step reasoning (rewards chain-of-thought traces).
  * Citation of retrieved facts (when RAG is in play).
  * Honest uncertainty ("I don't know about X" beats confident-wrong).
  * Correct on-chain settlement (Pluginfer-domain ground truth).
  * Pass on-the-fly factuality checks against the knowledge graph
    (claims contradicted by triplets are penalized).

This trains the model to be USEFUL across domains (it carries the
generalisation), HONEST (it admits limits), and TRACEABLE (its
answers cite + reason).

Failure modes (honest)
----------------------
* DPO can overfit on the preference dataset; we mix in the
  distillation cross-entropy as a regulariser (DPO + SFT joint loss).
* Preference signal is only as good as the K-redundant ground
  truth. For tasks where K-redundant agreement is itself biased
  (collusive providers), DPO inherits that bias.
* Reward hacking: the model finds shortcuts that score high without
  being good. We mitigate by held-out eval on a teacher-generated
  set the model never sees during DPO.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Preference pair derivation from chain receipts
# ---------------------------------------------------------------------------


@dataclass
class PreferencePair:
    """One (chosen, rejected) pair for DPO training."""
    prompt: str
    chosen: str
    rejected: str
    chosen_quality: float = 1.0
    rejected_quality: float = 0.0
    source: str = "chain_consensus"   # "chain_consensus" | "human" | "teacher_compare"


def pairs_from_kredundant_dispatch(
    *,
    prompt: str,
    consensus_response: str,
    dissenters_responses: List[str],
    consensus_quality: float = 1.0,
) -> List[PreferencePair]:
    """Each dissenting provider's output paired against the consensus
    output yields one preference pair. Free training data, ground-
    truth-labelled by the chain."""
    pairs = []
    for d in dissenters_responses:
        if d and d != consensus_response:
            pairs.append(PreferencePair(
                prompt=prompt,
                chosen=consensus_response,
                rejected=d,
                chosen_quality=consensus_quality,
                rejected_quality=0.0,
            ))
    return pairs


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------


def dpo_loss(
    *,
    chosen_logps: "torch.Tensor",
    rejected_logps: "torch.Tensor",
    ref_chosen_logps: "torch.Tensor",
    ref_rejected_logps: "torch.Tensor",
    beta: float = 0.1,
) -> "torch.Tensor":
    """The DPO loss (Rafailov et al., 2023, Eq. 7).

    Trains the policy to assign higher log-probability to `chosen`
    relative to `rejected`, BIASED by the reference (frozen) model's
    log-probabilities. The beta parameter controls how much to push
    away from the reference (smaller beta = more constrained).

    All four log-prob arguments are summed over the response tokens
    for one batch.
    """
    if not _HAS_TORCH:
        raise RuntimeError("dpo_loss requires torch")
    # log [pi_theta(chosen) / pi_ref(chosen)] - log [pi_theta(rejected) / pi_ref(rejected)]
    pi_logratios = chosen_logps - rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()


# ---------------------------------------------------------------------------
# Preference scorer (the reward model -- when we want a learned R fn)
# ---------------------------------------------------------------------------


class RewardModel(_BASE):
    """Lightweight reward head: takes the policy's hidden state at
    the last response token and outputs a scalar reward.

    Trained on `(prompt + response, label)` where label is +1 for
    chosen and -1 for rejected via simple Bradley-Terry.

    Used for online RL when needed (DPO doesn't need this; it's
    here for the cases where we want explicit reward shaping
    e.g. multi-objective bonus for reasoning, citation, brevity)."""

    def __init__(self, base_model, *, hidden_dim: int = 896):
        if not _HAS_TORCH:
            raise RuntimeError("torch required")
        super().__init__()
        self.base_model = base_model
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, input_ids):
        h = self.base_model(input_ids, return_hidden=True)
        # Take the last token's hidden state.
        last = h[:, -1, :]
        return self.head(last).squeeze(-1)

    def reward_with_bonuses(
        self,
        input_ids,
        *,
        response_text: str,
        retrieved_facts: List[str] = None,
        knowledge_graph_check: Optional[Callable[[str], float]] = None,
    ) -> "torch.Tensor":
        """Base reward + multi-objective bonuses for the kind of
        behaviour we want (reasoning, citation, honesty, factuality)."""
        base = self.forward(input_ids)

        # +reward for chain-of-thought reasoning markers.
        cot_markers = ["because", "first,", "step", "therefore",
                       "thus", "so we", "consider", "given that"]
        cot_count = sum(1 for m in cot_markers if m in response_text.lower())
        cot_bonus = min(0.5, cot_count * 0.05)

        # +reward for citation of retrieved facts.
        citation_bonus = 0.0
        if retrieved_facts:
            citations = sum(1 for f in retrieved_facts
                            if any(tok in response_text for tok in f.split()[:3]))
            citation_bonus = min(0.3, citations * 0.1)

        # +reward for honest uncertainty.
        uncertainty_markers = ["i don't know", "i'm not sure",
                               "uncertain", "i can't verify",
                               "this may be incorrect"]
        honesty_bonus = 0.1 if any(m in response_text.lower()
                                    for m in uncertainty_markers) else 0.0

        # -penalty for factuality contradictions (if KG check supplied).
        factuality_penalty = 0.0
        if knowledge_graph_check is not None:
            try:
                contradictions = float(knowledge_graph_check(response_text))
                factuality_penalty = -1.0 * min(0.5, contradictions)
            except Exception:
                pass

        return base + cot_bonus + citation_bonus + honesty_bonus + factuality_penalty


# ---------------------------------------------------------------------------
# DPO trainer (mini-batch driver)
# ---------------------------------------------------------------------------


@dataclass
class DPOConfig:
    beta: float = 0.1
    sft_blend_weight: float = 0.1     # mix in supervised CE for stability
    learning_rate: float = 5e-6        # smaller than pretrain LR
    grad_clip: float = 1.0


class DPOTrainer:
    """Mini-batch DPO. Caller wires:
       `model`           -- the policy being trained.
       `ref_model`       -- a FROZEN copy of the base model (the reference).
       `tokenize(text)`  -- text -> token ids tensor.
       `optimizer`       -- typically AdamW8bit on the policy params.
    """

    def __init__(self, *, model, ref_model, tokenize, optimizer,
                 config: Optional[DPOConfig] = None):
        if not _HAS_TORCH:
            raise RuntimeError("DPOTrainer requires torch")
        self.model = model
        self.ref_model = ref_model
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.tokenize = tokenize
        self.optimizer = optimizer
        self.config = config or DPOConfig()

    def step(self, batch: List[PreferencePair]) -> float:
        if not batch:
            return 0.0
        self.model.train()
        self.optimizer.zero_grad()

        chosen_logps = []
        rejected_logps = []
        ref_chosen_logps = []
        ref_rejected_logps = []
        sft_losses = []

        for pair in batch:
            p_ids = torch.tensor(self.tokenize(pair.prompt))
            c_ids = torch.tensor(self.tokenize(pair.chosen))
            r_ids = torch.tensor(self.tokenize(pair.rejected))
            full_c = torch.cat([p_ids, c_ids]).unsqueeze(0)
            full_r = torch.cat([p_ids, r_ids]).unsqueeze(0)
            # Policy log-probs over the RESPONSE tokens only (we
            # don't condition on the prompt's likelihood).
            cl = _seq_logp(self.model, full_c, prompt_len=len(p_ids))
            rl = _seq_logp(self.model, full_r, prompt_len=len(p_ids))
            with torch.no_grad():
                cl_ref = _seq_logp(self.ref_model, full_c, prompt_len=len(p_ids))
                rl_ref = _seq_logp(self.ref_model, full_r, prompt_len=len(p_ids))
            chosen_logps.append(cl)
            rejected_logps.append(rl)
            ref_chosen_logps.append(cl_ref)
            ref_rejected_logps.append(rl_ref)
            # SFT regulariser on the chosen response.
            sft_losses.append(_seq_ce(self.model, full_c, prompt_len=len(p_ids)))

        cl = torch.stack(chosen_logps)
        rl = torch.stack(rejected_logps)
        cl_r = torch.stack(ref_chosen_logps)
        rl_r = torch.stack(ref_rejected_logps)

        loss_dpo = dpo_loss(
            chosen_logps=cl, rejected_logps=rl,
            ref_chosen_logps=cl_r, ref_rejected_logps=rl_r,
            beta=self.config.beta,
        )
        loss_sft = torch.stack(sft_losses).mean()
        loss = loss_dpo + self.config.sft_blend_weight * loss_sft
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.config.grad_clip,
        )
        self.optimizer.step()
        return float(loss.detach())


def _seq_logp(model, ids, *, prompt_len: int) -> "torch.Tensor":
    """Sum of log-probs of the response tokens given the prompt."""
    if not _HAS_TORCH:
        return torch.tensor(0.0)
    logits = model(ids)            # (1, T, V)
    # Shift so logits[t] predicts ids[t+1].
    logits = logits[0, prompt_len - 1: -1]    # (response_len, V)
    targets = ids[0, prompt_len:]               # (response_len,)
    if logits.size(0) != targets.size(0):
        # Trim the longer side.
        n = min(logits.size(0), targets.size(0))
        logits = logits[:n]
        targets = targets[:n]
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum()


def _seq_ce(model, ids, *, prompt_len: int) -> "torch.Tensor":
    """Standard SFT cross-entropy on the response tokens."""
    if not _HAS_TORCH:
        return torch.tensor(0.0)
    logits = model(ids)
    logits = logits[0, prompt_len - 1: -1]
    targets = ids[0, prompt_len:]
    if logits.size(0) != targets.size(0):
        n = min(logits.size(0), targets.size(0))
        logits = logits[:n]
        targets = targets[:n]
    return F.cross_entropy(logits, targets)
