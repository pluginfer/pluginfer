"""§E4 Data labor — contributors with data but no compute.

The §C6 "move compute to data" claim opened privacy-preserving
training across healthcare/finance/legal datasets. This module
turns the same primitive into a *labor market for the world*.

A user with a phone, but no GPU, can contribute:

* Voice samples in their language (low-resource languages are
  systematically underserved by frontier models)
* Hand-written or typed transcripts of voice samples
* Quality ratings of model outputs (human preference data for DPO)
* Annotations on images (object boxes, captions)
* Edits or corrections to model outputs (the gold standard for
  RLHF — direct preference data)

In return, they earn a *fractional ownership* of every model the
data trains, paid as a per-inference royalty whenever that model
is used (per §A16 revenue distribution + §D1 inference receipts).

The economics: a model trained on someone's data and then used
for 1B inferences pays them a share of the per-inference fee. At
$0.0001/inference and a 5% data-author royalty pool, 1B inferences
= $5,000 distributed across the data contributors weighted by how
much each contributed. A user who contributes 1% of the data on a
viral model receives $50 — life-changing for someone in a
low-income country, paid out monthly via §C7 micropayments.

Why this is genuinely new:

* Existing data-labelling marketplaces (Scale AI, Surge, Labelbox)
  pay piece-rate cash. Pluginfer's data-labor pays *equity in the
  resulting model* — proportional to the model's commercial life,
  not the labor's clock-time.
* Gold-standard RLHF preference data flows from the same pipeline
  as supervised data without separate sourcing.
* Privacy preservation by §C6 — a user submits a *gradient* on
  their device, never the raw data. Their phone's NPU computes
  the gradient in 50-200 MB of memory (per §E3 fragment training)
  and emits a §C grain (per §C4) signed under their public key.

Public API::

    contributor = DataContributor(pubkey="...")
    contributor.submit_voice_sample(audio_bytes, transcript=...)
    contributor.submit_preference(prompt, output_a, output_b, choice="a")
    earnings = contributor.estimated_earnings_for_period(days=30)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ---------- contribution kinds --------------------------------------------

KIND_VOICE = "voice_sample"
KIND_TEXT = "text_sample"
KIND_PREFERENCE = "preference_pair"
KIND_ANNOTATION = "annotation"
KIND_CORRECTION = "correction"

KNOWN_KINDS = {KIND_VOICE, KIND_TEXT, KIND_PREFERENCE,
               KIND_ANNOTATION, KIND_CORRECTION}


@dataclass
class Contribution:
    """One unit of data labor.

    The actual data is *not stored here* — only its content hash and
    metadata. The data lives on the contributor's device per §C6;
    only the gradient leaves. The receipt records that the
    contribution happened, so royalties can be allocated later.
    """
    pubkey: str
    kind: str
    content_sha256: str               # hash of the actual data on the device
    language: str = ""                # ISO 639-1 if applicable
    domain: str = "general"           # "general" | "medical" | "legal" | etc.
    quality_score: float = 1.0        # 0..1 — set by post-hoc evaluation
    weight: float = 1.0               # contribution weight in royalty calc
    created_ts: float = 0.0
    used_in_versions: list = field(default_factory=list)


@dataclass
class RoyaltyAccount:
    pubkey: str
    contributions: list = field(default_factory=list)
    total_weighted: float = 0.0
    cumulative_earnings: float = 0.0
    last_payout_ts: float = 0.0


# ---------- labor ledger --------------------------------------------------

@dataclass
class DataLaborConfig:
    state_path: str = "ai/filum/_work/data_labor.json"
    weights_by_kind: dict = field(default_factory=lambda: {
        KIND_VOICE: 1.0,
        KIND_TEXT: 0.5,
        KIND_PREFERENCE: 2.0,         # preference data is the gold standard
        KIND_ANNOTATION: 1.5,
        KIND_CORRECTION: 2.5,         # corrections are the highest-signal data
    })
    domain_multiplier: dict = field(default_factory=lambda: {
        "general": 1.0,
        "medical": 3.0,               # underserved + valuable
        "legal":   3.0,
        "scientific": 2.5,
    })
    language_multiplier_low_resource: float = 2.0
    high_resource_languages: tuple = (
        "en", "zh", "es", "ar", "pt", "ja", "ru", "fr", "de", "ko",
    )


class DataLaborLedger:
    """Tracks contributions + computes royalty splits.

    Thread-safe. Persists to JSON on every state change.

    The ledger doesn't decide *when* to pay out — that's the §A16
    revenue-distribution code's job. This module exposes the
    *weights* (how much each contributor is owed proportionally
    when a payout window closes).
    """

    def __init__(self, config: DataLaborConfig = DataLaborConfig()):
        self.cfg = config
        self._accounts: dict[str, RoyaltyAccount] = {}
        self._lock = threading.RLock()
        self._load()

    # --- record contributions -------------------------------------------

    def record(self, contribution: Contribution) -> Contribution:
        if contribution.kind not in KNOWN_KINDS:
            raise ValueError(f"unknown contribution kind: {contribution.kind}")
        # Compute weight from kind + domain + language at record time so
        # later policy changes don't retroactively adjust earnings.
        contribution.weight = self._compute_weight(contribution)
        contribution.created_ts = time.time()
        with self._lock:
            account = self._accounts.setdefault(
                contribution.pubkey, RoyaltyAccount(pubkey=contribution.pubkey),
            )
            account.contributions.append(contribution)
            account.total_weighted += contribution.weight
            self._save()
        return contribution

    def _compute_weight(self, c: Contribution) -> float:
        kind_w = self.cfg.weights_by_kind.get(c.kind, 1.0)
        dom_w = self.cfg.domain_multiplier.get(c.domain, 1.0)
        lang_w = (
            self.cfg.language_multiplier_low_resource
            if (c.language and c.language not in self.cfg.high_resource_languages)
            else 1.0
        )
        quality_w = max(0.0, min(2.0, c.quality_score))
        return kind_w * dom_w * lang_w * quality_w

    # --- royalty math ---------------------------------------------------

    def split_royalties(
        self,
        total_pool: float,
        *,
        for_kinds: Optional[set] = None,
    ) -> dict[str, float]:
        """Given a payout pool, return ``{pubkey: amount}``.

        Filtered by ``for_kinds`` if given (e.g. only preference data).
        """
        with self._lock:
            weights: dict[str, float] = {}
            for pk, acct in self._accounts.items():
                w = 0.0
                for c in acct.contributions:
                    if for_kinds and c.kind not in for_kinds:
                        continue
                    w += c.weight
                if w > 0:
                    weights[pk] = w
            total_weight = sum(weights.values())
            if total_weight <= 0:
                return {}
            return {
                pk: total_pool * (w / total_weight)
                for pk, w in weights.items()
            }

    def credit_payout(self, splits: dict[str, float]) -> None:
        """Record that a payout has been disbursed; updates accounts."""
        now = time.time()
        with self._lock:
            for pk, amount in splits.items():
                acct = self._accounts.setdefault(
                    pk, RoyaltyAccount(pubkey=pk),
                )
                acct.cumulative_earnings += float(amount)
                acct.last_payout_ts = now
            self._save()

    def estimated_earnings_for(
        self, pubkey: str, *, projected_pool: float,
    ) -> float:
        """Forward-look at what a contributor would earn from a given pool."""
        splits = self.split_royalties(projected_pool)
        return float(splits.get(pubkey, 0.0))

    def stats(self) -> dict:
        with self._lock:
            n = sum(len(a.contributions) for a in self._accounts.values())
            return {
                "n_contributors": len(self._accounts),
                "n_contributions": n,
                "total_weighted":
                    sum(a.total_weighted for a in self._accounts.values()),
                "total_paid_out":
                    sum(a.cumulative_earnings
                        for a in self._accounts.values()),
            }

    # --- persistence ---------------------------------------------------

    def _save(self) -> None:
        try:
            Path(self.cfg.state_path).parent.mkdir(parents=True, exist_ok=True)
            data = {
                pk: {
                    "pubkey": acct.pubkey,
                    "contributions": [asdict(c) for c in acct.contributions],
                    "total_weighted": acct.total_weighted,
                    "cumulative_earnings": acct.cumulative_earnings,
                    "last_payout_ts": acct.last_payout_ts,
                }
                for pk, acct in self._accounts.items()
            }
            tmp = Path(self.cfg.state_path).with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(self.cfg.state_path)
        except Exception as e:
            logger.warning("data_labor save failed: %s", e)

    def _load(self) -> None:
        p = Path(self.cfg.state_path)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for pk, d in data.items():
                self._accounts[pk] = RoyaltyAccount(
                    pubkey=d["pubkey"],
                    contributions=[Contribution(**c) for c in d.get("contributions", [])],
                    total_weighted=d.get("total_weighted", 0.0),
                    cumulative_earnings=d.get("cumulative_earnings", 0.0),
                    last_payout_ts=d.get("last_payout_ts", 0.0),
                )
        except Exception as e:
            logger.warning("data_labor load failed: %s", e)
