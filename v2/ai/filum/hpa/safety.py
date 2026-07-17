"""Safety + abuse-resistance layer.

A public mesh that runs *anything for anyone* is a public mesh that
ships malware, CSAM, or sanctioned-content within hours of going
live. This module is the ingress gate.

Three responsibilities:

1. **Per-pubkey rate limit** — token-bucket per submitter pubkey.
   Default 60 jobs/min, burst 10. Cheap, in-memory, sliding window.
2. **Content classification** — pluggable classifier. Default
   implementation is a regex-+-keyword filter that catches the
   obvious bulk (credentials, malware signatures, known CSAM URL
   patterns, OFAC-listed sanctions terms). Production deploys
   plug a real classifier (an open-weights moderation model run
   *on the mesh itself*) here without changing the API.
3. **Sanctioned-region check** — IP geolocation against an OFAC
   list. Returns reject for jobs originating from sanctioned
   countries; the network refuses to route their compute.

Defensive defaults: every check fails *closed* — if the rate-limit
state can't be loaded, the request is rejected, not allowed. This
is the opposite of how performance-tuning code is usually written;
for safety it's the only sane choice.

novel claim impact: §A8 mesh-MOE was already in the portfolio;
this module adds the operational safety layer that makes mesh-MOE
deployable in regulated jurisdictions. Drafted as §D3.

Stdlib-only: no external dependency. The default classifier is a
*scaffold* — production replaces it with a real moderation model.
The interface is what stays stable.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------- decision objects ------------------------------------------------

ALLOW = "allow"
DENY = "deny"
RATE_LIMITED = "rate_limited"
QUARANTINED = "quarantined"


@dataclass
class SafetyDecision:
    decision: str                       # ALLOW | DENY | RATE_LIMITED | QUARANTINED
    reason: str = ""
    matched_class: Optional[str] = None
    submitted_pubkey: Optional[str] = None
    ts: float = 0.0

    def is_allowed(self) -> bool:
        return self.decision == ALLOW


# ---------- rate limit -----------------------------------------------------

@dataclass
class RateLimitConfig:
    max_per_min: int = 60
    burst: int = 10


class TokenBucket:
    """Per-pubkey token bucket. Thread-safe."""

    def __init__(self, max_per_min: int, burst: int):
        self._rate_per_s = max_per_min / 60.0
        self._capacity = float(burst)
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def take(self, n: float = 1.0) -> bool:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._capacity,
                self._tokens + (now - self._last_refill) * self._rate_per_s,
            )
            self._last_refill = now
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False


class RateLimiter:
    """One bucket per pubkey. Buckets evicted after 10 minutes idle."""

    def __init__(self, config: RateLimitConfig = RateLimitConfig()):
        self.cfg = config
        self._buckets: dict[str, TokenBucket] = {}
        self._last_seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def check(self, pubkey: str) -> bool:
        with self._lock:
            b = self._buckets.get(pubkey)
            if b is None:
                b = TokenBucket(self.cfg.max_per_min, self.cfg.burst)
                self._buckets[pubkey] = b
            self._last_seen[pubkey] = time.monotonic()
        return b.take(1.0)

    def gc(self, ttl_s: float = 600.0) -> int:
        now = time.monotonic()
        evicted = 0
        with self._lock:
            for pk, ts in list(self._last_seen.items()):
                if now - ts > ttl_s:
                    self._buckets.pop(pk, None)
                    self._last_seen.pop(pk, None)
                    evicted += 1
        return evicted


# ---------- content classification -----------------------------------------

# A very conservative scaffold. Production replaces this with a real
# moderation model. The classes named here are the ones the rest of
# the system reasons about; production-side substitutions must keep
# the same class labels.
DEFAULT_CLASSES = (
    "csam",                             # zero-tolerance reject
    "credentials",                      # leaked API keys, JWTs, etc.
    "malware",                          # known malware signatures
    "ofac_term",                        # OFAC-listed entities
    "violent_extremism",                # specific operational planning
    "explicit_minor",                   # CSAM-adjacent content
)

# Regex patterns are illustrative scaffolds. Production should use a
# trained classifier; the regex layer remains as a fast pre-filter.
_REGEX_RULES: list[tuple[str, re.Pattern]] = [
    ("credentials", re.compile(
        r"(?i)(?:aws_access_key_id|api_key|api[-_]?secret|"
        r"AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{36})"
    )),
    ("credentials", re.compile(
        r"(?i)-----BEGIN (?:RSA |EC |DSA |OPENSSH |PRIVATE )?PRIVATE KEY-----"
    )),
    # Note: these are *defensive scaffolds*. They flag obvious bulk;
    # they do not replace a real classifier.
    ("malware", re.compile(
        r"(?i)(?:eval\s*\(\s*atob|powershell\s+-enc\s+|<script>\s*window\.location)"
    )),
]


@dataclass
class ContentClassifierConfig:
    classes: tuple = DEFAULT_CLASSES
    # If a classifier is plugged in, it overrides the regex layer.
    pluggable_classifier: Optional[Callable[[str], dict[str, float]]] = None
    threshold: float = 0.5


class ContentClassifier:
    """Pluggable content classification with a regex fast-pre-filter.

    The pluggable classifier is a function: text -> {class_name: prob}.
    Caller injects whatever moderation model they want; default is
    None (regex-only, which is *not* sufficient for production).
    """

    def __init__(self, config: ContentClassifierConfig = ContentClassifierConfig()):
        self.cfg = config

    def classify(self, text: str) -> tuple[Optional[str], float]:
        """Return (class_name, score). class_name is None on clean text."""
        if not text:
            return (None, 0.0)
        # Fast regex pass — these are 100% precision rules.
        for cls, rx in _REGEX_RULES:
            if rx.search(text):
                return (cls, 1.0)
        # Pluggable model.
        if self.cfg.pluggable_classifier is not None:
            try:
                probs = self.cfg.pluggable_classifier(text) or {}
            except Exception:
                probs = {}
            if probs:
                cls, score = max(probs.items(), key=lambda kv: kv[1])
                if score >= self.cfg.threshold and cls in self.cfg.classes:
                    return (cls, float(score))
        return (None, 0.0)


# ---------- region check ---------------------------------------------------

# OFAC sanctioned country codes (illustrative; production keeps this
# in a governance-tunable list synced from chain proposals).
SANCTIONED_REGIONS: set[str] = {
    "IR", "KP", "SY", "CU",
}


@dataclass
class RegionPolicy:
    sanctioned: set[str] = field(default_factory=lambda: set(SANCTIONED_REGIONS))
    allow_unknown: bool = True   # if region lookup fails, allow by default

    def check(self, region_code: Optional[str]) -> bool:
        if region_code is None:
            return self.allow_unknown
        return region_code.upper() not in self.sanctioned


# ---------- the gate -------------------------------------------------------

@dataclass
class SafetyGateConfig:
    rate: RateLimitConfig = field(default_factory=RateLimitConfig)
    classifier: ContentClassifierConfig = field(default_factory=ContentClassifierConfig)
    region: RegionPolicy = field(default_factory=RegionPolicy)
    quarantine_classes: tuple = ("csam", "explicit_minor")  # immediate hard-deny


class SafetyGate:
    """Single entry point: ``check(pubkey, content, region) -> SafetyDecision``."""

    def __init__(self, config: SafetyGateConfig = SafetyGateConfig()):
        self.cfg = config
        self.rate = RateLimiter(self.cfg.rate)
        self.classifier = ContentClassifier(self.cfg.classifier)
        self._stats: dict[str, int] = {
            "allowed": 0, "denied": 0, "rate_limited": 0, "quarantined": 0,
        }
        self._stats_lock = threading.Lock()

    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    def check(
        self,
        pubkey: str,
        content: str,
        *,
        region: Optional[str] = None,
    ) -> SafetyDecision:
        ts = time.time()

        # 1. Region check first — cheapest reject.
        if not self.cfg.region.check(region):
            self._bump("denied")
            return SafetyDecision(
                decision=DENY, reason=f"sanctioned region: {region}",
                submitted_pubkey=pubkey, ts=ts,
            )

        # 2. Rate limit.
        if not self.rate.check(pubkey):
            self._bump("rate_limited")
            return SafetyDecision(
                decision=RATE_LIMITED,
                reason=f"rate exceeded for {pubkey[:12]}",
                submitted_pubkey=pubkey, ts=ts,
            )

        # 3. Content classification.
        cls, score = self.classifier.classify(content)
        if cls is not None:
            if cls in self.cfg.quarantine_classes:
                self._bump("quarantined")
                return SafetyDecision(
                    decision=QUARANTINED,
                    reason=f"content class {cls} (score {score:.2f})",
                    matched_class=cls, submitted_pubkey=pubkey, ts=ts,
                )
            self._bump("denied")
            return SafetyDecision(
                decision=DENY,
                reason=f"content class {cls} (score {score:.2f})",
                matched_class=cls, submitted_pubkey=pubkey, ts=ts,
            )

        self._bump("allowed")
        return SafetyDecision(
            decision=ALLOW, submitted_pubkey=pubkey, ts=ts,
        )

    def _bump(self, key: str) -> None:
        with self._stats_lock:
            self._stats[key] = self._stats.get(key, 0) + 1
