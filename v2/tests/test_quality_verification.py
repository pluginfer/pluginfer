"""§F4 Adversarial quality-verification tests.

Covers:
* maybe_audit selects with configured probability
* compare_outcome computes cosine similarity correctly
* Honest providers (high cosine) accumulate clean records
* Sybil (random noise) gets challenged + slashed after threshold
* Single-batch noise doesn't slash an honest provider
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_grain(contributor: str, payload: bytes,
                 m: int = 64, n: int = 32, version: int = 0):
    from ai.filum.hpa.grain import Grain, GrainMeta
    g = Grain(meta=GrainMeta(
        model_shard_id="L0",
        version_v=version,
        contributor_id=contributor,
        optimizer_seed=42,
        shape_m=m, shape_n=n,
    ))
    g.grad_bytes = payload
    g.meta.grain_id = g.compute_grain_id()
    return g


def _array_to_bytes(arr) -> bytes:
    return arr.astype("float32").tobytes()


def test_maybe_audit_respects_audit_rate():
    """At rate=1.0 every grain is audited; at rate=0.0 none are."""
    from ai.filum.hpa.quality_verification import (
        AdversarialQualityVerifier, QualityVerificationConfig,
    )

    sun_pool = lambda: [type("P", (), {"node_id": f"sun{i}"})()
                          for i in range(5)]
    v_all = AdversarialQualityVerifier(
        QualityVerificationConfig(audit_rate=1.0),
        sun_pool_fn=sun_pool,
    )
    v_none = AdversarialQualityVerifier(
        QualityVerificationConfig(audit_rate=0.0),
        sun_pool_fn=sun_pool,
    )
    g = _make_grain("alice", _array_to_bytes(np.zeros(8 * 8)),
                     m=8, n=8)
    assert v_all.maybe_audit(g) is True
    assert v_none.maybe_audit(g) is False


def test_compare_outcome_high_cosine_passes():
    """Two near-identical gradients pass without challenge."""
    from ai.filum.hpa.quality_verification import (
        AdversarialQualityVerifier, QualityVerificationConfig,
    )

    sun_pool = lambda: [type("P", (), {"node_id": "sun1"})()]
    v = AdversarialQualityVerifier(
        QualityVerificationConfig(audit_rate=1.0,
                                    challenge_threshold=0.85,
                                    audit_pool_min_size=2),
        sun_pool_fn=sun_pool,
    )
    rng = np.random.default_rng(7)
    base = rng.standard_normal(64).astype("float32")
    # Auditor gets the *same* gradient with tiny float noise.
    noisy = base + rng.standard_normal(64).astype("float32") * 0.001

    orig = _make_grain("alice", _array_to_bytes(base), m=8, n=8)
    audit = _make_grain("sun1", _array_to_bytes(noisy), m=8, n=8)

    v.maybe_audit(orig)
    outcome = v.compare_outcome(orig.meta.grain_id, audit)
    assert outcome is not None
    assert outcome.cosine_similarity > 0.95
    assert outcome.challenged is False


def test_compare_outcome_random_noise_is_challenged():
    """A Sybil submits random noise; the auditor's real gradient
    diverges; cosine ~0; challenge fires."""
    from ai.filum.hpa.quality_verification import (
        AdversarialQualityVerifier, QualityVerificationConfig,
    )

    sun_pool = lambda: [type("P", (), {"node_id": "sun1"})()]
    v = AdversarialQualityVerifier(
        QualityVerificationConfig(audit_rate=1.0,
                                    challenge_threshold=0.85,
                                    audit_pool_min_size=2),
        sun_pool_fn=sun_pool,
    )
    rng = np.random.default_rng(11)
    sybil_grad = rng.standard_normal(64).astype("float32")
    real_grad = rng.standard_normal(64).astype("float32")

    orig = _make_grain("malice", _array_to_bytes(sybil_grad), m=8, n=8)
    audit = _make_grain("sun1", _array_to_bytes(real_grad), m=8, n=8)

    v.maybe_audit(orig)
    outcome = v.compare_outcome(orig.meta.grain_id, audit)
    assert outcome is not None
    assert outcome.cosine_similarity < 0.5
    assert outcome.challenged is True


def test_three_consecutive_challenges_trigger_slash():
    """Slashing fires only after consecutive_for_slash repeated failures."""
    from ai.filum.hpa.quality_verification import (
        AdversarialQualityVerifier, QualityVerificationConfig,
    )

    slashed: list[tuple[str, float]] = []
    sun_pool = lambda: [type("P", (), {"node_id": "sun1"})()]
    v = AdversarialQualityVerifier(
        QualityVerificationConfig(
            audit_rate=1.0,
            challenge_threshold=0.85,
            slash_threshold=0.50,
            consecutive_for_slash=3,
            slash_multiplier=5.0,
            audit_pool_min_size=2,
        ),
        sun_pool_fn=sun_pool,
        slash_fn=lambda pk, tflop: slashed.append((pk, tflop)),
    )
    rng = np.random.default_rng(3)

    # Three back-to-back fakes from the same provider.
    for _ in range(3):
        sybil = rng.standard_normal(64).astype("float32")
        real = rng.standard_normal(64).astype("float32")
        orig = _make_grain("malice", _array_to_bytes(sybil), m=8, n=8,
                            version=_)
        audit = _make_grain("sun1", _array_to_bytes(real), m=8, n=8,
                              version=_)
        v.maybe_audit(orig)
        v.compare_outcome(orig.meta.grain_id, audit)
    assert len(slashed) == 1
    assert slashed[0][0] == "malice"
    assert slashed[0][1] > 0.0


def test_single_failed_audit_does_not_slash():
    """One bad batch (e.g. numerical noise on a real provider) doesn't
    slash. The provider's consecutive_challenges resets when they pass
    the next audit."""
    from ai.filum.hpa.quality_verification import (
        AdversarialQualityVerifier, QualityVerificationConfig,
    )

    slashed: list = []
    sun_pool = lambda: [type("P", (), {"node_id": "sun1"})()]
    v = AdversarialQualityVerifier(
        QualityVerificationConfig(
            audit_rate=1.0,
            challenge_threshold=0.85,
            slash_threshold=0.50,
            consecutive_for_slash=3,
            audit_pool_min_size=2,
        ),
        sun_pool_fn=sun_pool,
        slash_fn=lambda pk, tflop: slashed.append((pk, tflop)),
    )
    rng = np.random.default_rng(5)

    # 1 fail, then 2 passes, then 1 more fail. Should NOT slash because
    # consecutive_challenges resets after each pass.
    for kind in ("fail", "pass", "pass", "fail"):
        if kind == "fail":
            orig_g = rng.standard_normal(64).astype("float32")
            audit_g = rng.standard_normal(64).astype("float32")
        else:
            base = rng.standard_normal(64).astype("float32")
            orig_g = base
            audit_g = base + rng.standard_normal(64).astype("float32") * 0.001
        orig = _make_grain("provider", _array_to_bytes(orig_g),
                            m=8, n=8, version=hash(kind))
        audit = _make_grain("sun1", _array_to_bytes(audit_g),
                              m=8, n=8, version=hash(kind))
        v.maybe_audit(orig)
        v.compare_outcome(orig.meta.grain_id, audit)
    assert slashed == []


def test_compare_outcome_unknown_grain_returns_none():
    from ai.filum.hpa.quality_verification import AdversarialQualityVerifier

    v = AdversarialQualityVerifier()
    audit = _make_grain("sun1", _array_to_bytes(np.zeros(8)), m=8, n=1)
    out = v.compare_outcome("not-a-real-grain-id", audit)
    assert out is None


def test_stats_aggregates():
    from ai.filum.hpa.quality_verification import (
        AdversarialQualityVerifier, QualityVerificationConfig,
    )

    sun_pool = lambda: [type("P", (), {"node_id": "sun1"})()]
    v = AdversarialQualityVerifier(
        QualityVerificationConfig(audit_rate=1.0, audit_pool_min_size=2),
        sun_pool_fn=sun_pool,
    )
    rng = np.random.default_rng(7)
    base = rng.standard_normal(64).astype("float32")
    orig = _make_grain("p", _array_to_bytes(base), m=8, n=8)
    audit = _make_grain("sun1", _array_to_bytes(base), m=8, n=8)
    v.maybe_audit(orig)
    v.compare_outcome(orig.meta.grain_id, audit)
    s = v.stats()
    assert s["audits_total"] == 1
    assert s["audits_challenged"] == 0
    assert s["pending_audits"] == 0
