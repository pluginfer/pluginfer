"""Seed registry loader — multi-seed quorum bundle, env override,
reachability probe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import pytest

from core.seed_registry import SeedRecord, SeedRegistry, resolve_bootstrap_seed


def _write_bundle(tmp_path, *, records, min_sigs=2, tofu=False):
    p = tmp_path / "seed_registry.json"
    p.write_text(json.dumps({
        "schema": "pluginfer-seed-registry/v2",
        "min_signatures": min_sigs,
        "tofu_mode": tofu,
        "records": records,
    }))
    return str(p)


def test_loads_multi_seed_bundle(tmp_path):
    p = _write_bundle(tmp_path, records=[
        {"id": "eu", "host": "seed-eu.example", "port": 9000, "region": "eu",
         "quorum_signatures": [{"signer_fingerprint_sha256": "v1"},
                                {"signer_fingerprint_sha256": "v2"}]},
        {"id": "us", "host": "seed-us.example", "port": 9000, "region": "us",
         "quorum_signatures": [{"signer_fingerprint_sha256": "v1"},
                                {"signer_fingerprint_sha256": "v2"},
                                {"signer_fingerprint_sha256": "v3"}]},
    ])
    reg = SeedRegistry.from_file(p)
    assert len(reg.records) == 2
    assert reg.min_signatures == 2


def test_quorum_filter_drops_undersigned_records(tmp_path):
    """Records below min_signatures don't get returned by
    trusted_records — refuses to bootstrap from an unsigned seed."""
    p = _write_bundle(tmp_path, records=[
        {"id": "ok", "host": "seed-ok.example", "port": 9000,
         "quorum_signatures": [{"signer_fingerprint_sha256": "v1"},
                                {"signer_fingerprint_sha256": "v2"}]},
        {"id": "unsigned", "host": "seed-bad.example", "port": 9000,
         "quorum_signatures": []},
    ], min_sigs=2)
    reg = SeedRegistry.from_file(p)
    trusted = reg.trusted_records()
    assert {r.id for r in trusted} == {"ok"}


def test_tofu_mode_skips_quorum_check(tmp_path):
    """When the bundle is still bootstrapping the validator set,
    tofu_mode=True returns every record."""
    p = _write_bundle(tmp_path, records=[
        {"id": "fresh", "host": "seed-fresh.example", "port": 9000,
         "quorum_signatures": []},
    ], tofu=True)
    reg = SeedRegistry.from_file(p)
    assert len(reg.trusted_records()) == 1


def test_env_override_takes_precedence_over_bundle(monkeypatch, tmp_path):
    p = _write_bundle(tmp_path, records=[
        {"id": "bundle", "host": "bundle.example", "port": 9000,
         "quorum_signatures": [{"signer_fingerprint_sha256": "v1"},
                                {"signer_fingerprint_sha256": "v2"}]},
    ])
    monkeypatch.setenv("PLUGINFER_SEED_HOST", "127.0.0.1")
    monkeypatch.setenv("PLUGINFER_SEED_PORT", "1234")
    out = resolve_bootstrap_seed(bundle_path=p)
    assert out == ("127.0.0.1", 1234)


def test_resolve_returns_none_when_nothing_reachable(monkeypatch, tmp_path):
    monkeypatch.delenv("PLUGINFER_SEED_HOST", raising=False)
    monkeypatch.delenv("PLUGINFER_SEED_PORT", raising=False)
    p = _write_bundle(tmp_path, records=[
        # Domain that doesn't resolve / port that won't open.
        {"id": "unreachable", "host": "192.0.2.1", "port": 1, "region": "x",
         "quorum_signatures": [{"signer_fingerprint_sha256": "v1"},
                                {"signer_fingerprint_sha256": "v2"}]},
    ])
    out = resolve_bootstrap_seed(bundle_path=p)
    assert out is None


def test_bundled_default_registry_parses():
    """The default bundle ships with three placeholder records but
    must still load cleanly. Operator-published `quorum_signatures`
    will gate the trusted_records result."""
    default_path = str(V2 / "data" / "seed_registry.json")
    reg = SeedRegistry.from_file(default_path)
    assert len(reg.records) >= 3
    regions = {r.region for r in reg.records}
    assert {"eu-west", "us-east", "ap-south"} <= regions
