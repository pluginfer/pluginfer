"""When all bootstrap seeds are unreachable, the node MUST fall back
to a persisted peers.json (CP-2 contract). Otherwise a single-seed
outage takes the entire mesh down on next restart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[2]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))


def test_bootstrap_seeds_module_attr_present(tmp_path: Path):
    """The controller exposes BOOTSTRAP_SEEDS at module scope (CP-2
    contract) and _bootstrap_from_seeds + _persist_peers as instance
    methods. If a refactor renames either, the seed-fallback path
    silently regresses -- this test catches that."""
    from core import complete_mesh_controller as cmc
    assert hasattr(cmc, "BOOTSTRAP_SEEDS"), "CP-2 BOOTSTRAP_SEEDS missing"
    assert isinstance(cmc.BOOTSTRAP_SEEDS, list)
    cls = cmc.CompleteMeshController if hasattr(cmc, "CompleteMeshController") else None
    assert cls is not None, "CompleteMeshController class missing"
    for name in ("_bootstrap_from_seeds", "_persist_peers"):
        assert hasattr(cls, name), f"CompleteMeshController.{name} missing"


def test_bootstrap_seeds_empty_in_default_build():
    """The shipping repo MUST have an empty BOOTSTRAP_SEEDS list -- ops
    fills it post-deploy. Hardcoded production IPs in source = lock-in."""
    from core.complete_mesh_controller import BOOTSTRAP_SEEDS
    assert BOOTSTRAP_SEEDS == [], (
        "BOOTSTRAP_SEEDS must be empty in the source repo (production "
        "fills it via env var or config file -- never source-baked IPs)."
    )
