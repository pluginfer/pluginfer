"""G9 — audit-readiness package builder.

Bundles the cryptographic surface of Pluginfer into a single ZIP
archive an auditor can `unzip + cd + read` without needing to know
the rest of the repo. Each file gets a SHA-256; the bundle ships
with a CONTENTS.md table mapping {filename -> SHA-256 -> claim} so
the auditor can confirm they're reading exactly what production runs.

Usage:

    python -m tools.audit_prep --out audit_package.zip
    python -m tools.audit_prep --out audit_package.zip --include-tests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]


# (module path relative to v2/) -> (claim the auditor should validate)
CRYPTO_SURFACE = [
    ("core/pedersen.py",
     "Pedersen commitments + Schnorr PoK over SECP256K1. Validate: "
     "scalar math, bit-OR proof soundness, side-channel resistance."),
    ("core/bft_consensus.py",
     "Tendermint-style stake-weighted BFT. Validate: 2/3 quorum, "
     "equivocation detection, view-change correctness."),
    ("core/slash_evidence.py",
     "BFT slash evidence (W32). Validate: forged-attestation rejection, "
     "unbonding-lock semantics, stake-snapshot correctness."),
    ("core/compute_ledger.py",
     "PoW chain + state machine. Validate: fork resolution, "
     "supply cap, replay-nonce, tx-validation completeness."),
    ("core/smart_contracts.py",
     "Chain-derived smart contracts (W23). Validate: "
     "deterministic address derivation, signature checks, "
     "reorg invalidation of contract state."),
    ("core/wasm_executor.py",
     "wasmtime sandbox. Validate: capability denial, fuel "
     "metering, module cache integrity."),
    ("core/secure_sandbox.py",
     "Python AST + process isolation. Validate: builtins "
     "whitelist, dunder access blocks, process-kill on timeout."),
    ("core/ai_receipt.py",
     "PNIS-Receipt v1 schema. Validate: canonical JSON, "
     "signature binding, tamper detection."),
    ("core/tokenomics.py",
     "ECDSA SECP256K1 wallet + Tx hashing. Validate: tx hash "
     "domain separation, address derivation, at-rest encryption."),
    ("core/staking.py",
     "Stake snapshot for BFT quorum. Validate: slashed-validator "
     "exclusion, unbonding semantics."),
    ("core/kademlia.py",
     "Routing layer. Validate: LRU ping-and-evict, sha256[:20] "
     "node-id derivation."),
    ("core/payments.py",
     "Stripe + idempotency. Validate: replay safety, "
     "double-charge prevention."),
    ("core/anchored_bootstrap.py",
     "Bitcoin-anchored seed permutation (§A10). Validate: "
     "non-forgeable ordering, signature filter correctness."),
    ("core/seed_registry_builder.py",
     "TOFU + quorum promotion for seed registry. Validate: "
     "single-signer flag, distinct-signer counting."),
    ("core/sybil_guard.py",
     "Sybil resistance (G6). Validate: token bucket math, "
     "fingerprint window semantics, tier resolver."),
    ("core/compliance/sanctions.py",
     "OFAC/EU/UN screen (G5). Validate: address derivation "
     "matches Wallet.generate_address, list reload safety."),
    ("api/jobs_service.py",
     "Gateway attestation pipeline. Validate: "
     "attest_receipt input/output binding, signature integrity."),
    ("api/devserver.py",
     "OpenAI/Anthropic shim (§A21). Validate: header-bound "
     "receipt-id propagation, SSE stream termination, "
     "auth identity scoping."),
    ("api/routers/provider_jobs.py",
     "Browser-tab gateway endpoints (G6 + G5). Validate: "
     "sanctions denial, sybil-guard ordering, rate-limit semantics."),
    ("api/routers/receipts.py",
     "Public receipts leaderboard. Validate: JIT attestation "
     "fallback, signed-payload precedence."),
]

TESTS_SURFACE = [
    "tests/test_chain_integrity.py",
    "tests/test_bft_at_scale.py",
    "tests/test_slash_evidence.py",
    "tests/test_wasm_executor.py",
    "tests/test_smart_contract.py",
    "tests/test_payments_idempotency.py",
    "tests/test_storage_sqlite.py",
    "tests/test_provider_auction.py",
    "tests/test_devserver_shim.py",
    "tests/test_devserver_receipt_w49.py",
    "tests/test_provider_jobs_roundtrip.py",
    "tests/test_compliance_sanctions.py",
    "tests/test_sybil_guard.py",
    "tests/test_energy_accounting.py",
    "tests/test_devserver_streaming_g8.py",
    "tests/test_tax_reporting.py",
    "tests/test_flagship.py",
    "tests/test_seed_registry_builder.py",
    "tests/test_cp_final_realwire.py",
    "tests/fault_injection/test_byzantine_auction.py",
    "tests/fault_injection/test_malicious_provider.py",
    "tests/fault_injection/test_network_partition.py",
    "tests/fault_injection/test_node_crash_recovery.py",
    "tests/fault_injection/test_seed_node_down.py",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_audit_package(
    out_path: Path, *, include_tests: bool = True,
) -> dict:
    """Build the audit ZIP. Returns a manifest dict suitable for
    pretty-printing in CI logs."""
    files = [(p, claim) for p, claim in CRYPTO_SURFACE]
    if include_tests:
        for p in TESTS_SURFACE:
            files.append((p, "test asset"))

    manifest = {
        "schema": "pluginfer-audit-package/v1",
        "generated_at_unix": time.time(),
        "repo_root_relative": "v2",
        "contents": [],
    }

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for rel_path, claim in files:
            src = V2 / rel_path
            if not src.exists():
                manifest["contents"].append({
                    "path": rel_path, "status": "MISSING", "claim": claim,
                })
                continue
            sha = _sha256_file(src)
            z.write(src, arcname=rel_path)
            manifest["contents"].append({
                "path": rel_path,
                "sha256": sha,
                "size_bytes": src.stat().st_size,
                "claim": claim,
            })

        # Write the contents table as a Markdown file inside the ZIP
        # so auditors get it without unpacking. Belt + suspenders also
        # emit a CONTENTS.json (machine-readable).
        md_lines = [
            "# Pluginfer audit package — CONTENTS",
            "",
            f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}.",
            "",
            "| File | SHA-256 | Claim |",
            "| ---- | ------- | ----- |",
        ]
        for entry in manifest["contents"]:
            if "sha256" not in entry:
                continue
            md_lines.append(
                f"| `{entry['path']}` | `{entry['sha256']}` | "
                f"{entry['claim']} |"
            )
        z.writestr("CONTENTS.md", "\n".join(md_lines))
        z.writestr("CONTENTS.json", json.dumps(manifest, indent=2, sort_keys=False))

    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Bundle the crypto surface for auditors.")
    ap.add_argument("--out", default="audit_package.zip")
    ap.add_argument("--no-tests", action="store_true")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    manifest = build_audit_package(
        out_path=out, include_tests=not args.no_tests,
    )
    counted = sum(1 for c in manifest["contents"] if "sha256" in c)
    missing = sum(1 for c in manifest["contents"] if c.get("status") == "MISSING")
    print(json.dumps({
        "out": str(out),
        "files_bundled": counted,
        "missing": missing,
    }, indent=2))
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
