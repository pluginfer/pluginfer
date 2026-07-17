"""Generate a signed seed_registry.json from operator-published seed
pubkeys.

Usage:
    python -m deploy.generate_seed_registry \
        --validator-key /etc/pluginfer/validator_a.pem \
        --validator-key /etc/pluginfer/validator_b.pem \
        --seed seed-eu.pluginfer.network:9000:<pubkey-pem-path> \
        --seed seed-us.pluginfer.network:9000:<pubkey-pem-path> \
        --seed seed-sg.pluginfer.network:9000:<pubkey-pem-path> \
        > seed_registry.json

Each seed record gets signed by every validator key supplied; the
resulting `quorum_signatures` list satisfies `min_signatures` in
the registry consumer.

When this lands, `core/seed_registry.SeedRegistry.trusted_records`
will return all three seeds without operator overrides; auto_mesh
clients pulling the bundled registry get an honest, signed list
they can trust.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))


def _load_pubkey_pem(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def _load_wallet(path: str, passphrase: bytes):
    from core.tokenomics import Wallet
    return Wallet(filename=path, passphrase=passphrase)


def _signing_message(host: str, port: int, pubkey_fp: str) -> str:
    return f"SEED|{host}|{port}|{pubkey_fp}"


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--validator-key", action="append", default=[],
        help="Path to a validator wallet PEM. Repeat for each "
             "validator participating in quorum signing.",
    )
    ap.add_argument(
        "--validator-passphrase-env", default="PLUGINFER_VALIDATOR_PASSPHRASE",
        help="Env var holding the validator wallet passphrase.",
    )
    ap.add_argument(
        "--seed", action="append", default=[],
        help="Format: HOST:PORT:PUBKEY_PEM_PATH",
    )
    ap.add_argument(
        "--min-signatures", type=int, default=2,
        help="How many validator signatures a record needs to be "
             "trusted (default 2).",
    )
    args = ap.parse_args(argv)

    import os
    passphrase = os.environ.get(
        args.validator_passphrase_env, "",
    ).encode("utf-8")
    validators = [_load_wallet(p, passphrase) for p in args.validator_key]

    records = []
    for entry in args.seed:
        parts = entry.split(":")
        if len(parts) != 3:
            raise SystemExit(f"bad --seed format: {entry}")
        host, port, pubkey_path = parts[0], int(parts[1]), parts[2]
        pubkey_pem = _load_pubkey_pem(pubkey_path)
        fp = hashlib.sha256(pubkey_pem.encode("utf-8")).hexdigest()
        msg = _signing_message(host, port, fp)
        sigs = []
        for v in validators:
            sigs.append({
                "signer_fingerprint_sha256": hashlib.sha256(
                    v.public_key_pem.encode("utf-8")
                ).hexdigest(),
                "signature_b64": v.sign(msg),
            })
        records.append({
            "id": f"seed-{host}",
            "host": host,
            "port": port,
            "region": "auto",
            "pubkey_fingerprint_sha256": fp,
            "quorum_signatures": sigs,
        })

    out = {
        "schema": "pluginfer-seed-registry/v2",
        "min_signatures": args.min_signatures,
        "tofu_mode": False,
        "records": records,
        "operator_overrides": {
            "env_seed_host": "PLUGINFER_SEED_HOST",
            "env_seed_port": "PLUGINFER_SEED_PORT",
        },
    }
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
