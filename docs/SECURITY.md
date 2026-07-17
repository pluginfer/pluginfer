# Pluginfer security posture

> Last reviewed: 2026-05-08 — CP-5 of the v1.0 production launch.

## Threat model

Pluginfer is a peer-to-peer GPU compute marketplace. Attackers we
defend against:

| Adversary           | Capabilities                                         | Worst-case outcome we prevent                     |
| ------------------- | ---------------------------------------------------- | -------------------------------------------------- |
| LAN attacker        | Send arbitrary UDP/TCP to mesh nodes                | Hijack the mesh by spoofing announces             |
| Remote attacker     | Hit the REST API, run arbitrary clients             | Forge auth, replay txs, spam mempool              |
| Malicious provider  | Win an auction with a real bid, then misbehave      | Get paid for a wrong / forged result              |
| Malicious peer      | Send forged blocks / txs / gossip envelopes        | Corrupt local state, mint coins                   |
| Filesystem attacker | Modify `~/.pluginfer/*` between runs                | Drain wallet, forge auth tokens                   |
| Build-supply chain  | Tamper with downloaded installer                    | Push backdoored binary to users                   |

## Static security audit (CP-5)

Run from the repo root with our pinned config:

```sh
python -m bandit -r v2/core v2/api -ll -c .bandit
```

**Current state (2026-05-08, commit `<HEAD>`):**

| Severity | Count | Notes                                                                   |
| -------- | ----- | ----------------------------------------------------------------------- |
| HIGH     | 0     | SHA1 truncation in `kademlia.py` replaced with SHA256[:20] (CP-5)       |
| MEDIUM   | 0     | Pickle in `advanced_mesh_features.py` replaced with JSON+sha256 envelope |
| LOW      | 51    | Standard hardening tips; reviewed and benign                             |

The `.bandit` config skips three rules with a justification per skip:

- **B102 exec** — `secure_sandbox.py` runs untrusted user code in a
  prebuilt whitelist namespace (W22 hardening). `exec` *is* the
  feature; the hardening is documented in
  `core/secure_sandbox.py`.
- **B104 0.0.0.0** — Mesh node listeners must accept all interfaces.
  Binding to a single IP would break DHCP and any host with multiple
  NICs.
- **B310 urllib** — `core/updater.py` only fetches https:// URLs and
  ECDSA-verifies the manifest signature **before** trusting any
  downloaded artefact (W31, commit `94f25c2`).

## Replay & tamper protection

Closed in CP-5:

- **Tx replay** (sec3.3): per-sender monotonic nonce; tx_id pre-image
  includes the nonce; mempool enforces strictly-increasing nonces;
  remote-block validator re-derives tx_id from declared fields and
  rejects mismatches.
- **Tx-after-sign mutation** (CP-5 finding): `add_transaction` now
  recomputes the tx_id from current field values and refuses if it
  disagrees with the stored id. Closes a local-state-pollution gap
  where a swapped recipient could pass `Wallet.verify` because the
  stored tx_id was stale.
- **Gossip replay** (W24): every envelope is ECDSA-signed; receivers
  dedup via `(origin, id)` + verify signature → `is_new=False` on
  re-broadcast.
- **Auth replay**: API auth challenges expire in 30s and are
  one-shot; sessions live in an in-memory dict with explicit TTL.

## Fault injection

`tests/fault_injection/` covers:

- `test_byzantine_auction.py` — negative price, eta out of range,
  quality > 1, raising provider, empty provider set.
- `test_malicious_provider.py` — lying provider hash, null result,
  exception during execute.
- `test_network_partition.py` — diverged chains heal without
  double-credit on the recipient.
- `test_node_crash_recovery.py` — save/load round-trips balances and
  nonces; pending pool is volatile.
- `test_seed_node_down.py` — `BOOTSTRAP_SEEDS` empty in the source
  repo (production fills it); persistence helpers exist.

## Wallet & key handling

- Wallet at rest: `core/tokenomics.py` refuses to write an unencrypted
  wallet (W31 enforcement). Passphrase comes from
  `PLUGINFER_WALLET_PASSPHRASE`, OS keychain, or explicit param.
- Release manifest signed by `PLUGINFER_RELEASE_PRIVKEY_PEM` (kept
  off the developer's laptop, in CI secrets only). Verified by
  `core/updater.py` against the public key baked at build time.
- API auth keys: only `sha256(key)` is stored; raw key is shown to
  the user once at issuance and never persisted.

## Dependency CVE scanning

Run via `pip-audit` with the pinned production dep list (`requirements-prod.txt`).
The full developer environment includes many vendored dirs and ML
frameworks that fall outside the production runtime; CI scans only
the production set.

## Compliance + regulatory posture (G5/G6/G7/G11)

Pluginfer is a US-founder-operated marketplace that settles USD
between pseudonymous wallets across borders. The compliance posture
matches that exposure:

- **OFAC sanctions screening** (`core/compliance/sanctions.py`,
  G5) — every buyer + provider wallet is matched against the
  OFAC SDN list + EU CFSP + UN Consolidated lists at auction time.
  Source-IP country code is matched against the comprehensive-
  sanctions jurisdictions (CU/IR/KP/SY/RU/BY + Crimea/Donetsk/
  Luhansk sub-codes). A match returns `451 Unavailable for Legal
  Reasons` + a compliance event in the append-only audit log.
  See `docs/AML_POLICY.md` for the operator-facing programme.
- **Sybil resistance** (`core/sybil_guard.py`, G6) — three
  stacking defences on the browser-tab provider gateway:
  per-/24 token bucket, WebGPU adapter-fingerprint Sybil
  detection, stake-to-register tier promotion (untrusted /
  staked / verified).
- **Per-inference energy + carbon disclosure** (`core/energy.py`,
  G7) — every completed §A1 PNIS receipt stamps `energy_mj` +
  `carbon_gco2e` + ISO-3166 zone + intensity. EU AI Act 2026 +
  SEC AI disclosure 2027 ready out of the box.
- **Tax reporting** (`core/tax/reporting.py`, G11) — annual
  1099-NEC (US ≥ $600), GSTR-1 (India), EU VAT-MOSS draft CSVs
  emitted from the receipt log.

## Browser-tab provider threat model

Adding the §A21 browser-tab provider expands the attack surface:

| Adversary           | New capabilities                    | Mitigation                                      |
| ------------------- | ----------------------------------- | ----------------------------------------------- |
| Mass-Sybil tab farm | Spin 10k headless tabs from one IP  | per-/24 rate limit (G6); fingerprint Sybil block |
| Job-refusal attacker | Win a job, never deliver          | execute() times out at 30s -> auction releases escrow; staked tier loses deposit (G6) |
| Lying provider       | Return wrong bytes, hope nobody verifies | Receipt sidecar binds upstream signature + result_hash; cross-quorum re-execution on dispute (§A9) |
| Sanctioned-region operator | Register from blocked country | CF-IPCountry-driven region screen at /register (G5) |

## Audit-readiness package

For external auditors (Trail of Bits / NCC / Halborn / Code4rena):

    python -m tools.audit_prep --out audit_package.zip

The tool collects the crypto surface (`core/pedersen.py`,
`core/bft_consensus.py`, `core/slash_evidence.py`, `core/compute_ledger.py`,
`core/smart_contracts.py`, `core/wasm_executor.py`,
`core/ai_receipt.py`, `core/secure_sandbox.py`,
`core/tokenomics.py`) into a single tarball with file SHA-256s and
a CONTENTS.md table-of-claims. Auditors get exactly the load-bearing
code without the noise of the rest of the repo.

## Responsible disclosure

Email `security@pluginfer.network` with reproducer + your PGP public
key. We will respond within 72 hours and coordinate disclosure on a
shared timeline. See `docs/RESPONSIBLE_DISCLOSURE.md` for the full
policy + bounty pre-spec. Bounties are tracked in
`docs/SECURITY-BOUNTIES.md` (post-launch).
