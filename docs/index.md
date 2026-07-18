# Pluginfer

Two products, one repo:

- **Signet** — a fail-closed AI spend gateway: hard budget caps
  (overruns become HTTP 402, not a bill), Ed25519 hash-chained
  receipts (optionally anchored in Bitcoin), multi-LLM routing, and
  measured-only savings.
- **The mesh** — a compute marketplace where jobs go to a sealed-bid
  auction across peer GPUs, with quorum verification, stake +
  slashing economics, and a self-auditing **USD** money ledger.
  Testnet now; **no token, ever** — no emissions, no presale, payouts
  only from real buyer payments.

Docs:

- [Quickstart](quickstart.md) — 5 minutes from zero to a running node
  with free testnet credits and a signed job
- [Setup guides](SETUP_GUIDES.md) — every shipped feature, step by step
- [Architecture](architecture.md) — the full system design
- [API reference](api-reference.md) — REST endpoints
- [Security](SECURITY.md) — threat model + audit posture
- [Responsible disclosure](RESPONSIBLE_DISCLOSURE.md) — how to report
  vulnerabilities (and what we honestly offer)
- [Signing setup](SIGNING_SETUP.md) — release verification + the
  code-signing certs still pending
- [Changelog](CHANGELOG.md) — release summary index

Read [`AUDIT.md`](../AUDIT.md) first if you want the self-critical
version: what's proven, what's mitigated, and what's still open.

## Why

Centralised LLM API providers run a vertical stack: their hardware,
their model, their margin. Pluginfer's auction puts peer GPUs running
open-weight models (in users' off-peak hours) against those APIs, on
identical cost / latency / privacy / quality constraints set by the
caller. As the mesh grows, routine workloads route to the cheapest
qualifying provider — and every job settles with a signed,
independently verifiable receipt.

This is the cost-dynamics shift Pluginfer is targeting.
