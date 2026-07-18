# Security Policy

Pluginfer moves (test-)money and runs jobs on strangers' machines, so
security reports are taken seriously and handled fast.

## Reporting a vulnerability

- **Preferred:** GitHub → the repo's **Security** tab → **Report a
  vulnerability** (private advisory; only maintainers see it).
- Please include reproduction steps and the commit/version you tested.

Please do **not** open a public issue for anything exploitable —
especially anything touching the ledger (`core/buyer_ledger.py`,
`core/payment_flows.py`), receipts/signatures, the swarm-key gate
(`core/swarm_auth.py`), or the release pipeline.

## What you can expect

- Acknowledgement within 72 hours.
- An honest assessment — if it's real, it gets fixed and credited to
  you in the release notes (or kept anonymous if you prefer).
- No legal threats for good-faith research. Testing against **your own
  nodes** is always in scope; don't test against nodes you don't own.

## Scope notes

- Testnet balances are explicitly not redeemable for cash; ledger
  integrity still matters and reports are welcome.
- The swarm key is a shared symmetric secret by design (documented in
  `core/swarm_auth.py`) — reports should target bypasses of its stated
  guarantees, not the absence of per-node PKI (a tracked milestone).
