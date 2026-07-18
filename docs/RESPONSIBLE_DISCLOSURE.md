# Responsible Disclosure

Pluginfer moves (test-)money and runs jobs on other people's machines,
so security reports are taken seriously and handled fast.

## How to report

Use **GitHub's private vulnerability reporting**: the repo's
**Security tab → Report a vulnerability**. It is enabled, private to
maintainers, and the channel we actually watch. Include a reproducer
(commands, request bodies, expected vs actual) and the commit or
version you tested.

We acknowledge within **72 hours** and aim to ship a fix or
mitigation within: **7 days** for critical (fund-accounting
corruption, RCE, receipt forgery that passes verification),
**30 days** for high (signature forgery, auth bypass, wallet-material
leak), **90 days** otherwise.

## Rewards — stated honestly

There is **no funded bounty program yet**, and this project does not
promise money it does not have. What a valid report gets today:

- a fix, fast, with your finding credited in the release notes and
  the repo's security acknowledgements (or anonymity if you prefer);
- a coordinated CVE where applicable.

There is no token and never will be, so nothing is ever paid in one.
If a funded cash bounty program launches later, it will be announced
in this file and on GitHub Releases **before** it applies.

## Scope

In scope: `v2/core/` (money ledger, economic layer, quorum, wallet,
sandboxing, mesh networking), `v2/api/` (node + devserver endpoints),
`v2/governance/` (Signet gateway, budgets, receipts, signing, auth),
`v2/tools/` (node runtime, swarm auth), the build pipeline
(`v2/build/`, `.github/workflows/`), and anything that lets a forged
receipt verify.

Out of scope: third-party LLM providers (report to those vendors),
self-DoS on your own hardware, and social engineering of maintainers.

## Safe harbour

We will not pursue legal action against good-faith research that
stays in scope, touches no more user data than needed to demonstrate
the issue, does not degrade service for others, and gives us
reasonable time to patch before public disclosure. This mirrors the
disclose.io baseline.
