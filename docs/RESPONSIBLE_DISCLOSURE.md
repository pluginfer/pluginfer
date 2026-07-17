# Pluginfer — Responsible Disclosure & Pre-Launch Bounty Spec

> Document owner: Pluginfer security team. Last updated: 2026-05-13.

## How to report

Email **`security@pluginfer.network`** with:

1. A clear reproducer (commands, request bodies, expected vs actual).
2. The commit SHA you tested against.
3. Your contact info + PGP public key (we'll encrypt sensitive
   follow-up).
4. Suggested CVSS rating, if you've got one.

PGP key fingerprint: published at `https://pluginfer.network/.well-known/security.txt`
when DNS is live. Until then, request the key in the first email and
we'll reply with it through a signed channel.

We acknowledge **within 72 hours** of receipt. We aim to ship a fix
or mitigation within:

- **Critical** (chain corruption, fund theft, RCE on the gateway): **7 days**.
- **High** (signature forgery, escalation across tiers, leak of
  wallet material): **30 days**.
- **Medium/Low**: **90 days**.

## Scope (in scope)

* Pluginfer core (`v2/core/`) — chain, BFT, slash-evidence, wallet,
  Pedersen, smart contracts, WASM sandbox, kademlia, gossip,
  payments, staking, providers, ai_receipt.
* Pluginfer API (`v2/api/`) — REST endpoints, middleware, devserver
  shim, browser-provider gateway, receipts router.
* Build pipeline (`v2/build/`, `.github/workflows/`).
* Browser-tab provider (`v2/ui/browser_provider/`).
* PNIS receipt spec — anything that lets a forged receipt verify
  under `AIReceipt.verify()`.

## Scope (out of scope)

* Self-signed dev certs (`tools/dev_cert.py`) — these are
  documented as not-for-production.
* Third-party LLM API providers (OpenAI, Anthropic, Gemini) — report
  directly to those vendors.
* Sanctions-list staleness — that's a process bug, not a
  vulnerability. Email `compliance@pluginfer.network`.
* Self-DoS scenarios (your laptop, your network, your problem).

## Bounty pre-spec (post-launch)

Until external funding lands, bounty payouts are **goodwill** — a
public credit + a Pluginfer wallet airdrop of PLG sized to the
severity, plus a coordinated-disclosure CVE. After funding, we'll
publish a tiered cash bounty matching the table below and run it
through Code4rena / Sherlock for the major-version pre-launch
audit window.

| Severity | Description                                                                     | Pre-launch reward                                | Post-launch USD target |
| -------- | ------------------------------------------------------------------------------- | ------------------------------------------------ | ----------------------- |
| Critical | Chain corruption; fund theft from any wallet; RCE on the gateway; receipt forgery that passes `AIReceipt.verify()`. | Up to 10,000 PLG + public CVE credit             | $20k–$50k                |
| High     | Signature forgery; escalation across provider tiers; leak of any wallet material at rest. | Up to 3,000 PLG + CVE credit                     | $5k–$15k                 |
| Medium   | Denial-of-service that's expensive to mitigate; sanctions screen bypass; tax/compliance data leak. | Up to 1,000 PLG + CVE credit                     | $1k–$3k                  |
| Low      | Information leaks not exploitable for fund/data theft; misconfiguration risks.    | Public credit + small thank-you airdrop          | $100–$500                |

## Safe-harbour

We will not pursue legal action against good-faith security research
that:

1. Stays within the scope above.
2. Does not exfiltrate user data beyond the minimum needed to
   demonstrate the vulnerability.
3. Does not degrade service for other users (no public DoS).
4. Does not extort us — engage with us, give us reasonable time to
   patch, and we'll coordinate the public disclosure together.

This safe-harbour mirrors the disclose.io baseline and applies to
any researcher acting in good faith.
