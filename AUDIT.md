# Pluginfer — Third-Party Audit & Honest Status

_Last updated 2026-07-18. This file exists because the project makes
money- and trust-related claims, and a public repo invites scrutiny.
It states plainly what is real, what is fixed, and what is not — so no
reader is misled and no contributor rebuilds on a false premise._

## Summary verdict

Two products live here:

1. **Governance gateway** (`governance/`) — an on-prem, mesh-free
   drop-in in front of any OpenAI/Anthropic-compatible endpoint that
   enforces budgets, thrifts tokens, and produces a signed audit log.
   **Genuine and working.** It enters a crowded category (LiteLLM,
   Portkey, Helicone, Cloudflare AI Gateway). It is **not** a category
   redefinition; it is a competent, honestly-instrumented entrant with
   one differentiator worth pursuing (provable, billable savings).
2. **Compute mesh** (`core/`, `tools/`) — a decentralized marketplace
   for AI compute. **Ambitious and coherent; now proven across a real
   WAN, but not yet at scale.** Its hardest problem (trusting an
   anonymous node's output) is mitigated, not solved. A cross-internet
   two-node run has cleared (a home node behind NAT served a signed job
   submitted by a cloud runner — public `wan-proof` workflow); the
   still-open step is two *independent home networks* introduced by a
   hosted seed. Private meshes (one org's datacenters) are gated by a
   shared swarm key (`PLUGINFER_SWARM_KEY`) enforced in one middleware
   on every mesh surface — verified live: 401 for strangers/wrong keys
   through a real tunnel, keyed peer cleared a signed job. Stated
   limits: symmetric shared key, no per-node identity/revocation yet,
   TLS transport required (tunnel or reverse proxy).

## Shortfalls the audit named, and their status

| # | Shortfall | Status |
|---|-----------|--------|
| 1 | Gateway endpoints open — anyone reaching the port could spend the upstream key and read all spend | **FIXED** — `governance/auth.py`: client keys required to forward, read keys to view spend, admin keys to administer; fail-closed once configured; `main()` refuses a public bind with no auth |
| 2 | Receipts unsigned by default; "blockchain" verify recomputed from scratch so an operator could rewrite history | **FIXED** — `governance/signing.py`: Ed25519 signatures by default (publicly verifiable with the public key alone), HMAC fallback honestly labelled; `verify_chain` checks signatures + chain; the word "blockchain" is gone. The remaining insider gap (operator holds the signing key) is now closed by opt-in external anchoring: `PLUGINFER_GW_ANCHOR=ots` publishes the chain head to public OpenTimestamps calendars (→ Bitcoin); emitted `.ots` proofs verified byte-identical against the independent reference implementation, live, against 3 real calendars. Scope: proofs are *pending* until Bitcoin-batched (hours); anchoring is fail-open and off by default (airgap-safe) |
| 3 | Token holds sized by chars/4 | **FIXED** — `governance/tokenizer.py`: real tokenizer (tiktoken when present) + improved fallback; backend reported; settlement still uses upstream usage |
| 4 | Untrusted-node output unverifiable on the mesh | **MITIGATED** — `core/quorum_verify.py`: K-of-N redundant execution + agreement; single liar outvoted and unpaid; disputes surfaced. Defeats independent faults, **not** a colluding majority (needs the economic/reputation layer on top). Opt-in per job (cost = N× compute) |
| 5 | Savings numbers over-claimable | **HELD** — only measured counterfactuals in `net_saved`; compression savings labelled ESTIMATE in a separate bucket; the product promises "we show you what we saved," never a headline % |
| 6 | Semantic cache is lexical, not neural | **HONEST + PLUGGABLE** — labelled `lexical-3gram`; a real neural embedder plugs into `embed_fn`. Not claimed to be paraphrase-aware |
| 7 | Compression is dedup/whitespace, not LLMLingua | **HONEST + PLUGGABLE** — `compress_fn` slot for a local model; shipped transforms are deterministic and itemised |
| 8 | No real WAN proof; zero live nodes | **PARTLY DONE** — a cross-internet run has cleared: a buyer-only node on a GitHub runner (Microsoft's network) submitted a signed job that a home node behind NAT executed, via a Cloudflare tunnel, no shared network or seed (public `.github/workflows/wan-proof.yml` + its Actions logs; re-runnable). Still **OPEN**: two *independent home networks* introduced by a hosted seed VM |
| 9 | No HA / multi-tenant / TLS on the gateway | **OPEN** — single-process, file-backed; fine for a pilot, not for scale. TLS should terminate at a reverse proxy in front |
| 10 | Not run against a real LLM API | **OPEN** — needs a paid API key; subscription OAuth (e.g. Claude Code's) cannot be proxied |

## What is genuinely differentiated

- **Provable, billable savings.** Every saved dollar is a signed
  receipt with a measured counterfactual. This enables a savings-share
  business model competitors cannot easily match, because they lack the
  proof layer. This is the idea most worth pursuing.
- **One budget primitive spanning the gateway AND the mesh.** The
  gateway is the demand funnel; the mesh is the supply. No pure gateway
  has a mesh; no pure mesh marketplace has a demand product.

## What must not be claimed

- Not "blockchain." One writer, no consensus. It is a signed,
  hash-chained, externally-anchorable audit log.
- Not "3× cheaper" as a headline. Savings are traffic-dependent and
  only ever reported as measured.
- Not "replaces the datacenter." The realistic mesh market is
  latency-tolerant / cost-sensitive / privacy-sensitive batch.
- Not "trustless compute" until the economic layer above quorum exists.

## Next milestones (in order)

1. ~~One real cross-internet job clears~~ **DONE** (cloud runner → home
   node behind NAT; `wan-proof` workflow). Next: two *independent home
   networks* introduced by a hosted seed VM.
2. Run the gateway against one real API + one real workload; publish
   the measured savings.
3. The economic layer above quorum (stake + slashing + reputation)
   that turns "detect disagreement" into "make cheating unprofitable."
