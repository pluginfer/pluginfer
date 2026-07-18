# Contributing to Pluginfer

Contributions are welcome — this project gets better with more eyes and
more hands. Apache-2.0 in, Apache-2.0 out: by submitting a PR you agree
your contribution is licensed under the repo's [LICENSE](LICENSE). No
CLA, no paperwork.

## Ground rules (the short version)

1. **No unmeasured claims.** This repo's brand is honesty. If your
   feature saves money, the saving must be *measured* (a counterfactual
   from real numbers), never projected. If something is estimated,
   label it estimated. README/AUDIT text must match what the code
   actually does.
2. **Tests are the price of admission.** Every behavior change needs a
   test that fails without it. Features listed in the README each have
   a passing suite in `v2/tests/` — keep that invariant.
3. **No placeholders, no mocks-as-product.** Default behavior must work
   with real data. Fakes belong in tests only, clearly labeled.
4. **Fix the class, not the instance.** If you're fixing a bug, sweep
   the codebase for the same pattern and add the regression test.

## Dev setup

```sh
git clone https://github.com/pluginfer/pluginfer
cd pluginfer/v2
pip install -r requirements.txt
python -m pytest tests/ -q          # full suite (~2 min)
python pluginfer.py up              # run a node locally
```

CI runs the suite on Ubuntu, Windows, and macOS × Python 3.11/3.12,
plus `bandit` (security) and `pip-audit`. A green local run on any one
OS is enough to open the PR — the matrix catches the rest.

## Good places to start

- Issues labeled `good first issue`.
- Anything in [AUDIT.md](AUDIT.md) marked **OPEN** — these are the
  honestly-tracked gaps, from "TLS/HA on the gateway" to "hosted seed
  for two-home-networks discovery". Closing one is a headline
  contribution.
- Provider adapters: wrappers that make more runtimes joinable as
  auction supply (see `core/meshllm_provider.py` for the pattern).
- The router's task classifier (`governance/router.py`) is transparent
  keyword heuristics by design — better *explainable* classification is
  welcome; a black-box model is not.

## PR checklist

- [ ] Tests added/updated and passing locally
- [ ] No new claims in docs that the code doesn't measurably back
- [ ] `python -m bandit -r v2/core v2/api -ll` clean (or `# nosec` with
      a written justification)
- [ ] One logical change per PR — small PRs merge fast

## Security issues

Please do **not** open a public issue for vulnerabilities — see
[SECURITY.md](SECURITY.md).
