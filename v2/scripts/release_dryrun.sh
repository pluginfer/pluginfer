#!/usr/bin/env bash
# Pluginfer release dry-run.
#
# Verifies the three distribution artefacts (Python SDK wheel, JS SDK
# tarball, devserver Docker image) all build cleanly on the local
# machine BEFORE pushing a tag. Catches packaging regressions the
# moment they land instead of at release time.
#
# Usage:
#   bash v2/scripts/release_dryrun.sh        # build everything
#   bash v2/scripts/release_dryrun.sh python # only the Python sdk
#   bash v2/scripts/release_dryrun.sh js
#   bash v2/scripts/release_dryrun.sh docker
#
# Exits non-zero on the first failure so this is CI-safe.

set -euo pipefail

TARGET=${1:-all}
HERE=$(cd "$(dirname "$0")" && pwd)
V2=$(cd "$HERE/.." && pwd)

cd "$V2"

step() { printf "\n\033[1;36m▎ %s\033[0m\n" "$*"; }
fail() { printf "\n\033[1;31m✗ %s\033[0m\n" "$*" >&2; exit 1; }
ok()   { printf "\033[1;32m✓ %s\033[0m\n" "$*"; }

if [[ "$TARGET" == "python" || "$TARGET" == "all" ]]; then
  step "Build Python SDK"
  pushd sdk/python >/dev/null
  rm -rf dist build *.egg-info
  python -m pip install --quiet --upgrade pip build twine
  python -m build
  python -m twine check dist/*
  ls -l dist/
  popd >/dev/null
  ok  "Python SDK builds clean (dist/ ready for twine upload)"
fi

if [[ "$TARGET" == "js" || "$TARGET" == "all" ]]; then
  step "Build JS SDK"
  pushd sdk/javascript >/dev/null
  # npm install is idempotent and lockfile-respecting; npm ci would
  # require package-lock.json which we don't ship as part of the repo
  # tree intentionally (consumers shouldn't be forced to a specific
  # node_modules layout).
  if ! command -v npm >/dev/null; then
    fail "npm not on PATH — install node 18+ and retry"
  fi
  npm install --silent --no-audit --no-fund
  npm run build
  npm pack
  ls -l ./*.tgz
  popd >/dev/null
  ok  "JS SDK builds clean (tarball ready for npm publish --provenance)"
fi

if [[ "$TARGET" == "docker" || "$TARGET" == "all" ]]; then
  step "Build devserver Docker image"
  if ! command -v docker >/dev/null; then
    fail "docker not on PATH — install Docker Desktop / engine and retry"
  fi
  docker build -t pluginfer/devserver:dryrun -f api/Dockerfile .
  # Boot a throwaway container and curl /healthz to prove the image
  # actually starts. 5s is generous — uvicorn is up in ~300ms.
  ID=$(docker run -d --rm -p 11434:11434 pluginfer/devserver:dryrun)
  trap 'docker rm -f "$ID" >/dev/null 2>&1 || true' EXIT
  for _ in $(seq 1 25); do
    if curl --fail --silent http://127.0.0.1:11434/healthz >/dev/null; then
      break
    fi
    sleep 0.2
  done
  curl --fail --silent http://127.0.0.1:11434/healthz | head -c 200
  echo
  ok  "Devserver image starts + answers /healthz"
fi

echo
ok  "All requested artefacts built. Ready to tag + push."
