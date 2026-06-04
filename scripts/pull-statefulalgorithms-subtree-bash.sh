#!/usr/bin/env bash
set -e

cd "$(git rev-parse --show-toplevel)"
branch="${1:-main}"
stashed=0

if [[ -n "$(git status --porcelain)" ]]; then
  git stash push --include-untracked -m "pre-subtree-pull-processes-$(date +%s)"
  stashed=1
fi

# Pull the requested StatefulAlgorithms.jl branch into the vendored subtree.
git subtree pull --prefix=deps/StatefulAlgorithms https://github.com/f-ij/StatefulAlgorithms.jl.git "$branch" --squash -m "Pull deps/StatefulAlgorithms subtree from $branch"

if [[ "$stashed" -eq 1 ]]; then
  git stash pop
fi
