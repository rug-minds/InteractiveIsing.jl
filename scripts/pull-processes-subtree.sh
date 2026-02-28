#!/usr/bin/env zsh
set -e

cd "$(git rev-parse --show-toplevel)"
stashed=0

if [[ -n "$(git status --porcelain)" ]]; then
  git stash push --include-untracked -m "pre-subtree-pull-processes-$(date +%s)"
  stashed=1
fi

git subtree pull --prefix=deps/Processes https://github.com/f-ij/Processes.jl.git main --squash -m "Pull deps/Processes subtree"

if [[ "$stashed" -eq 1 ]]; then
  git stash pop
fi
