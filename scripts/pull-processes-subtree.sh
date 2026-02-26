#!/usr/bin/env zsh
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
stash_ref=""

restore_stash() {
  if [[ -n "$stash_ref" ]]; then
    git stash apply "$stash_ref"
  fi
}

if [[ -n "$(git status --porcelain)" ]]; then
  stash_msg="pre-subtree-pull-processes-$(date +%s)"
  git stash push --include-untracked -m "$stash_msg"
  stash_ref="$(git stash list --format='%gd %s' | awk -v msg="$stash_msg" '$0 ~ msg { print $1; exit }')"
fi

trap restore_stash EXIT

git subtree pull --prefix=deps/Processes https://github.com/f-ij/Processes.jl.git main --squash -m "Pull deps/Processes subtree"

if [[ -n "$stash_ref" ]]; then
  git stash drop "$stash_ref" >/dev/null
  stash_ref=""
fi

trap - EXIT
