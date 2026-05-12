#!/usr/bin/env bash
set -euo pipefail

# Recreate deps/Processes as its own nested git repo (so GUIs can push it separately).
# Default remote is HTTPS to avoid SSH-key setup on new machines.

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${ROOT}" ]]; then
  echo "error: must run inside the InteractiveIsing git repo" >&2
  exit 2
fi

PREFIX="deps/Processes"
REMOTE_URL_DEFAULT="https://github.com/f-ij/Processes.jl.git"
REMOTE_URL="${PROCESSES_REMOTE_URL:-$REMOTE_URL_DEFAULT}"
BRANCH="${PROCESSES_BRANCH:-main}"

YES=0
NO_BACKUP=0

usage() {
  cat <<EOF
Usage: scripts/reclone-processes.sh [--yes] [--no-backup] [--url <remote>] [--branch <branch>]

Environment:
  PROCESSES_REMOTE_URL   default: $REMOTE_URL_DEFAULT
  PROCESSES_BRANCH       default: main

Examples:
  bash scripts/reclone-processes.sh
  PROCESSES_REMOTE_URL=git@github.com:f-ij/Processes.jl.git bash scripts/reclone-processes.sh --yes
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y) YES=1; shift ;;
    --no-backup) NO_BACKUP=1; shift ;;
    --url) REMOTE_URL="${2:-}"; shift 2 ;;
    --branch) BRANCH="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

cd "$ROOT"

if [[ "$YES" -ne 1 ]]; then
  echo "This will DELETE '$PREFIX' and re-clone Processes.jl into it."
  read -r -p "Continue? [y/N] " ans
  case "${ans:-}" in
    y|Y|yes|YES) ;;
    *) echo "aborted"; exit 1 ;;
  esac
fi

if [[ -d "$PREFIX" ]]; then
  if [[ "$NO_BACKUP" -ne 1 ]]; then
    ts="$(date +%Y%m%d-%H%M%S)"
    backup="/tmp/Processes-backup-$ts"
    echo "Backing up existing '$PREFIX' to '$backup' ..."
    cp -a "$PREFIX" "$backup"
  fi
  rm -rf "$PREFIX"
fi

mkdir -p "$(dirname "$PREFIX")"
echo "Cloning $REMOTE_URL -> $PREFIX ..."
git clone "$REMOTE_URL" "$PREFIX"

if git -C "$PREFIX" show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
  git -C "$PREFIX" checkout -B "$BRANCH" "origin/$BRANCH"
else
  # If the remote uses a different default branch, leave whatever git cloned.
  echo "note: remote does not have origin/$BRANCH; leaving default branch as-cloned" >&2
fi

echo "ok: '$PREFIX' is now a git repo you can commit/push independently."

