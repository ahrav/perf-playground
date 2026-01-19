#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./scripts/pin_cpu.sh <cpu_list> -- <command> [args...]
Example: ./scripts/pin_cpu.sh 0-3 -- ./experiments/.../run.sh ...
USAGE
}

if [[ $# -lt 3 ]]; then
  usage
  exit 1
fi

CPU_LIST="$1"
shift

if [[ "$1" != "--" ]]; then
  usage
  exit 1
fi
shift

if ! command -v taskset >/dev/null 2>&1; then
  echo "taskset not found" >&2
  exit 1
fi

exec taskset -c "$CPU_LIST" "$@"
