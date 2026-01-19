#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./scripts/perf_stat.sh <out_file> -- <command> [args...]
Example: ./scripts/perf_stat.sh ./perf_stat.txt -- ./experiments/.../run.sh ...
USAGE
}

if [[ $# -lt 3 ]]; then
  usage
  exit 1
fi

OUT_FILE="$1"
shift

if [[ "$1" != "--" ]]; then
  usage
  exit 1
fi
shift

if ! command -v perf >/dev/null 2>&1; then
  echo "perf not found" >&2
  exit 1
fi

perf stat -d -d -o "$OUT_FILE" "$@"
