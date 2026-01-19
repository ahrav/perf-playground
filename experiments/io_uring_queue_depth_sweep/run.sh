#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./experiments/io_uring_queue_depth_sweep/run.sh --config <cfg> --out <dir>
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

ORIG_ARGS=("$@")
CONFIG=""
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --out)
      OUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$CONFIG" || -z "$OUT_DIR" ]]; then
  usage
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUT_DIR"

if [[ -e "$OUT_DIR/output.csv" ]]; then
  echo "output.csv already exists in $OUT_DIR" >&2
  exit 1
fi

cp "$CONFIG" "$OUT_DIR/config.toml"

{
  echo "cmd: ${BASH_SOURCE[0]} ${ORIG_ARGS[*]}"
  echo "git_commit: $(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "rustc: $(rustc --version 2>/dev/null || echo unknown)"
} > "$OUT_DIR/cmd.txt"

"$ROOT_DIR/scripts/collect_env.sh" --out "$OUT_DIR/env.json"

cargo run --release --manifest-path "$ROOT_DIR/crates/io_uring_queue_depth_sweep/Cargo.toml" \
  -- --config "$CONFIG" --out "$OUT_DIR"

printf "\nRun complete. Raw results at %s\n" "$OUT_DIR/output.csv"
printf "Parse with: python3 %s/parse.py --in %s/output.csv --out %s\n" "$EXP_DIR" "$OUT_DIR" "$OUT_DIR"
