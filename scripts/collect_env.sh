#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./scripts/collect_env.sh --out <file>
USAGE
}

OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT="$2"
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

if [[ -z "$OUT" ]]; then
  usage
  exit 1
fi

python3 - <<'PY' "$OUT"
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

out_path = sys.argv[1]
repo_root = Path(__file__).resolve().parents[1]


def cmd(cmd, cwd=None):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, cwd=cwd).strip()
    except Exception:
        return ""


def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


data = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "hostname": platform.node(),
    "kernel": cmd("uname -a"),
    "kernel_release": platform.release(),
    "os_release": cmd("cat /etc/os-release"),
    "git_commit": cmd("git rev-parse HEAD", cwd=repo_root),
    "git_dirty": bool(cmd("git status --porcelain", cwd=repo_root)),
    "rustc": cmd("rustc --version"),
    "cargo": cmd("cargo --version"),
    "lscpu": cmd("lscpu"),
}

cpu_governor = read_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
if cpu_governor:
    data["cpu_governor"] = cpu_governor

cpu_freq = read_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
if cpu_freq:
    data["cpu_freq_khz"] = cpu_freq

lsblk_json = cmd("lsblk -J -d -o NAME,MODEL,SERIAL,SIZE,ROTA,TRAN")
if lsblk_json.startswith("{"):
    try:
        data["lsblk"] = json.loads(lsblk_json)
    except Exception:
        data["lsblk"] = lsblk_json
else:
    data["lsblk"] = cmd("lsblk -d -o NAME,MODEL,SERIAL,SIZE,ROTA,TRAN")

findmnt = cmd("findmnt -J")
if findmnt.startswith("{"):
    try:
        data["mounts"] = json.loads(findmnt)
    except Exception:
        data["mounts"] = findmnt
else:
    data["mounts"] = cmd("mount")

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True)
PY
