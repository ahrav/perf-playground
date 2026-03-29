#!/usr/bin/env python3
import argparse
import csv
import subprocess
from pathlib import Path


def run_cmd(cmd, cwd=None):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, cwd=cwd).strip()
    except Exception:
        return ""


def read_rows(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "mode": row["mode"],
                    "chunk": int(row["chunk"]),
                    "qd": int(row["qd"]),
                    "iters": int(row["iters"]),
                    "completed": int(row["completed"]),
                    "p50_ns": int(row["p50_ns"]),
                    "p95_ns": int(row["p95_ns"]),
                    "p99_ns": int(row["p99_ns"]),
                    "mb_s": float(row["mb_s"]),
                    "ops_s": float(row["ops_s"]),
                    "errors": int(row["errors"]),
                    "latencies_csv": row["latencies_csv"],
                }
            )
    return rows


def plot_best_by_mode(rows, out_path):
    import matplotlib.pyplot as plt

    best = {}
    for row in rows:
        mode = row["mode"]
        if mode not in best or row["mb_s"] > best[mode]["mb_s"]:
            best[mode] = row

    modes = sorted(best.keys())
    values = [best[m]["mb_s"] for m in modes]

    plt.figure(figsize=(6.0, 4.0))
    plt.bar(modes, values)
    plt.ylabel("best MB/s")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_csv", required=True)
    parser.add_argument("--out", dest="out_dir", required=True)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_csv)
    if not rows:
        raise SystemExit("no rows in output.csv")

    modes = sorted({r["mode"] for r in rows})
    chunks = sorted({r["chunk"] for r in rows})
    qds = sorted({r["qd"] for r in rows})

    best_overall = max(rows, key=lambda r: r["mb_s"])
    best_by_mode = {}
    for row in rows:
        mode = row["mode"]
        if mode not in best_by_mode or row["mb_s"] > best_by_mode[mode]["mb_s"]:
            best_by_mode[mode] = row

    plot_best_by_mode(rows, plots_dir / "best_mbs_by_mode.png")

    repo_root = Path(__file__).resolve().parents[2]
    commit = run_cmd("git rev-parse HEAD", cwd=repo_root)

    summary = [
        "# Summary",
        "",
        f"rows: {len(rows)}",
        f"modes: {', '.join(modes)}",
        f"chunks: {', '.join(str(c) for c in chunks)}",
        f"qds: {', '.join(str(q) for q in qds)}",
        f"best_overall_mb_s: {best_overall['mb_s']:.1f} (mode={best_overall['mode']}, chunk={best_overall['chunk']}, qd={best_overall['qd']})",
    ]

    for mode in modes:
        row = best_by_mode[mode]
        summary.append(
            f"best_{mode}_mb_s: {row['mb_s']:.1f} (chunk={row['chunk']}, qd={row['qd']}, p99={row['p99_ns']} ns)"
        )

    if commit:
        summary.append(f"git_commit: {commit}")

    (out_dir / "summary.md").write_text("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
