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
                    "qd": int(row["qd"]),
                    "mb_s": float(row["mb_s"]),
                    "ops_s": float(row["ops_s"]),
                    "p50_ms": float(row["p50_ms"]),
                    "p95_ms": float(row["p95_ms"]),
                    "p99_ms": float(row["p99_ms"]),
                    "errors": int(row["errors"]),
                }
            )
    return rows


def plot_series(xs, ys, out_path, ylabel):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7.2, 4.0))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.xlabel("queue depth")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
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

    xs = [r["qd"] for r in rows]
    mb_s = [r["mb_s"] for r in rows]
    p99 = [r["p99_ms"] for r in rows]

    plot_series(xs, mb_s, plots_dir / "qd_vs_mbs.png", "MB/s")
    plot_series(xs, p99, plots_dir / "qd_vs_p99.png", "p99 latency (ms)")

    best_idx = max(range(len(rows)), key=lambda i: rows[i]["mb_s"])
    total_errors = sum(r["errors"] for r in rows)

    repo_root = Path(__file__).resolve().parents[2]
    commit = run_cmd("git rev-parse HEAD", cwd=repo_root)

    summary = [
        "# Summary",
        "",
        f"rows: {len(rows)}",
        f"best_mb_s: {rows[best_idx]['mb_s']:.1f} at qd={rows[best_idx]['qd']}",
        f"best_ops_s: {rows[best_idx]['ops_s']:.0f} at qd={rows[best_idx]['qd']}",
        f"p99_ms_at_best: {rows[best_idx]['p99_ms']:.3f}",
        f"total_errors: {total_errors}",
    ]
    if commit:
        summary.append(f"git_commit: {commit}")

    (out_dir / "summary.md").write_text("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
