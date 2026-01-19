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
                    "name": row["name"],
                    "n_keys": int(row["n_keys"]),
                    "seed": int(row["seed"]),
                    "build_wall_ms": float(row["build_wall_ms"]),
                    "build_rss_kb": int(row["build_rss_kb"]),
                    "qps_single_pos": float(row["qps_single_pos"]),
                    "qps_single_neg": float(row["qps_single_neg"]),
                    "qps_multi_pos": float(row["qps_multi_pos"]),
                    "qps_multi_neg": float(row["qps_multi_neg"]),
                    "fp_rate": float(row["fp_rate"]),
                }
            )
    return rows


def plot_bar(labels, values, out_path, ylabel):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8.0, 4.0))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
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

    labels = [r["name"] for r in rows]
    plot_bar(
        labels,
        [r["qps_single_pos"] for r in rows],
        plots_dir / "qps_single_pos.png",
        "single-thread QPS (positives)",
    )
    plot_bar(
        labels,
        [r["qps_multi_pos"] for r in rows],
        plots_dir / "qps_multi_pos.png",
        "multi-thread QPS (positives)",
    )
    plot_bar(
        labels,
        [r["fp_rate"] for r in rows],
        plots_dir / "fp_rate.png",
        "false positive rate",
    )

    best_single = max(rows, key=lambda r: r["qps_single_pos"])
    best_multi = max(rows, key=lambda r: r["qps_multi_pos"])
    fastest_build = min(rows, key=lambda r: r["build_wall_ms"])
    lowest_fp = min(rows, key=lambda r: r["fp_rate"])

    repo_root = Path(__file__).resolve().parents[2]
    commit = run_cmd("git rev-parse HEAD", cwd=repo_root)

    summary = [
        "# Summary",
        "",
        f"rows: {len(rows)}",
        f"n_keys: {rows[0]['n_keys']}",
        f"seed: {rows[0]['seed']}",
        f"best_single_pos_qps: {best_single['qps_single_pos']:.0f} ({best_single['name']})",
        f"best_multi_pos_qps: {best_multi['qps_multi_pos']:.0f} ({best_multi['name']})",
        f"fastest_build_ms: {fastest_build['build_wall_ms']:.2f} ({fastest_build['name']})",
        f"lowest_fp_rate: {lowest_fp['fp_rate']:.6f} ({lowest_fp['name']})",
    ]
    if commit:
        summary.append(f"git_commit: {commit}")

    (out_dir / "summary.md").write_text("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
