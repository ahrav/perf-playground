#!/usr/bin/env python3
import argparse
import csv
import math
import subprocess
from collections import defaultdict
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
                    "algorithm": row["algorithm"],
                    "input_bytes": int(row["input_bytes"]),
                    "pattern": row["pattern"],
                    "iterations": int(row["iterations"]),
                    "total_ns": int(row["total_ns"]),
                    "ns_per_op": float(row["ns_per_op"]),
                    "throughput_mbs": float(row["throughput_mbs"]),
                }
            )
    return rows


def save_fig(fig, out_path, formats):
    """Save a figure in all requested formats."""
    for fmt in formats:
        p = out_path.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=160, format=fmt)


def plot_throughput(rows, pattern, out_path, formats):
    """Throughput vs input size (log2 x-axis), one line per algorithm."""
    import matplotlib.pyplot as plt

    # Group by algorithm.
    by_algo = defaultdict(list)
    for r in rows:
        if r["pattern"] == pattern:
            by_algo[r["algorithm"]].append(r)

    if not by_algo:
        return

    fig = plt.figure(figsize=(10.0, 6.0))
    for algo, data in sorted(by_algo.items()):
        data.sort(key=lambda r: r["input_bytes"])
        sizes = [r["input_bytes"] for r in data]
        throughputs = [r["throughput_mbs"] for r in data]
        x = [math.log2(s) for s in sizes]
        plt.plot(x, throughputs, marker="o", markersize=4, label=algo)

    plt.xlabel("Input size (log2 bytes)")
    plt.ylabel("Throughput (MB/s)")
    plt.title(f"Hash throughput — {pattern}")
    plt.legend(fontsize=7, loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    save_fig(fig, out_path, formats)
    plt.close()


def plot_latency_bar(rows, pattern, size, out_path, formats):
    """Horizontal bar chart of ns/op at a given input size, sorted fastest-first."""
    import matplotlib.pyplot as plt

    data = [
        r for r in rows if r["pattern"] == pattern and r["input_bytes"] == size
    ]
    if not data:
        return

    data.sort(key=lambda r: r["ns_per_op"])
    labels = [r["algorithm"] for r in data]
    values = [r["ns_per_op"] for r in data]

    fig = plt.figure(figsize=(8.0, max(3.0, 0.35 * len(labels))))
    plt.barh(labels, values)
    plt.xlabel("Latency (ns/op)")
    plt.title(f"Hash latency at {size}B — {pattern}")
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    save_fig(fig, out_path, formats)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_csv", required=True)
    parser.add_argument("--out", dest="out_dir", required=True)
    parser.add_argument(
        "--format",
        dest="formats",
        default="png,svg",
        help="Comma-separated output formats (default: png,svg)",
    )
    args = parser.parse_args()
    formats = [f.strip() for f in args.formats.split(",")]

    input_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_csv)
    if not rows:
        raise SystemExit("no rows in output.csv")

    patterns = sorted(set(r["pattern"] for r in rows))
    sizes = sorted(set(r["input_bytes"] for r in rows))

    # Plot 1: Throughput vs input size — one plot per pattern.
    for pattern in patterns:
        plot_throughput(rows, pattern, plots_dir / f"throughput_{pattern}", formats)

    # Plot 2: Latency bar charts at representative small and large sizes for each pattern.
    # Pick the second-smallest and second-largest sizes when available, falling back to
    # the smallest and largest so charts remain useful with fewer than 3 sizes.
    latency_sizes = set()
    if len(sizes) >= 3:
        latency_sizes.add(sizes[1])   # second-smallest
        latency_sizes.add(sizes[-2])  # second-largest
    elif len(sizes) == 2:
        latency_sizes.update(sizes)
    elif sizes:
        latency_sizes.add(sizes[0])

    for pattern in patterns:
        for size in sorted(latency_sizes):
            plot_latency_bar(
                rows,
                pattern,
                size,
                plots_dir / f"latency_{size}B_{pattern}",
                formats,
            )

    # Summary.
    repo_root = Path(__file__).resolve().parents[2]
    commit = run_cmd("git rev-parse HEAD", cwd=repo_root)

    # Find fastest at small (64B) and large (65536B) for random pattern.
    random_rows = [r for r in rows if r["pattern"] == "random"]
    small_rows = [r for r in random_rows if r["input_bytes"] == 64]
    large_rows = [r for r in random_rows if r["input_bytes"] == 65536]

    summary = ["# Summary", ""]
    summary.append(f"rows: {len(rows)}")

    seed_values = set(run_cmd(f"grep '^seed' {input_csv.parent / 'config.toml'}", cwd=repo_root).split())
    summary.append(f"seed: (see config.toml)")

    if small_rows:
        best_small = max(small_rows, key=lambda r: r["throughput_mbs"])
        summary.append(
            f"fastest_at_64B: {best_small['throughput_mbs']:.2f} MB/s ({best_small['algorithm']})"
        )
    if large_rows:
        best_large = max(large_rows, key=lambda r: r["throughput_mbs"])
        summary.append(
            f"fastest_at_65536B: {best_large['throughput_mbs']:.2f} MB/s ({best_large['algorithm']})"
        )

    if commit:
        summary.append(f"git_commit: {commit}")

    (out_dir / "summary.md").write_text("\n".join(summary) + "\n")
    print(f"Wrote summary to {out_dir / 'summary.md'}")
    print(f"Wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
