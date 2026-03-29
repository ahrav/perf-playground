# uring_bench

## Goal

Compare three read paths on Linux:

- blocking pread ("readv")
- io_uring with registered buffers ("reg")
- io_uring with buf_ring buffer selection ("bufring")

## Hardware assumptions

- Linux kernel with io_uring support (5.10+). buf_ring requires 5.19+.
- Fast local storage (NVMe recommended).
- Stable CPU frequency (performance governor recommended).

## Workload definition

- Access pattern: sequential (wraps at EOF)
- Block size: `chunks` from config
- Queue depth: `qds` from config (applies to io_uring modes)
- Iterations: `iters` per run
- Warmup: `warmup_secs` per run

## Metrics

- p50/p95/p99 latency (ns)
- MB/s throughput
- ops/s
- error count

## Repro steps (3 commands max)

```
./experiments/uring_bench/run.sh --config ./experiments/uring_bench/configs/default.toml --out ./experiments/uring_bench/results/<date>_<machine>
python3 ./experiments/uring_bench/parse.py --in ./experiments/uring_bench/results/<date>_<machine>/output.csv --out ./experiments/uring_bench/results/<date>_<machine>
```

## Expected outputs

- `output.csv` (raw results)
- `latencies_*.csv` (per-run latencies)
- `env.json` (environment capture)
- `cmd.txt` (exact command line + git hash)
- `summary.md` (parsed summary)
- `plots/best_mbs_by_mode.png`

## Caveats

- buf_ring requires sufficient `memlock` limits and kernel >= 5.19.
- `readv` mode is blocking pread; queue depth does not apply.
- If `path` is not a multiple of `chunk`, final reads may be short.
