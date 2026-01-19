# io_uring_queue_depth_sweep

## Goal

Measure how queue depth impacts throughput and tail latency for sequential I/O using io_uring.

## Hardware assumptions

- Linux kernel with io_uring support (5.10+ recommended)
- Local NVMe or fast block device for consistent measurements
- CPU frequency governor set to performance for stable results

## Workload definition

- Access pattern: sequential
- Block size: `buf_size` (bytes) from config
- File size: `file_len` (bytes) from config
- Direct I/O: `direct = true` enables O_DIRECT
- Modes compared: buffered vs O_DIRECT, fixed buffers on/off, SQPOLL on/off

## Metrics

- MB/s
- ops/s
- p50/p95/p99 latency (ms)
- error count (failed or short I/O)

## Repro steps (3 commands max)

```
./experiments/io_uring_queue_depth_sweep/run.sh --config ./experiments/io_uring_queue_depth_sweep/configs/default.toml --out ./experiments/io_uring_queue_depth_sweep/results/<date>_<machine>
python3 ./experiments/io_uring_queue_depth_sweep/parse.py --in ./experiments/io_uring_queue_depth_sweep/results/<date>_<machine>/output.csv --out ./experiments/io_uring_queue_depth_sweep/results/<date>_<machine>
```

## Expected outputs

- `output.csv` (raw results)
- `env.json` (environment capture)
- `cmd.txt` (exact command line + git hash)
- `summary.md` (parsed summary)
- `plots/qd_vs_mbs.png`
- `plots/qd_vs_p99.png`

## Caveats

- O_DIRECT requires aligned buffers and offsets; ensure `buf_size` is a multiple of the device block size.
- SQPOLL requires elevated privileges or configured `memlock` limits.
- Results are sensitive to CPU governor, thermal throttling, and filesystem mount options.
