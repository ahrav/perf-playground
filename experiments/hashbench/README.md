# hashbench

Compare throughput and latency across high-performance hash algorithm implementations
on x86_64 and aarch64.

## Run

```
./experiments/hashbench/run.sh --config ./experiments/hashbench/configs/default.toml --out ./experiments/hashbench/results/$(date +%F)_$(hostname)
```

## Config

The config is a TOML file. Supported fields:

- seed
- input_sizes
- patterns (random, low_entropy, repeating)
- min_time_ms (minimum measurement duration per combination)
- warmup_iters
- algorithms (empty = all)
- criterion (true to run inline Criterion benches)
- criterion_sample_size
- criterion_warmup_ms
- criterion_measurement_secs
- criterion_plots

Environment overrides:

- SEED, MIN_TIME_MS, WARMUP_ITERS
- ALGORITHMS (comma-separated filter list)
- CRITERION

## Outputs

Each run writes:

- output.csv (columns: algorithm, input_bytes, pattern, iterations, total_ns, ns_per_op, throughput_mbs)
- env.json
- cmd.txt
- summary.md (from parse.py)
- plots/ (from parse.py)

## Algorithms

blake3, blake3_keyed, highway, siphash13, siphash24, xxh3, xxh3_seeded, wyhash, crc32, ahash, gxhash (x86_64/aarch64 only), foldhash
