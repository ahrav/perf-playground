# filterbench

Compare construction time, query throughput, resident memory, and empirical false positive rate
across several filter implementations.

## Run

```
./experiments/filterbench/run.sh --config ./experiments/filterbench/configs/default.toml --out ./experiments/filterbench/results/$(date +%F)_$(hostname)
```

## Config

The config is a TOML file. Supported fields:

- n_keys
- seed
- bloom_fp_rate
- scalable_fp_rate
- rayon_threads (0 means use Rayon default)
- criterion (true to run Criterion benches)
- criterion_sample_size
- criterion_warmup_ms
- criterion_measurement_secs
- criterion_plots

Environment overrides:

- N_KEYS, SEED
- BLOOM_FP_RATE, SCALABLE_FP_RATE
- RAYON_NUM_THREADS (preferred) or RAYON_THREADS
- CRITERION

## Outputs

Each run writes:

- output.csv
- env.json
- cmd.txt
- summary.md (from parse.py)
- plots/ (from parse.py)

Notes:

- RSS deltas are only available on Linux. Other platforms report 0 KB.
- For perf counters, run the binary with `perf stat -e cycles,instructions,branches,branch-misses`.
