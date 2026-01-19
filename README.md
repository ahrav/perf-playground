# perf-playground

Reproducible performance experiments with strict separation between runnable code, experiment definitions, and results.

## What this repo is for

- Reproducible performance experiments
  - Code, configs, dataset generators, and exact run commands
- Results that survive time
  - Raw outputs, parsed summaries, plots, notes, and what changed
- A place to park microbenchmarks without polluting product repos

## What not to do

- Do not commit huge raw datasets or multi-GB artifacts
- Do not hand-wave environment details
- Do not store results without the command line and commit hash that produced them

## Layout

```
perf-playground/
  README.md
  LICENSE
  .gitignore
  CONTRIBUTING.md
  Cargo.toml

  experiments/
    io_uring_queue_depth_sweep/
      README.md
      run.sh
      parse.py
      configs/
        default.toml
      notes.md
      results/
        README.md
    filterbench/
      README.md
      run.sh
      parse.py
      configs/
        default.toml
      notes.md
      results/
        README.md

  crates/
    io_uring_queue_depth_sweep/
      Cargo.toml
      src/
        main.rs
    filterbench/
      Cargo.toml
      src/
        main.rs

  scripts/
    collect_env.sh
    pin_cpu.sh
    perf_stat.sh

  docs/
    methodology.md
    glossary.md
```

## Standard command contract

Every experiment must support:

```
./experiments/<name>/run.sh --config <cfg> --out <dir>
```

It must always produce `output.csv` and `env.json` in the output directory.

## Quickstart

1. Build the io_uring sweep harness:

```
cargo build --release --manifest-path crates/io_uring_queue_depth_sweep/Cargo.toml
```

Note: the io_uring harness builds and runs only on Linux.

2. Run an experiment (example):

```
./experiments/io_uring_queue_depth_sweep/run.sh \
  --config ./experiments/io_uring_queue_depth_sweep/configs/default.toml \
  --out ./experiments/io_uring_queue_depth_sweep/results/$(date +%F)_$(hostname)
```

3. Parse and plot results:

```
python3 ./experiments/io_uring_queue_depth_sweep/parse.py \
  --in ./experiments/io_uring_queue_depth_sweep/results/<date>_<machine>/output.csv \
  --out ./experiments/io_uring_queue_depth_sweep/results/<date>_<machine>
```

## Versioning results

- Store raw CSV and summary text in-repo
- Large logs go in a gist/S3 and linked from `summary.md`
- Include:
  - git commit hash
  - compiler version
  - kernel version
  - storage device details
  - CPU frequency/governor

## Tooling choices

- Rust for harnesses
- Python for parsing + plotting
- Optional: `justfile` to standardize commands
