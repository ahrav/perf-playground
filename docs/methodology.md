# Methodology

This repo favors experiments that are reproducible, parameterized, and minimally confounded by environment drift.

## Core principles

- Capture environment details for every run.
- Treat experiments as code: versioned, repeatable, and peer-reviewable.
- Keep command lines short and explicit.
- Prefer controlled, synthetic workloads when isolating a variable.

## Expected artifacts

For each run, produce:

- `output.csv` (raw numeric data)
- `env.json` (host and kernel details)
- `summary.md` (interpretation and links to large logs)
- plots under `plots/`
