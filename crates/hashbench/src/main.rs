use anyhow::{bail, Context, Result};
use clap::Parser;
use criterion::Criterion;
use hashbench::{all_hashfns, detect_cpu_features, generate_input, DataPattern, HashFn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(about = "Compare throughput and latency across hash algorithm implementations")]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    out: PathBuf,
}

fn default_min_time_ms() -> u64 {
    1000
}

fn default_warmup_iters() -> usize {
    100
}

fn default_criterion_sample_size() -> usize {
    100
}

fn default_criterion_warmup_ms() -> u64 {
    500
}

fn default_criterion_measurement_secs() -> u64 {
    3
}

#[derive(Debug, Deserialize)]
struct RunCfg {
    seed: u64,
    input_sizes: Vec<usize>,
    patterns: Vec<DataPattern>,
    #[serde(default = "default_min_time_ms")]
    min_time_ms: u64,
    #[serde(default = "default_warmup_iters")]
    warmup_iters: usize,
    #[serde(default)]
    algorithms: Vec<String>,
    #[serde(default)]
    criterion: bool,
    #[serde(default = "default_criterion_sample_size")]
    criterion_sample_size: usize,
    #[serde(default = "default_criterion_warmup_ms")]
    criterion_warmup_ms: u64,
    #[serde(default = "default_criterion_measurement_secs")]
    criterion_measurement_secs: u64,
    #[serde(default)]
    criterion_plots: bool,
}

struct CriterionCfg {
    enabled: bool,
    sample_size: usize,
    warmup: Duration,
    measurement: Duration,
    plots: bool,
}

struct MeasureResult {
    iterations: u64,
    total_ns: u64,
    ns_per_op: f64,
    throughput_mbs: f64,
}

#[derive(Debug, Serialize)]
struct OutputRow {
    algorithm: String,
    input_bytes: usize,
    pattern: String,
    iterations: u64,
    total_ns: u64,
    ns_per_op: f64,
    throughput_mbs: f64,
}

fn measure_hash(
    hash_fn: &dyn HashFn,
    input: &[u8],
    warmup_iters: usize,
    min_time: Duration,
) -> MeasureResult {
    // Warmup phase.
    for _ in 0..warmup_iters {
        std::hint::black_box(hash_fn.hash_bytes(std::hint::black_box(input)));
    }

    // Measurement phase: iterate until min_time exceeded, checking elapsed every 1024 iterations.
    let start = Instant::now();
    let mut iterations: u64 = 0;
    loop {
        for _ in 0..1024 {
            std::hint::black_box(hash_fn.hash_bytes(std::hint::black_box(input)));
        }
        iterations += 1024;
        if start.elapsed() >= min_time {
            break;
        }
    }
    let elapsed = start.elapsed();
    let total_ns = elapsed.as_nanos() as u64;
    let ns_per_op = total_ns as f64 / iterations as f64;
    let throughput_mbs =
        (input.len() as f64 * iterations as f64) / (elapsed.as_secs_f64() * 1_000_000.0);

    MeasureResult {
        iterations,
        total_ns,
        ns_per_op,
        throughput_mbs,
    }
}

fn write_csv(path: &std::path::Path, rows: &[OutputRow]) -> Result<()> {
    let mut writer =
        csv::Writer::from_path(path).with_context(|| format!("open csv {}", path.display()))?;
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

fn run_criterion(
    hash_fns: &[Box<dyn HashFn>],
    inputs: &HashMap<(usize, DataPattern), Vec<u8>>,
    input_sizes: &[usize],
    cfg: &CriterionCfg,
) {
    if !cfg.enabled {
        return;
    }

    let mut c = Criterion::default()
        .sample_size(cfg.sample_size)
        .warm_up_time(cfg.warmup)
        .measurement_time(cfg.measurement);
    if cfg.plots {
        c = c.with_plots();
    }

    let pattern = DataPattern::Random;
    for &size in input_sizes {
        let key = (size, pattern);
        if let Some(input) = inputs.get(&key) {
            let mut group = c.benchmark_group(format!("throughput/{}B", size));
            group.throughput(criterion::Throughput::Bytes(size as u64));
            for hash_fn in hash_fns {
                let name = hash_fn.name();
                group.bench_function(name, |b| {
                    b.iter(|| {
                        std::hint::black_box(hash_fn.hash_bytes(std::hint::black_box(input)))
                    });
                });
            }
            group.finish();
        }
    }
    c.final_summary();
}

fn env_u64(key: &str) -> Option<u64> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

fn env_bool(key: &str) -> Option<bool> {
    std::env::var(key).ok().and_then(|v| {
        let v = v.to_ascii_lowercase();
        match v.as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        }
    })
}

fn env_string(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg_text = std::fs::read_to_string(&args.config).context("read hashbench config")?;
    let cfg: RunCfg = toml::from_str(&cfg_text).context("parse hashbench config")?;

    // Apply environment variable overrides.
    let seed = env_u64("SEED").unwrap_or(cfg.seed);
    let min_time_ms = env_u64("MIN_TIME_MS").unwrap_or(cfg.min_time_ms);
    let warmup_iters = env_usize("WARMUP_ITERS").unwrap_or(cfg.warmup_iters);
    let criterion_enabled = env_bool("CRITERION").unwrap_or(cfg.criterion);
    let algorithms_override = env_string("ALGORITHMS");

    let algorithm_filter: Vec<String> = if let Some(ref alg_str) = algorithms_override {
        alg_str.split(',').map(|s| s.trim().to_string()).collect()
    } else {
        cfg.algorithms
    };

    // Validate.
    if cfg.input_sizes.is_empty() {
        bail!("input_sizes must not be empty");
    }
    if cfg.patterns.is_empty() {
        bail!("patterns must not be empty");
    }
    if min_time_ms == 0 {
        bail!("min_time_ms must be > 0");
    }

    let criterion_cfg = CriterionCfg {
        enabled: criterion_enabled,
        sample_size: cfg.criterion_sample_size,
        warmup: Duration::from_millis(cfg.criterion_warmup_ms),
        measurement: Duration::from_secs(cfg.criterion_measurement_secs),
        plots: cfg.criterion_plots,
    };

    // Create output directory.
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("create output dir {}", args.out.display()))?;
    let out_csv = args.out.join("output.csv");
    if out_csv.exists() {
        bail!("output.csv already exists at {}", out_csv.display());
    }

    // Detect CPU features.
    let features = detect_cpu_features();
    println!("CPU features: {}", features.join(", "));

    // Create hash functions.
    let all_fns = all_hashfns(seed);
    let hash_fns: Vec<Box<dyn HashFn>> = if algorithm_filter.is_empty() {
        all_fns
    } else {
        all_fns
            .into_iter()
            .filter(|f| algorithm_filter.iter().any(|a| a == f.name()))
            .collect()
    };

    if hash_fns.is_empty() {
        bail!("no hash functions selected (check algorithms filter)");
    }

    println!(
        "Algorithms ({}): {}",
        hash_fns.len(),
        hash_fns
            .iter()
            .map(|f| f.name())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "Input sizes ({}): {:?}",
        cfg.input_sizes.len(),
        cfg.input_sizes
    );
    println!("Patterns ({}): {:?}", cfg.patterns.len(), cfg.patterns);
    println!(
        "min_time={}ms warmup_iters={} seed=0x{:x}",
        min_time_ms, warmup_iters, seed
    );

    // Pre-generate all input buffers.
    let mut inputs: HashMap<(usize, DataPattern), Vec<u8>> = HashMap::new();
    for &size in &cfg.input_sizes {
        for &pattern in &cfg.patterns {
            inputs.insert((size, pattern), generate_input(size, pattern, seed));
        }
    }

    let min_time = Duration::from_millis(min_time_ms);
    let total_combos = hash_fns.len() * cfg.input_sizes.len() * cfg.patterns.len();
    let mut rows = Vec::with_capacity(total_combos);
    let mut done = 0usize;

    for hash_fn in &hash_fns {
        for &size in &cfg.input_sizes {
            for &pattern in &cfg.patterns {
                done += 1;
                let input = &inputs[&(size, pattern)];
                let result = measure_hash(&**hash_fn, input, warmup_iters, min_time);

                println!(
                    "[{}/{}] {:16} {:>7}B {:12} {:>10.2} ns/op {:>10.2} MB/s ({} iters)",
                    done,
                    total_combos,
                    hash_fn.name(),
                    size,
                    pattern,
                    result.ns_per_op,
                    result.throughput_mbs,
                    result.iterations,
                );

                rows.push(OutputRow {
                    algorithm: hash_fn.name().to_string(),
                    input_bytes: size,
                    pattern: pattern.to_string(),
                    iterations: result.iterations,
                    total_ns: result.total_ns,
                    ns_per_op: result.ns_per_op,
                    throughput_mbs: result.throughput_mbs,
                });
            }
        }
    }

    write_csv(&out_csv, &rows)?;
    println!("\nWrote {} rows to {}", rows.len(), out_csv.display());

    run_criterion(&hash_fns, &inputs, &cfg.input_sizes, &criterion_cfg);

    Ok(())
}
