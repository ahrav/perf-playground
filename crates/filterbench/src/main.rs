//! Membership filter benchmarking harness.
//!
//! Compares build time, memory usage, query throughput, and false positive rates across
//! multiple filter implementations:
//! - [`xorf::Xor8`] and [`xorf::BinaryFuse8`]
//! - `xorfilter_rs::Xor8`
//! - [`bloomfilter::Bloom`]
//! - [`scalable_bloom_filter::ScalableBloomFilter`]
//!
//! # Architecture
//!
//! Filters implement the [`StaticFilter`] trait (query interface) and [`StaticFilterBuild`]
//! trait (construction). This separation allows benchmarking query performance independently
//! from construction.
//!
//! # Output
//!
//! Produces `output.csv` with columns: name, n_keys, seed, build_wall_ms, build_rss_kb,
//! qps_single_pos, qps_single_neg, qps_multi_pos, qps_multi_neg, fp_rate.
//!
//! # Environment Overrides
//!
//! Configuration values can be overridden via environment variables:
//! - `N_KEYS`: Number of keys to insert
//! - `SEED`: RNG seed for reproducibility
//! - `BLOOM_FP_RATE`: Target false positive rate for Bloom filter
//! - `SCALABLE_FP_RATE`: Target false positive rate for scalable Bloom filter
//! - `RAYON_THREADS`: Thread count for parallel queries (config file)
//! - `RAYON_NUM_THREADS`: Thread count (standard Rayon env var, takes precedence)
//! - `CRITERION`: Enable Criterion.rs benchmarks (true/false)

use anyhow::{bail, Context, Result};
use clap::Parser;
use criterion::Criterion;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(target_os = "linux")]
use procfs::process::Process;

#[derive(Parser, Debug)]
#[command(about = "Compare build/query characteristics for membership filters")]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    out: PathBuf,
}

fn default_criterion_sample_size() -> usize {
    20
}

fn default_criterion_warmup_ms() -> u64 {
    200
}

fn default_criterion_measurement_secs() -> u64 {
    2
}

/// Benchmark configuration loaded from TOML.
#[derive(Debug, Deserialize)]
struct RunCfg {
    /// Number of keys to insert into filters. Must be > 0.
    n_keys: usize,
    /// RNG seed for reproducible key generation.
    seed: u64,
    /// Target false positive rate for `bloomfilter::Bloom`. Must be in (0, 1).
    bloom_fp_rate: f64,
    /// Target false positive rate for `ScalableBloomFilter`. Must be in (0, 1).
    scalable_fp_rate: f64,
    /// Thread count for Rayon parallel queries. `None` uses Rayon's default.
    #[serde(default)]
    rayon_threads: Option<usize>,
    /// Enable Criterion.rs statistical benchmarks.
    #[serde(default)]
    criterion: bool,
    /// Criterion sample size (iterations per benchmark). Default: 20.
    #[serde(default = "default_criterion_sample_size")]
    criterion_sample_size: usize,
    /// Criterion warmup duration in milliseconds. Default: 200.
    #[serde(default = "default_criterion_warmup_ms")]
    criterion_warmup_ms: u64,
    /// Criterion measurement duration in seconds. Default: 2.
    #[serde(default = "default_criterion_measurement_secs")]
    criterion_measurement_secs: u64,
    /// Generate Criterion HTML plots.
    #[serde(default)]
    criterion_plots: bool,
}

/// Parameters passed to filter constructors.
///
/// Only certain parameters apply to each filter type:
/// - `seed`: Used by `bloomfilter::Bloom` for hash function seeding
/// - `bloom_fp_rate`: Used only by `bloomfilter::Bloom`
/// - `scalable_fp_rate`: Used only by `ScalableBloomFilter`
///
/// Xor filters ignore all parameters as they have fixed construction behavior.
#[derive(Clone, Copy, Debug)]
struct FilterParams {
    seed: u64,
    bloom_fp_rate: f64,
    scalable_fp_rate: f64,
}

#[derive(Debug)]
struct CriterionCfg {
    enabled: bool,
    sample_size: usize,
    warmup: Duration,
    measurement: Duration,
    plots: bool,
}

/// Query interface for membership filters.
///
/// The `Sync` bound enables parallel query benchmarks via Rayon.
trait StaticFilter: Sync {
    /// Returns the filter implementation name for reporting.
    fn name(&self) -> &'static str;

    /// Returns `true` if `key` is probably in the set, `false` if definitely not.
    fn contains(&self, key: u64) -> bool;
}

/// Construction interface for membership filters.
///
/// Separated from [`StaticFilter`] to allow measuring build time independently
/// from query performance.
trait StaticFilterBuild: Sized {
    /// Builds the filter from the given keys using the provided parameters.
    fn build(keys: &[u64], params: FilterParams) -> Result<Self>;
}

// ---- Candidates -----------------------------------------------------------------------
mod xorf_impl {
    use super::*;
    use xorf::{Filter, Xor8};

    pub struct Xorf8 {
        f: Xor8,
    }

    impl StaticFilter for Xorf8 {
        fn name(&self) -> &'static str {
            "xorf::Xor8"
        }
        fn contains(&self, k: u64) -> bool {
            self.f.contains(&k)
        }
    }

    impl StaticFilterBuild for Xorf8 {
        fn build(keys: &[u64], _params: FilterParams) -> Result<Self> {
            let f = Xor8::from(keys);
            Ok(Self { f })
        }
    }
}

mod xorf_binary_fuse_impl {
    use super::*;
    use xorf::{BinaryFuse8, Filter};

    pub struct XorfBinaryFuse8 {
        f: BinaryFuse8,
    }

    impl StaticFilter for XorfBinaryFuse8 {
        fn name(&self) -> &'static str {
            "xorf::BinaryFuse8"
        }
        fn contains(&self, k: u64) -> bool {
            self.f.contains(&k)
        }
    }

    impl StaticFilterBuild for XorfBinaryFuse8 {
        fn build(keys: &[u64], _params: FilterParams) -> Result<Self> {
            let f = BinaryFuse8::try_from(keys).map_err(anyhow::Error::msg)?;
            Ok(Self { f })
        }
    }
}

mod xorfilter_rs_impl {
    use super::*;
    use xorfilter::Xor8;

    pub struct XorfilterRs {
        f: Xor8,
    }

    impl StaticFilter for XorfilterRs {
        fn name(&self) -> &'static str {
            "xorfilter_rs::Xor8"
        }
        fn contains(&self, k: u64) -> bool {
            self.f.contains_key(k)
        }
    }

    impl StaticFilterBuild for XorfilterRs {
        fn build(keys: &[u64], _params: FilterParams) -> Result<Self> {
            let mut f = Xor8::new();
            f.build_keys(keys).context("build xorfilter_rs::Xor8")?;
            Ok(Self { f })
        }
    }
}

mod bloom_impl {
    use super::*;
    use bloomfilter::Bloom;

    pub struct BloomFilter {
        f: Bloom<u64>,
    }

    impl StaticFilter for BloomFilter {
        fn name(&self) -> &'static str {
            "bloomfilter::Bloom"
        }
        fn contains(&self, k: u64) -> bool {
            self.f.check(&k)
        }
    }

    impl StaticFilterBuild for BloomFilter {
        fn build(keys: &[u64], params: FilterParams) -> Result<Self> {
            let seed = seed_from_u64(params.seed);
            let mut f = Bloom::new_for_fp_rate_with_seed(keys.len(), params.bloom_fp_rate, &seed)
                .map_err(anyhow::Error::msg)?;
            for &k in keys {
                f.set(&k);
            }
            Ok(Self { f })
        }
    }
}

mod scalable_bloom_impl {
    use super::*;
    use scalable_bloom_filter::ScalableBloomFilter;

    pub struct ScalableBloom {
        f: ScalableBloomFilter<u64>,
    }

    impl StaticFilter for ScalableBloom {
        fn name(&self) -> &'static str {
            "scalable_bloom_filter"
        }
        fn contains(&self, k: u64) -> bool {
            self.f.contains(&k)
        }
    }

    impl StaticFilterBuild for ScalableBloom {
        fn build(keys: &[u64], params: FilterParams) -> Result<Self> {
            let mut f = ScalableBloomFilter::new(keys.len(), params.scalable_fp_rate);
            for &k in keys {
                f.insert(&k);
            }
            Ok(Self { f })
        }
    }
}

/// Generates deterministic test datasets.
///
/// Returns `(positives, negatives)` where:
/// - `positives`: `n` unique random keys to insert into filters
/// - `negatives`: `n` unique random keys guaranteed NOT in the positive set
///
/// The negative set is used to measure false positive rates.
fn generate_keys(n: usize, seed: u64) -> (Vec<u64>, Vec<u64>) {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut pos = Vec::with_capacity(n);
    let mut neg = Vec::with_capacity(n);
    let mut seen: HashSet<u64> = HashSet::with_capacity(n * 2);

    while pos.len() < n {
        let x = rng.gen::<u64>();
        if seen.insert(x) {
            pos.push(x);
        }
    }
    while neg.len() < n {
        let x = rng.gen::<u64>();
        if !seen.contains(&x) {
            neg.push(x);
        }
    }
    (pos, neg)
}

// ---- Peak RSS snapshot (approx) -------------------------------------------------------
#[cfg(target_os = "linux")]
fn current_rss_kb() -> u64 {
    Process::myself()
        .ok()
        .and_then(|p| p.statm().ok())
        .map(|m| {
            let page_size = procfs::page_size().unwrap_or(4096) as u64;
            m.resident * page_size / 1024
        })
        .unwrap_or(0)
}

#[cfg(not(target_os = "linux"))]
fn current_rss_kb() -> u64 {
    0
}

// ---- Micro-bench utilities ------------------------------------------------------------
#[derive(Debug)]
struct BuildMetrics {
    wall: Duration,
    peak_rss_kb: u64,
}

#[derive(Debug)]
struct QueryMetrics {
    qps_single_pos: f64,
    qps_single_neg: f64,
    qps_multi_pos: f64,
    qps_multi_neg: f64,
    fp_rate: f64,
}

/// Builds a filter and measures construction metrics.
///
/// Returns the constructed filter and metrics (wall time, approximate RSS delta).
///
/// RSS measurement is approximate: it captures process-wide RSS before/after construction,
/// which may include unrelated allocations. On non-Linux platforms, RSS is always 0.
fn measure_build<B>(
    keys: &[u64],
    params: FilterParams,
) -> Result<(Arc<dyn StaticFilter + Send + Sync>, BuildMetrics)>
where
    B: StaticFilter + StaticFilterBuild + Send + Sync + 'static,
{
    let rss0 = current_rss_kb();
    let start = Instant::now();
    let filter = B::build(keys, params)?;
    let wall = start.elapsed();
    let rss1 = current_rss_kb();
    let peak = rss1.saturating_sub(rss0);
    let name = filter.name();
    let filter = Arc::new(filter) as Arc<dyn StaticFilter + Send + Sync>;

    println!("[BUILD] {:20} wall={:?} peak_rss~={}KB", name, wall, peak);
    Ok((
        filter,
        BuildMetrics {
            wall,
            peak_rss_kb: peak,
        },
    ))
}

/// Measures single-threaded query throughput (queries per second).
///
/// Uses `black_box` to prevent the compiler from optimizing away the query results.
fn qps_single<F: StaticFilter + ?Sized>(f: &F, keys: &[u64]) -> f64 {
    let start = Instant::now();
    let mut hits = 0u64;
    for &k in keys {
        if f.contains(k) {
            hits += 1;
        }
    }
    std::hint::black_box(hits);
    let dt = start.elapsed().as_secs_f64();
    (keys.len() as f64) / dt
}

/// Measures multi-threaded query throughput using Rayon.
///
/// Uses `black_box` to prevent the compiler from optimizing away the query results.
fn qps_multi<F: StaticFilter + Sync + ?Sized>(f: &F, keys: &[u64]) -> f64 {
    let start = Instant::now();
    let hits: u64 = keys.par_iter().map(|&k| f.contains(k) as u64).sum();
    std::hint::black_box(hits);
    let dt = start.elapsed().as_secs_f64();
    (keys.len() as f64) / dt
}

fn measure_query(f: &dyn StaticFilter, pos: &[u64], neg: &[u64]) -> QueryMetrics {
    let qps_sp = qps_single(f, pos);
    let qps_sn = qps_single(f, neg);
    let qps_mp = qps_multi(f, pos);
    let qps_mn = qps_multi(f, neg);

    let fp = neg.iter().filter(|&&k| f.contains(k)).count();
    let fp_rate = (fp as f64) / (neg.len() as f64);

    println!(
        "[QUERY] {:20} sp={:.0} qps  sn={:.0} qps  mp={:.0} qps  mn={:.0} qps  fp={:.6}",
        f.name(),
        qps_sp,
        qps_sn,
        qps_mp,
        qps_mn,
        fp_rate
    );

    QueryMetrics {
        qps_single_pos: qps_sp,
        qps_single_neg: qps_sn,
        qps_multi_pos: qps_mp,
        qps_multi_neg: qps_mn,
        fp_rate,
    }
}

#[derive(Debug, Serialize)]
struct OutputRow {
    name: String,
    n_keys: usize,
    seed: u64,
    build_wall_ms: f64,
    build_rss_kb: u64,
    qps_single_pos: f64,
    qps_single_neg: f64,
    qps_multi_pos: f64,
    qps_multi_neg: f64,
    fp_rate: f64,
}

fn write_csv(path: &Path, rows: &[OutputRow]) -> Result<()> {
    let mut writer =
        csv::Writer::from_path(path).with_context(|| format!("open csv {}", path.display()))?;
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn perf_hint() {
    println!(
        "[PERF] Hint: run `perf stat -e cycles,instructions,branches,branch-misses` on this binary"
    );
}

#[cfg(not(target_os = "linux"))]
fn perf_hint() {}

fn run_criterion(
    filters: &[Arc<dyn StaticFilter + Send + Sync>],
    positives: Arc<Vec<u64>>,
    negatives: Arc<Vec<u64>>,
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

    for filter in filters {
        let name = filter.name();

        let pos = Arc::clone(&positives);
        let filter_pos = Arc::clone(filter);
        c.bench_function(&format!("single_pos/{}", name), |b| {
            b.iter(|| {
                for &k in pos.iter() {
                    std::hint::black_box(filter_pos.contains(k));
                }
            });
        });

        let neg = Arc::clone(&negatives);
        let filter_neg = Arc::clone(filter);
        c.bench_function(&format!("single_neg/{}", name), |b| {
            b.iter(|| {
                for &k in neg.iter() {
                    std::hint::black_box(filter_neg.contains(k));
                }
            });
        });
    }
    c.final_summary();
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

fn env_u64(key: &str) -> Option<u64> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

fn env_f64(key: &str) -> Option<f64> {
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

/// Expands a u64 seed into a 32-byte array for bloom filter seeding.
///
/// The 8-byte seed is repeated 4 times to fill the 32-byte output.
/// This provides deterministic seeding for bloom filters that require [u8; 32].
fn seed_from_u64(seed: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    let bytes = seed.to_le_bytes();
    for chunk in out.chunks_exact_mut(8) {
        chunk.copy_from_slice(&bytes);
    }
    out
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg_text = std::fs::read_to_string(&args.config).context("read filterbench config")?;
    let cfg: RunCfg = toml::from_str(&cfg_text).context("parse filterbench config")?;

    let n_keys = env_usize("N_KEYS").unwrap_or(cfg.n_keys);
    let seed = env_u64("SEED").unwrap_or(cfg.seed);
    let bloom_fp_rate = env_f64("BLOOM_FP_RATE").unwrap_or(cfg.bloom_fp_rate);
    let scalable_fp_rate = env_f64("SCALABLE_FP_RATE").unwrap_or(cfg.scalable_fp_rate);
    let criterion_enabled = env_bool("CRITERION").unwrap_or(cfg.criterion);

    if n_keys == 0 {
        bail!("n_keys must be > 0");
    }
    if !(0.0 < bloom_fp_rate && bloom_fp_rate < 1.0) {
        bail!("bloom_fp_rate must be in (0,1)");
    }
    if !(0.0 < scalable_fp_rate && scalable_fp_rate < 1.0) {
        bail!("scalable_fp_rate must be in (0,1)");
    }
    if criterion_enabled {
        if cfg.criterion_sample_size == 0 {
            bail!("criterion_sample_size must be > 0");
        }
        if cfg.criterion_measurement_secs == 0 {
            bail!("criterion_measurement_secs must be > 0");
        }
    }

    // Thread configuration precedence:
    // 1. RAYON_NUM_THREADS (standard Rayon env var) - let Rayon handle it
    // 2. RAYON_THREADS env var - our custom override
    // 3. rayon_threads from config file
    // 4. None = Rayon's default (usually num_cpus)
    let mut rayon_threads = cfg.rayon_threads;
    if std::env::var("RAYON_NUM_THREADS").is_ok() {
        rayon_threads = None;
    } else {
        rayon_threads = env_usize("RAYON_THREADS").or(rayon_threads);
    }
    if let Some(threads) = rayon_threads.filter(|t| *t > 0) {
        if let Err(err) = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
        {
            eprintln!("rayon threadpool setup failed: {}", err);
        }
    }

    let criterion_cfg = CriterionCfg {
        enabled: criterion_enabled,
        sample_size: cfg.criterion_sample_size,
        warmup: Duration::from_millis(cfg.criterion_warmup_ms),
        measurement: Duration::from_secs(cfg.criterion_measurement_secs),
        plots: cfg.criterion_plots,
    };

    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("create output dir {}", args.out.display()))?;
    let out_csv = args.out.join("output.csv");
    if out_csv.exists() {
        bail!("output.csv already exists at {}", out_csv.display());
    }

    println!(
        "Generating dataset: {} positives + {} negatives (seed=0x{:x})",
        n_keys, n_keys, seed
    );
    let (pos, neg) = generate_keys(n_keys, seed);
    let positives = Arc::new(pos);
    let negatives = Arc::new(neg);

    let params = FilterParams {
        seed,
        bloom_fp_rate,
        scalable_fp_rate,
    };

    let mut filters: Vec<(Arc<dyn StaticFilter + Send + Sync>, BuildMetrics)> = Vec::new();

    {
        use xorf_impl::Xorf8 as Impl;
        let (f, m) = measure_build::<Impl>(&positives, params)?;
        filters.push((f, m));
    }
    {
        use xorf_binary_fuse_impl::XorfBinaryFuse8 as Impl;
        let (f, m) = measure_build::<Impl>(&positives, params)?;
        filters.push((f, m));
    }
    {
        use xorfilter_rs_impl::XorfilterRs as Impl;
        let (f, m) = measure_build::<Impl>(&positives, params)?;
        filters.push((f, m));
    }
    {
        use bloom_impl::BloomFilter as Impl;
        let (f, m) = measure_build::<Impl>(&positives, params)?;
        filters.push((f, m));
    }
    {
        use scalable_bloom_impl::ScalableBloom as Impl;
        let (f, m) = measure_build::<Impl>(&positives, params)?;
        filters.push((f, m));
    }

    println!("\n== RESULTS ==");
    perf_hint();

    let mut rows = Vec::with_capacity(filters.len());
    for (filter, build_metrics) in &filters {
        let query = measure_query(&**filter, &positives, &negatives);
        rows.push(OutputRow {
            name: filter.name().to_string(),
            n_keys,
            seed,
            build_wall_ms: build_metrics.wall.as_secs_f64() * 1000.0,
            build_rss_kb: build_metrics.peak_rss_kb,
            qps_single_pos: query.qps_single_pos,
            qps_single_neg: query.qps_single_neg,
            qps_multi_pos: query.qps_multi_pos,
            qps_multi_neg: query.qps_multi_neg,
            fp_rate: query.fp_rate,
        });
    }

    write_csv(&out_csv, &rows)?;

    run_criterion(
        &filters
            .iter()
            .map(|(f, _)| Arc::clone(f))
            .collect::<Vec<_>>(),
        Arc::clone(&positives),
        Arc::clone(&negatives),
        &criterion_cfg,
    );

    Ok(())
}
