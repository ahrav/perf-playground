//! uring_bench: compare blocking pread ("readv"), io_uring fixed buffers ("reg"),
//! and io_uring buffer ring ("bufring") read paths.
//!
//! Output: output.csv (summary) + per-run latencies_*.csv files.

use anyhow::Result;

#[cfg(target_os = "linux")]
use anyhow::{bail, Context};
#[cfg(target_os = "linux")]
use clap::Parser;
#[cfg(target_os = "linux")]
use io_uring::{cqueue, opcode, squeue, types, IoUring};
#[cfg(target_os = "linux")]
use serde::Deserialize;
#[cfg(target_os = "linux")]
use std::alloc::{alloc_zeroed, dealloc, Layout};
#[cfg(target_os = "linux")]
use std::collections::VecDeque;
#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::io::Write;
#[cfg(target_os = "linux")]
use std::os::unix::io::AsRawFd;
#[cfg(target_os = "linux")]
use std::path::{Path, PathBuf};
#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicU16, Ordering};
#[cfg(target_os = "linux")]
use std::time::{Duration, Instant};

#[cfg(target_os = "linux")]
#[derive(Parser, Debug)]
#[command(about = "Compare blocking pread vs io_uring registered buffers and buf_ring")]
struct Args {
    /// Config file for sweeps. If omitted, a single run is executed from CLI flags.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Output directory (required for --config; optional for single runs).
    #[arg(long)]
    out: Option<PathBuf>,

    /// Mode for single runs: readv, reg, bufring
    #[arg(long)]
    mode: Option<String>,
    /// Chunk size in bytes for single runs
    #[arg(long)]
    chunk: Option<usize>,
    /// Queue depth for single runs
    #[arg(long)]
    qd: Option<usize>,
    /// Iterations for single runs
    #[arg(long)]
    iters: Option<usize>,
    /// Path to file for single runs
    #[arg(long)]
    path: Option<PathBuf>,
    /// Warmup duration in seconds for single runs
    #[arg(long, default_value_t = 5)]
    warmup_secs: u64,
    /// io_uring submission queue entries override
    #[arg(long)]
    ring_entries: Option<u32>,
    /// Buffer ring entries override (bufring mode only)
    #[arg(long)]
    buf_ring_entries: Option<u16>,
}

#[cfg(target_os = "linux")]
#[derive(Debug, Deserialize)]
struct RunCfg {
    /// Path to the file for I/O operations.
    path: String,
    /// Sizes of each I/O buffer in bytes.
    chunks: Vec<usize>,
    /// Queue depths to test.
    qds: Vec<usize>,
    /// Modes to test: readv, reg, bufring.
    modes: Vec<String>,
    /// Iterations per run.
    iters: usize,
    /// Warmup duration in seconds.
    #[serde(default = "default_warmup_secs")]
    warmup_secs: u64,
    /// io_uring submission queue entries override.
    ring_entries: Option<u32>,
    /// Buffer ring entries override (bufring mode only).
    buf_ring_entries: Option<u16>,
}

#[cfg(target_os = "linux")]
fn default_warmup_secs() -> u64 {
    5
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug)]
enum Mode {
    Readv,
    Reg,
    Bufring,
}

#[cfg(target_os = "linux")]
impl Mode {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "readv" => Ok(Self::Readv),
            "reg" => Ok(Self::Reg),
            "bufring" => Ok(Self::Bufring),
            _ => bail!("unknown mode: {s}"),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Readv => "readv",
            Self::Reg => "reg",
            Self::Bufring => "bufring",
        }
    }
}

#[cfg(target_os = "linux")]
#[derive(Clone, Debug)]
struct RunSpec {
    mode: Mode,
    chunk: usize,
    qd: usize,
    iters: usize,
    warmup: Duration,
    path: PathBuf,
    ring_entries: Option<u32>,
    buf_ring_entries: Option<u16>,
}

#[cfg(target_os = "linux")]
#[derive(Debug)]
struct RunResult {
    latencies_ns: Vec<u64>,
    wall: Duration,
    errors: usize,
}

#[cfg(target_os = "linux")]
#[derive(Debug)]
struct CsvRow {
    mode: String,
    chunk: usize,
    qd: usize,
    iters: usize,
    completed: usize,
    p50_ns: u64,
    p95_ns: u64,
    p99_ns: u64,
    mb_s: f64,
    ops_s: f64,
    errors: usize,
    latencies_csv: String,
}

#[cfg(target_os = "linux")]
fn main() -> Result<()> {
    let args = Args::parse();

    if let Some(cfg_path) = args.config {
        let out_dir = args.out.context("--out is required with --config")?;
        run_from_config(&cfg_path, &out_dir)
    } else {
        let spec = RunSpec {
            mode: Mode::parse(args.mode.as_deref().unwrap_or("readv"))?,
            chunk: args.chunk.unwrap_or(8192),
            qd: args.qd.unwrap_or(64),
            iters: args.iters.unwrap_or(100_000),
            warmup: Duration::from_secs(args.warmup_secs),
            path: args.path.unwrap_or_else(|| PathBuf::from("/tmp/testfile")),
            ring_entries: args.ring_entries,
            buf_ring_entries: args.buf_ring_entries,
        };
        let out_dir = args.out.unwrap_or_else(|| PathBuf::from("."));
        run_single(&spec, &out_dir)
    }
}

#[cfg(not(target_os = "linux"))]
fn main() -> Result<()> {
    eprintln!("uring_bench only supports Linux.");
    Ok(())
}

#[cfg(target_os = "linux")]
fn run_from_config(cfg_path: &Path, out_dir: &Path) -> Result<()> {
    let cfg_text = std::fs::read_to_string(cfg_path)
        .with_context(|| format!("read config {}", cfg_path.display()))?;
    let cfg: RunCfg = toml::from_str(&cfg_text)
        .with_context(|| format!("parse config {}", cfg_path.display()))?;

    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("create out dir {}", out_dir.display()))?;

    let rows = run_sweep(&cfg, out_dir)?;
    write_csv(out_dir.join("output.csv"), &rows)?;

    Ok(())
}

#[cfg(target_os = "linux")]
fn run_single(spec: &RunSpec, out_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("create out dir {}", out_dir.display()))?;

    let row = run_and_record(spec, out_dir)?;
    write_csv(out_dir.join("output.csv"), &[row])?;

    Ok(())
}

#[cfg(target_os = "linux")]
fn run_sweep(cfg: &RunCfg, out_dir: &Path) -> Result<Vec<CsvRow>> {
    if cfg.chunks.is_empty() {
        bail!("chunks must not be empty");
    }
    if cfg.qds.is_empty() {
        bail!("qds must not be empty");
    }
    if cfg.modes.is_empty() {
        bail!("modes must not be empty");
    }
    if cfg.iters == 0 {
        bail!("iters must be > 0");
    }

    let mut rows = Vec::new();
    for mode_s in &cfg.modes {
        let mode = Mode::parse(mode_s)?;
        for &chunk in &cfg.chunks {
            for &qd in &cfg.qds {
                let spec = RunSpec {
                    mode,
                    chunk,
                    qd,
                    iters: cfg.iters,
                    warmup: Duration::from_secs(cfg.warmup_secs),
                    path: PathBuf::from(&cfg.path),
                    ring_entries: cfg.ring_entries,
                    buf_ring_entries: cfg.buf_ring_entries,
                };
                let row = run_and_record(&spec, out_dir)?;
                rows.push(row);
            }
        }
    }
    Ok(rows)
}

#[cfg(target_os = "linux")]
fn run_and_record(spec: &RunSpec, out_dir: &Path) -> Result<CsvRow> {
    validate_spec(spec)?;

    let result = run_one(spec)?;

    let lat_name = format!(
        "latencies_{}_chunk{}_qd{}.csv",
        spec.mode.as_str(),
        spec.chunk,
        spec.qd
    );
    write_latencies(&out_dir.join(&lat_name), &result.latencies_ns)?;

    let (p50, p95, p99) = percentiles_ns(&result.latencies_ns);
    let completed = result.latencies_ns.len();
    let (mb_s, ops_s) = throughput(spec.chunk, completed, result.wall);

    println!(
        "mode={}, chunk={}, qd={}, samples={}, p50={} ns, p95={} ns, p99={} ns, MB/s={:.1}",
        spec.mode.as_str(),
        spec.chunk,
        spec.qd,
        completed,
        p50,
        p95,
        p99,
        mb_s
    );

    Ok(CsvRow {
        mode: spec.mode.as_str().to_string(),
        chunk: spec.chunk,
        qd: spec.qd,
        iters: spec.iters,
        completed,
        p50_ns: p50,
        p95_ns: p95,
        p99_ns: p99,
        mb_s,
        ops_s,
        errors: result.errors,
        latencies_csv: lat_name,
    })
}

#[cfg(target_os = "linux")]
fn validate_spec(spec: &RunSpec) -> Result<()> {
    if spec.chunk == 0 {
        bail!("chunk must be > 0");
    }
    if spec.qd == 0 {
        bail!("qd must be > 0");
    }
    if spec.iters == 0 {
        bail!("iters must be > 0");
    }
    if spec.chunk > u32::MAX as usize {
        bail!("chunk too large for io_uring read: {}", spec.chunk);
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn run_one(spec: &RunSpec) -> Result<RunResult> {
    let file =
        File::open(&spec.path).with_context(|| format!("open file {}", spec.path.display()))?;
    let fd = file.as_raw_fd();
    let file_sz = file.metadata()?.len() as usize;
    if file_sz < spec.chunk {
        bail!("file is smaller than chunk size");
    }

    match spec.mode {
        Mode::Readv => run_readv(fd, file_sz, spec),
        Mode::Reg => run_reg(fd, file_sz, spec),
        Mode::Bufring => run_bufring(fd, file_sz, spec),
    }
}

#[cfg(target_os = "linux")]
fn run_readv(fd: i32, file_sz: usize, spec: &RunSpec) -> Result<RunResult> {
    let mut off = 0usize;
    let mut buf = vec![0u8; spec.chunk];

    let warm_start = Instant::now();
    while warm_start.elapsed() < spec.warmup {
        let res = unsafe { libc::pread(fd, buf.as_mut_ptr() as *mut _, spec.chunk, off as i64) };
        if res < 0 {
            bail!("pread failed during warmup");
        }
        off = (off + spec.chunk) % file_sz;
    }

    off = 0;
    let start = Instant::now();
    let mut latencies_ns = Vec::with_capacity(spec.iters);
    let mut errors = 0usize;

    for _ in 0..spec.iters {
        let t0 = Instant::now();
        let res = unsafe { libc::pread(fd, buf.as_mut_ptr() as *mut _, spec.chunk, off as i64) };
        if res < 0 {
            errors += 1;
        }
        let ns = t0.elapsed().as_nanos() as u64;
        latencies_ns.push(ns.max(1));
        off = (off + spec.chunk) % file_sz;
    }

    Ok(RunResult {
        latencies_ns,
        wall: start.elapsed(),
        errors,
    })
}

#[cfg(target_os = "linux")]
fn run_reg(fd: i32, file_sz: usize, spec: &RunSpec) -> Result<RunResult> {
    let qd = spec.qd;
    let pool_len = qd.max(32);
    if pool_len > u16::MAX as usize {
        bail!("buffer pool too large for fixed buffer index");
    }

    let ring_entries = spec.ring_entries.unwrap_or(qd as u32).max(qd as u32);
    let mut ring = IoUring::new(ring_entries)?;

    let mut pool: Vec<Vec<u8>> = (0..pool_len).map(|_| vec![0u8; spec.chunk]).collect();
    let iov: Vec<libc::iovec> = pool
        .iter_mut()
        .map(|b| libc::iovec {
            iov_base: b.as_mut_ptr() as *mut _,
            iov_len: b.len(),
        })
        .collect();
    unsafe {
        ring.submitter().register_buffers(&iov)?;
    }

    let mut off = 0usize;
    warmup_reg(
        &mut ring,
        fd,
        &pool,
        spec.chunk,
        qd,
        &mut off,
        file_sz,
        spec.warmup,
    )?;

    off = 0;
    let mut latencies_ns = Vec::with_capacity(spec.iters);
    let mut start_ns = vec![0u64; spec.iters];
    let mut buf_for_op: Vec<u16> = vec![0; spec.iters];
    let mut available: VecDeque<u16> = (0..pool_len as u16).collect();

    let mut submitted = 0usize;
    let mut completed = 0usize;
    let mut in_flight = 0usize;
    let mut errors = 0usize;
    let base = Instant::now();

    while completed < spec.iters {
        while submitted < spec.iters && in_flight < qd {
            let buf_idx = available
                .pop_front()
                .context("no available fixed buffers")?;
            let buf = &pool[buf_idx as usize];
            let t0 = base.elapsed().as_nanos() as u64;
            start_ns[submitted] = t0;
            buf_for_op[submitted] = buf_idx;

            let entry = opcode::ReadFixed::new(
                types::Fd(fd),
                buf.as_ptr() as *mut u8,
                spec.chunk as u32,
                buf_idx,
            )
            .offset(off as u64)
            .build()
            .user_data(submitted as u64);

            push_entry(&mut ring, entry)?;

            off = (off + spec.chunk) % file_sz;
            submitted += 1;
            in_flight += 1;
        }

        ring.submit_and_wait(1)?;

        let mut cq = ring.completion();
        for cqe in &mut cq {
            let op_idx = cqe.user_data() as usize;
            if op_idx >= spec.iters {
                bail!("completion user_data out of range");
            }
            let res = cqe.result();
            if res < 0 {
                errors += 1;
            }
            let t1 = base.elapsed().as_nanos() as u64;
            let latency = t1.saturating_sub(start_ns[op_idx]).max(1);
            latencies_ns.push(latency);

            let buf_idx = buf_for_op[op_idx];
            available.push_back(buf_idx);

            completed += 1;
            in_flight = in_flight.saturating_sub(1);
        }
    }

    Ok(RunResult {
        latencies_ns,
        wall: base.elapsed(),
        errors,
    })
}

#[cfg(target_os = "linux")]
fn run_bufring(fd: i32, file_sz: usize, spec: &RunSpec) -> Result<RunResult> {
    let qd = spec.qd;
    let ring_entries = spec.ring_entries.unwrap_or(qd as u32).max(qd as u32);
    let mut ring = IoUring::new(ring_entries)?;

    let mut bufring = BufRing::new(resolve_bufring_entries(spec)?, 0, spec.chunk)?;
    unsafe {
        ring.submitter().register_buf_ring_with_flags(
            bufring.addr(),
            bufring.entries(),
            bufring.bgid(),
            0,
        )?;
    }
    bufring.init()?;

    let mut off = 0usize;
    warmup_bufring(
        &mut ring,
        fd,
        &mut bufring,
        spec.chunk,
        qd,
        &mut off,
        file_sz,
        spec.warmup,
    )?;

    off = 0;
    let mut latencies_ns = Vec::with_capacity(spec.iters);
    let mut start_ns = vec![0u64; spec.iters];

    let mut submitted = 0usize;
    let mut completed = 0usize;
    let mut in_flight = 0usize;
    let mut errors = 0usize;
    let base = Instant::now();

    while completed < spec.iters {
        while submitted < spec.iters && in_flight < qd {
            let t0 = base.elapsed().as_nanos() as u64;
            start_ns[submitted] = t0;

            let entry = opcode::Read::new(types::Fd(fd), std::ptr::null_mut(), spec.chunk as u32)
                .offset(off as u64)
                .buf_group(bufring.bgid())
                .build()
                .flags(squeue::Flags::BUFFER_SELECT)
                .user_data(submitted as u64);

            push_entry(&mut ring, entry)?;

            off = (off + spec.chunk) % file_sz;
            submitted += 1;
            in_flight += 1;
        }

        ring.submit_and_wait(1)?;

        let mut cq = ring.completion();
        for cqe in &mut cq {
            let op_idx = cqe.user_data() as usize;
            if op_idx >= spec.iters {
                bail!("completion user_data out of range");
            }
            let res = cqe.result();
            if res < 0 {
                errors += 1;
            }
            let t1 = base.elapsed().as_nanos() as u64;
            let latency = t1.saturating_sub(start_ns[op_idx]).max(1);
            latencies_ns.push(latency);

            if let Some(bid) = cqueue::buffer_select(cqe.flags()) {
                if let Err(err) = bufring.recycle(bid) {
                    errors += 1;
                    eprintln!("bufring recycle error: {err}");
                }
            }

            completed += 1;
            in_flight = in_flight.saturating_sub(1);
        }
    }

    ring.submitter().unregister_buf_ring(bufring.bgid())?;

    Ok(RunResult {
        latencies_ns,
        wall: base.elapsed(),
        errors,
    })
}

#[cfg(target_os = "linux")]
fn warmup_reg(
    ring: &mut IoUring,
    fd: i32,
    pool: &[Vec<u8>],
    chunk: usize,
    qd: usize,
    off: &mut usize,
    file_sz: usize,
    warmup: Duration,
) -> Result<()> {
    let start = Instant::now();
    while start.elapsed() < warmup {
        let mut sq = ring.submission();
        for i in 0..qd {
            let buf_idx = i % pool.len();
            let buf = &pool[buf_idx];
            let entry = opcode::ReadFixed::new(
                types::Fd(fd),
                buf.as_ptr() as *mut u8,
                chunk as u32,
                buf_idx as u16,
            )
            .offset(*off as u64)
            .build();
            unsafe {
                sq.push(&entry).context("sq full during warmup")?;
            }
            *off = (*off + chunk) % file_sz;
        }
        drop(sq);
        ring.submit_and_wait(qd as u32)?;
        let mut cq = ring.completion();
        for _ in 0..qd {
            let cqe = cq.next().context("warmup CQ underflow")?;
            if cqe.result() < 0 {
                bail!("read_fixed failed during warmup");
            }
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn warmup_bufring(
    ring: &mut IoUring,
    fd: i32,
    bufring: &mut BufRing,
    chunk: usize,
    qd: usize,
    off: &mut usize,
    file_sz: usize,
    warmup: Duration,
) -> Result<()> {
    let start = Instant::now();
    while start.elapsed() < warmup {
        let mut sq = ring.submission();
        for _ in 0..qd {
            let entry = opcode::Read::new(types::Fd(fd), std::ptr::null_mut(), chunk as u32)
                .offset(*off as u64)
                .buf_group(bufring.bgid())
                .build()
                .flags(squeue::Flags::BUFFER_SELECT);
            unsafe {
                sq.push(&entry).context("sq full during warmup")?;
            }
            *off = (*off + chunk) % file_sz;
        }
        drop(sq);
        ring.submit_and_wait(qd as u32)?;
        let mut cq = ring.completion();
        for _ in 0..qd {
            let cqe = cq.next().context("warmup CQ underflow")?;
            if cqe.result() < 0 {
                bail!("read failed during warmup");
            }
            if let Some(bid) = cqueue::buffer_select(cqe.flags()) {
                bufring.recycle(bid)?;
            }
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn resolve_bufring_entries(spec: &RunSpec) -> Result<u16> {
    let mut entries = spec.buf_ring_entries.unwrap_or(spec.qd.max(32) as u16);
    if entries < spec.qd as u16 {
        entries = spec.qd as u16;
    }
    let entries_u32 = entries as u32;
    let pow2 = entries_u32.next_power_of_two();
    if pow2 > 32768 {
        bail!("buf_ring_entries too large (max 32768)");
    }
    Ok(pow2 as u16)
}

#[cfg(target_os = "linux")]
fn push_entry(ring: &mut IoUring, entry: squeue::Entry) -> Result<()> {
    let mut sq = ring.submission();
    unsafe {
        sq.push(&entry)
            .map_err(|_| anyhow::anyhow!("sq overflow"))?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn percentiles_ns(samples: &[u64]) -> (u64, u64, u64) {
    if samples.is_empty() {
        return (0, 0, 0);
    }
    let mut s = samples.to_vec();
    s.sort_unstable();
    let p = |q: f64| -> u64 {
        let idx = ((s.len() as f64 - 1.0) * q).round() as usize;
        s[idx]
    };
    (p(0.50), p(0.95), p(0.99))
}

#[cfg(target_os = "linux")]
fn throughput(chunk: usize, completed: usize, wall: Duration) -> (f64, f64) {
    if wall.as_secs_f64() == 0.0 {
        return (0.0, 0.0);
    }
    let bytes = completed as f64 * chunk as f64;
    let mb_s = (bytes / (1024.0 * 1024.0)) / wall.as_secs_f64();
    let ops_s = (completed as f64) / wall.as_secs_f64();
    (mb_s, ops_s)
}

#[cfg(target_os = "linux")]
fn write_latencies(path: &Path, latencies: &[u64]) -> Result<()> {
    let mut f =
        File::create(path).with_context(|| format!("create latencies file {}", path.display()))?;
    for v in latencies {
        writeln!(f, "{}", v)?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn write_csv(path: PathBuf, rows: &[CsvRow]) -> Result<()> {
    let mut f =
        File::create(&path).with_context(|| format!("create output csv {}", path.display()))?;
    writeln!(
        f,
        "mode,chunk,qd,iters,completed,p50_ns,p95_ns,p99_ns,mb_s,ops_s,errors,latencies_csv"
    )?;
    for row in rows {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{:.3},{:.3},{},{}",
            row.mode,
            row.chunk,
            row.qd,
            row.iters,
            row.completed,
            row.p50_ns,
            row.p95_ns,
            row.p99_ns,
            row.mb_s,
            row.ops_s,
            row.errors,
            row.latencies_csv
        )?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
struct BufRing {
    ring: *mut types::BufRingEntry,
    ring_layout: Layout,
    entries: u16,
    mask: u16,
    tail: u16,
    tail_ptr: *const AtomicU16,
    bgid: u16,
    buffers: Vec<Vec<u8>>,
}

#[cfg(target_os = "linux")]
impl BufRing {
    fn new(entries: u16, bgid: u16, chunk: usize) -> Result<Self> {
        let entries_usize = entries as usize;
        let ring_bytes = entries_usize
            .checked_mul(std::mem::size_of::<types::BufRingEntry>())
            .context("buf ring size overflow")?;
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size <= 0 {
            bail!("invalid page size");
        }
        let layout = Layout::from_size_align(ring_bytes, page_size as usize)
            .context("invalid buf ring layout")?;
        let ring_ptr = unsafe { alloc_zeroed(layout) } as *mut types::BufRingEntry;
        if ring_ptr.is_null() {
            bail!("buf ring allocation failed");
        }

        let tail_ptr = unsafe { types::BufRingEntry::tail(ring_ptr as *const types::BufRingEntry) };
        let tail_ptr = tail_ptr as *const AtomicU16;

        let mut buffers = Vec::with_capacity(entries_usize);
        for _ in 0..entries_usize {
            buffers.push(vec![0u8; chunk]);
        }

        Ok(Self {
            ring: ring_ptr,
            ring_layout: layout,
            entries,
            mask: entries - 1,
            tail: 0,
            tail_ptr,
            bgid,
            buffers,
        })
    }

    fn addr(&self) -> u64 {
        self.ring as u64
    }

    fn entries(&self) -> u16 {
        self.entries
    }

    fn bgid(&self) -> u16 {
        self.bgid
    }

    fn init(&mut self) -> Result<()> {
        for bid in 0..self.entries {
            self.push_at_tail(bid)?;
            self.tail = self.tail.wrapping_add(1);
        }
        self.publish_tail();
        Ok(())
    }

    fn recycle(&mut self, bid: u16) -> Result<()> {
        self.push_at_tail(bid)?;
        self.tail = self.tail.wrapping_add(1);
        self.publish_tail();
        Ok(())
    }

    fn push_at_tail(&mut self, bid: u16) -> Result<()> {
        let idx = (self.tail as usize) & (self.mask as usize);
        let buf_idx = bid as usize;
        if buf_idx >= self.buffers.len() {
            bail!("bufring bid out of range: {}", bid);
        }
        let buf = &self.buffers[buf_idx];
        let entry = unsafe { &mut *self.ring.add(idx) };
        entry.set_addr(buf.as_ptr() as u64);
        entry.set_len(buf.len() as u32);
        entry.set_bid(bid);
        Ok(())
    }

    fn publish_tail(&self) {
        unsafe {
            (*self.tail_ptr).store(self.tail, Ordering::Release);
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for BufRing {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ring.cast(), self.ring_layout);
        }
    }
}
