//! io_uring queue depth sweep benchmark.
//!
//! Measures sequential I/O throughput and latency across varying queue depths using
//! Linux's io_uring interface. Supports both read and write operations with optional
//! O_DIRECT, SQPOLL, IOPOLL, and fixed buffer modes.
//!
//! # Requirements
//!
//! - Linux kernel 5.10+ (for io_uring features used)
//! - Elevated privileges for SQPOLL mode
//! - Aligned buffers when using O_DIRECT (page-aligned, typically 4KB)
//!
//! # Output
//!
//! Produces `output.csv` with columns:
//! - `qd`: Queue depth tested
//! - `mb_s`: Throughput in MB/s
//! - `ops_s`: Operations per second
//! - `p50_ms`, `p95_ms`, `p99_ms`: Latency percentiles in milliseconds
//! - `errors`: Number of I/O errors encountered
//!
//! # Example
//!
//! ```bash
//! ./io_uring_queue_depth_sweep --config config.toml --out ./results
//! ```

use anyhow::Result;
#[cfg(target_os = "linux")]
use anyhow::{bail, Context};
#[cfg(target_os = "linux")]
use clap::Parser;
#[cfg(target_os = "linux")]
use serde::Deserialize;
#[cfg(target_os = "linux")]
use std::alloc::{alloc_zeroed, dealloc, Layout};
#[cfg(target_os = "linux")]
use std::path::PathBuf;

#[cfg(target_os = "linux")]
use hdrhistogram::Histogram;
#[cfg(target_os = "linux")]
use io_uring::{opcode, types, IoUring};
#[cfg(target_os = "linux")]
use std::collections::VecDeque;
#[cfg(target_os = "linux")]
use std::fs::OpenOptions;
#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt;
#[cfg(target_os = "linux")]
use std::os::unix::io::AsRawFd;
#[cfg(target_os = "linux")]
use std::time::Instant;

#[cfg(target_os = "linux")]
#[derive(Parser, Debug)]
#[command(about = "io_uring queue depth sweep")]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    out: PathBuf,
}

/// Benchmark configuration loaded from TOML.
#[cfg(target_os = "linux")]
#[derive(Deserialize, Debug, Clone)]
struct RunCfg {
    /// Path to the file for I/O operations.
    path: String,
    /// Size of each I/O buffer in bytes. Must be > 0 and page-aligned for O_DIRECT.
    buf_size: usize,
    /// Total file length to read/write. Must be a multiple of `buf_size`.
    file_len: u64,
    /// If true, perform writes; otherwise reads.
    write: bool,
    /// Use O_DIRECT for unbuffered I/O. Requires page-aligned buffers.
    direct: bool,
    /// Use SQPOLL mode (kernel-side submission polling). Requires elevated privileges.
    sqpoll: bool,
    /// Use IOPOLL mode for polling completions. Requires `direct=true`.
    iopoll: bool,
    /// Use fixed/registered buffers for reduced syscall overhead.
    use_fixed: bool,
    /// Pre-allocate the file to `file_len` before writing.
    preallocate: Option<bool>,
    /// Queue depths to test. Each value must be > 0.
    qds: Vec<usize>,
    /// Ring size override. Defaults to next power of 2 >= max(qds) * 2.
    ring_entries: Option<u32>,
}

/// Results for a single queue depth run.
#[cfg(target_os = "linux")]
#[derive(Debug, Clone)]
struct ResultRow {
    /// Queue depth used for this run.
    qd: usize,
    /// Throughput in megabytes per second.
    mb_s: f64,
    /// Operations (reads or writes) completed per second.
    ops_s: f64,
    /// 50th percentile latency in milliseconds.
    p50_ms: f64,
    /// 95th percentile latency in milliseconds.
    p95_ms: f64,
    /// 99th percentile latency in milliseconds.
    p99_ms: f64,
    /// Number of I/O operations that returned errors.
    errors: usize,
}

/// Page-aligned buffer for O_DIRECT I/O operations.
///
/// O_DIRECT requires buffers to be aligned to the filesystem's logical block size
/// (typically page size, 4KB). This struct manages allocation and deallocation of
/// such aligned memory.
#[cfg(target_os = "linux")]
struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
    layout: Layout,
}

#[cfg(target_os = "linux")]
impl AlignedBuf {
    fn new(len: usize, alignment: usize) -> Result<Self> {
        let layout = Layout::from_size_align(len, alignment).context("invalid alignment")?;
        // SAFETY: Layout is validated above. The returned pointer is checked for null below.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            bail!("aligned allocation failed");
        }
        Ok(Self { ptr, len, layout })
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(target_os = "linux")]
impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: self.ptr was allocated with self.layout in new(), and this is the only
        // deallocation point. AlignedBuf has sole ownership of this allocation.
        unsafe { dealloc(self.ptr, self.layout) };
    }
}

#[cfg(target_os = "linux")]
fn main() -> Result<()> {
    let args = Args::parse();
    let cfg_text = std::fs::read_to_string(&args.config)
        .with_context(|| format!("read config {}", args.config.display()))?;
    let cfg: RunCfg = toml::from_str(&cfg_text)
        .with_context(|| format!("parse config {}", args.config.display()))?;

    let out_dir = &args.out;
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("create out dir {}", out_dir.display()))?;

    let rows = run_sweep(&cfg)?;
    write_csv(out_dir.join("output.csv"), &rows)?;

    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() -> Result<()> {
    eprintln!("io_uring_queue_depth_sweep only supports Linux.");
    Ok(())
}

/// Runs the queue depth sweep benchmark.
///
/// For each queue depth in `cfg.qds`, performs all I/O operations (file_len / buf_size)
/// and measures throughput and latency percentiles.
///
/// Uses a producer-consumer pattern:
/// 1. Submit operations up to the current queue depth limit
/// 2. Wait for at least one completion
/// 3. Reap completions and record latencies
/// 4. Repeat until all operations complete
#[cfg(target_os = "linux")]
fn run_sweep(cfg: &RunCfg) -> Result<Vec<ResultRow>> {
    if cfg.qds.is_empty() {
        bail!("qds must not be empty");
    }
    if cfg.buf_size == 0 {
        bail!("buf_size must be > 0");
    }
    if cfg.file_len == 0 {
        bail!("file_len must be > 0");
    }
    if cfg.file_len % cfg.buf_size as u64 != 0 {
        bail!("file_len must be a multiple of buf_size");
    }
    if cfg.qds.iter().any(|&qd| qd == 0) {
        bail!("queue depths must be > 0");
    }

    let alignment = direct_alignment(cfg.direct);
    if cfg.direct && cfg.buf_size % alignment != 0 {
        bail!(
            "buf_size must be a multiple of alignment for O_DIRECT (alignment={})",
            alignment
        );
    }
    if cfg.iopoll && !cfg.direct {
        bail!("iopoll requires direct=true");
    }

    let max_qd = *cfg.qds.iter().max().unwrap();
    if max_qd > u16::MAX as usize && cfg.use_fixed {
        bail!("max qd exceeds u16::MAX required for fixed buffers");
    }

    let ring_entries = cfg.ring_entries.unwrap_or_else(|| {
        let mut entries = (max_qd * 2).max(1);
        entries = entries.next_power_of_two();
        entries as u32
    });
    if ring_entries < max_qd as u32 {
        bail!("ring_entries must be >= max qd");
    }

    let mut builder = IoUring::builder();
    if cfg.sqpoll {
        builder.setup_sqpoll(1);
    }
    if cfg.iopoll {
        builder.setup_iopoll();
    }
    let mut ring = builder.build(ring_entries)?;

    let mut bufs: Vec<AlignedBuf> = (0..max_qd)
        .map(|_| AlignedBuf::new(cfg.buf_size, alignment))
        .collect::<Result<_>>()?;

    if cfg.use_fixed {
        let iovecs: Vec<libc::iovec> = bufs
            .iter_mut()
            .map(|b| libc::iovec {
                iov_base: b.as_mut_ptr() as *mut _,
                iov_len: b.len(),
            })
            .collect();
        // SAFETY: The buffers in `bufs` outlive all io_uring operations. They are not
        // dropped until after run_sweep returns, and all completions are reaped before then.
        unsafe { ring.submitter().register_buffers(&iovecs)? };
    }

    let mut open_opts = OpenOptions::new();
    if cfg.write {
        open_opts.write(true).create(true).read(true);
    } else {
        open_opts.read(true);
    }

    if cfg.direct {
        #[cfg(target_os = "linux")]
        {
            open_opts.custom_flags(libc::O_DIRECT);
        }
        #[cfg(not(target_os = "linux"))]
        {
            bail!("direct=true is only supported on linux");
        }
    }

    let file = open_opts
        .open(&cfg.path)
        .with_context(|| format!("open {}", cfg.path))?;
    if cfg.write && cfg.preallocate.unwrap_or(false) {
        file.set_len(cfg.file_len)
            .context("preallocate file_len failed")?;
    }
    if !cfg.write {
        let len = file.metadata()?.len();
        if len < cfg.file_len {
            bail!(
                "file_len ({}) is larger than file size ({})",
                cfg.file_len,
                len
            );
        }
    }

    let fd = file.as_raw_fd();
    let total_ops = (cfg.file_len / cfg.buf_size as u64) as usize;

    let mut rows = Vec::new();
    for &qd in &cfg.qds {
        let mut hist = Histogram::<u64>::new(3)?;
        let mut start_us: Vec<u64> = vec![0; total_ops];
        let mut buf_for_op: Vec<usize> = vec![0; total_ops];
        let mut available: VecDeque<usize> = (0..max_qd).collect();

        let mut submitted = 0usize;
        let mut completed = 0usize;
        let mut in_flight = 0usize;
        let mut errors = 0usize;

        let base = Instant::now();

        while completed < total_ops {
            // Phase 1: Submit operations up to queue depth limit
            while submitted < total_ops && in_flight < qd {
                let buf_idx = available.pop_front().context("no available buffers")?;
                let offset = (submitted as u64) * (cfg.buf_size as u64);
                let t0_us = base.elapsed().as_micros() as u64;
                start_us[submitted] = t0_us;
                buf_for_op[submitted] = buf_idx;

                submit_one(
                    &mut ring,
                    cfg,
                    fd,
                    &mut bufs[buf_idx],
                    buf_idx as u16,
                    offset,
                    submitted as u64,
                )?;

                submitted += 1;
                in_flight += 1;
            }

            // Phase 2: Submit to kernel and wait for at least one completion
            ring.submit_and_wait(1)?;

            // Phase 3: Reap completions and record latencies
            let mut cq = ring.completion();
            for cqe in &mut cq {
                let op_idx = cqe.user_data() as usize;
                let res = cqe.result();
                if res < 0 || res as usize != cfg.buf_size {
                    errors += 1;
                }
                let t1_us = base.elapsed().as_micros() as u64;
                let t0_us = start_us[op_idx];
                let latency_us = t1_us.saturating_sub(t0_us).max(1);
                hist.record(latency_us)?;

                // Return buffer to available pool for reuse
                let buf_idx = buf_for_op[op_idx];
                available.push_back(buf_idx);

                completed += 1;
                in_flight = in_flight.saturating_sub(1);
            }
        }

        let secs = base.elapsed().as_secs_f64();
        let total_bytes = (submitted as u64) * (cfg.buf_size as u64);
        let mb = total_bytes as f64 / (1024.0 * 1024.0);
        let mb_s = mb / secs;
        let ops_s = submitted as f64 / secs;
        let p50 = hist.value_at_quantile(0.50) as f64 / 1000.0;
        let p95 = hist.value_at_quantile(0.95) as f64 / 1000.0;
        let p99 = hist.value_at_quantile(0.99) as f64 / 1000.0;

        rows.push(ResultRow {
            qd,
            mb_s,
            ops_s,
            p50_ms: p50,
            p95_ms: p95,
            p99_ms: p99,
            errors,
        });
    }

    Ok(rows)
}

/// Submits a single I/O operation to the ring.
///
/// Creates and pushes a read or write SQE depending on `cfg.write` and `cfg.use_fixed`.
/// The `user_data` is used to correlate completions with submission timestamps.
#[cfg(target_os = "linux")]
fn submit_one(
    ring: &mut IoUring,
    cfg: &RunCfg,
    fd: i32,
    buf: &mut AlignedBuf,
    buf_index: u16,
    offset: u64,
    user_data: u64,
) -> Result<()> {
    let len = cfg.buf_size as u32;

    let entry = if cfg.use_fixed {
        if cfg.write {
            opcode::WriteFixed::new(types::Fd(fd), buf.as_ptr(), len, buf_index)
                .offset(offset)
                .build()
        } else {
            opcode::ReadFixed::new(types::Fd(fd), buf.as_mut_ptr(), len, buf_index)
                .offset(offset)
                .build()
        }
    } else if cfg.write {
        opcode::Write::new(types::Fd(fd), buf.as_ptr(), len)
            .offset(offset)
            .build()
    } else {
        opcode::Read::new(types::Fd(fd), buf.as_mut_ptr(), len)
            .offset(offset)
            .build()
    };

    let entry = entry.user_data(user_data);
    // SAFETY: The buffer (`buf`) remains valid and is not reused until its corresponding
    // completion is reaped. Buffer indices are tracked in buf_for_op and returned to the
    // available pool only after completion.
    unsafe {
        ring.submission()
            .push(&entry)
            .context("submission queue is full")?;
    }

    Ok(())
}

#[cfg(target_os = "linux")]
fn write_csv(path: PathBuf, rows: &[ResultRow]) -> Result<()> {
    let mut wtr =
        csv::Writer::from_path(&path).with_context(|| format!("create csv {}", path.display()))?;

    wtr.write_record([
        "qd", "mb_s", "ops_s", "p50_ms", "p95_ms", "p99_ms", "errors",
    ])?;

    for row in rows {
        wtr.write_record([
            row.qd.to_string(),
            format!("{:.1}", row.mb_s),
            format!("{:.0}", row.ops_s),
            format!("{:.3}", row.p50_ms),
            format!("{:.3}", row.p95_ms),
            format!("{:.3}", row.p99_ms),
            row.errors.to_string(),
        ])?;
    }
    wtr.flush()?;

    Ok(())
}

/// Returns the required buffer alignment for O_DIRECT or default alignment.
///
/// O_DIRECT requires page-aligned buffers. Queries the system page size via sysconf.
/// Falls back to 4096 if sysconf fails. Returns 8 for non-direct I/O.
#[cfg(target_os = "linux")]
fn direct_alignment(direct: bool) -> usize {
    if !direct {
        return 8;
    }
    // SAFETY: sysconf is safe to call with _SC_PAGESIZE. Returns -1 on error.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page > 0 {
        page as usize
    } else {
        4096
    }
}
