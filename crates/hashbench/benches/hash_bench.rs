use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use hashbench::{all_hashfns, generate_input, DataPattern};

const SEED: u64 = 12648430;

fn throughput_benches(c: &mut Criterion) {
    let hash_fns = all_hashfns(SEED);

    for size in [8, 16, 32, 64, 256, 1024, 4096, 16384, 65536] {
        let input = generate_input(size, DataPattern::Random, SEED);
        let mut group = c.benchmark_group(format!("throughput/{}B", size));
        group.throughput(Throughput::Bytes(size as u64));

        for hash_fn in &hash_fns {
            group.bench_function(hash_fn.name(), |b| {
                b.iter(|| std::hint::black_box(hash_fn.hash_bytes(std::hint::black_box(&input))));
            });
        }
        group.finish();
    }
}

fn latency_benches(c: &mut Criterion) {
    let hash_fns = all_hashfns(SEED);

    for size in [64, 4096] {
        let input = generate_input(size, DataPattern::Random, SEED);
        let mut group = c.benchmark_group(format!("latency/{}B", size));

        for hash_fn in &hash_fns {
            group.bench_function(hash_fn.name(), |b| {
                b.iter(|| std::hint::black_box(hash_fn.hash_bytes(std::hint::black_box(&input))));
            });
        }
        group.finish();
    }
}

criterion_group!(benches, throughput_benches, latency_benches);
criterion_main!(benches);
