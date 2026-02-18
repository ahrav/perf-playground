use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use serde::{Deserialize, Serialize};

/// Object-safe trait for benchmarking hash functions.
pub trait HashFn: Sync + Send {
    /// Algorithm name for CSV output and display.
    fn name(&self) -> &'static str;

    /// Hash input bytes, returning a u64 digest (wider outputs truncated to first 8 bytes LE).
    fn hash_bytes(&self, input: &[u8]) -> u64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataPattern {
    Random,
    LowEntropy,
    Repeating,
}

impl std::fmt::Display for DataPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataPattern::Random => write!(f, "random"),
            DataPattern::LowEntropy => write!(f, "low_entropy"),
            DataPattern::Repeating => write!(f, "repeating"),
        }
    }
}

/// Generate deterministic input data of the given size and pattern.
pub fn generate_input(size: usize, pattern: DataPattern, seed: u64) -> Vec<u8> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    match pattern {
        DataPattern::Random => {
            let mut buf = vec![0u8; size];
            rng.fill(&mut buf[..]);
            buf
        }
        DataPattern::LowEntropy => {
            // Values drawn from a 4-byte alphabet.
            (0..size).map(|_| rng.gen::<u8>() & 0x03).collect()
        }
        DataPattern::Repeating => {
            let byte: u8 = rng.gen();
            vec![byte; size]
        }
    }
}

// ---- Key derivation helpers -----------------------------------------------------------

fn blake3_key_from_seed(seed: u64) -> [u8; 32] {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut key = [0u8; 32];
    rng.fill(&mut key);
    key
}

fn siphash_keys_from_seed(seed: u64) -> (u64, u64) {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    (rng.gen(), rng.gen())
}

fn highway_key_from_seed(seed: u64) -> [u64; 4] {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    [rng.gen(), rng.gen(), rng.gen(), rng.gen()]
}

fn four_seeds_from_seed(seed: u64) -> (u64, u64, u64, u64) {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    (rng.gen(), rng.gen(), rng.gen(), rng.gen())
}

// ---- Algorithm implementations --------------------------------------------------------

mod blake3_impl {
    use super::*;

    pub struct Blake3Hash;

    impl HashFn for Blake3Hash {
        fn name(&self) -> &'static str {
            "blake3"
        }
        fn hash_bytes(&self, input: &[u8]) -> u64 {
            let hash = blake3::hash(input);
            u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap())
        }
    }
}

mod blake3_keyed_impl {
    use super::*;

    pub struct Blake3Keyed {
        pub key: [u8; 32],
    }

    impl HashFn for Blake3Keyed {
        fn name(&self) -> &'static str {
            "blake3_keyed"
        }
        fn hash_bytes(&self, input: &[u8]) -> u64 {
            let hash = blake3::keyed_hash(&self.key, input);
            u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap())
        }
    }
}

mod highway_impl {
    use super::*;
    use highway::{HighwayHash, HighwayHasher, Key};

    pub struct Highway {
        pub key: Key,
    }

    impl HashFn for Highway {
        fn name(&self) -> &'static str {
            "highway"
        }
        fn hash_bytes(&self, input: &[u8]) -> u64 {
            let mut hasher = HighwayHasher::new(self.key);
            hasher.append(input);
            hasher.finalize64()
        }
    }
}

mod siphash_impl {
    use super::*;
    use siphasher::sip::SipHasher13 as Sip13;
    use siphasher::sip::SipHasher24 as Sip24;
    use std::hash::Hasher;

    pub struct SipHash13 {
        pub k0: u64,
        pub k1: u64,
    }

    impl HashFn for SipHash13 {
        fn name(&self) -> &'static str {
            "siphash13"
        }
        fn hash_bytes(&self, input: &[u8]) -> u64 {
            let mut h = Sip13::new_with_keys(self.k0, self.k1);
            h.write(input);
            h.finish()
        }
    }

    pub struct SipHash24 {
        pub k0: u64,
        pub k1: u64,
    }

    impl HashFn for SipHash24 {
        fn name(&self) -> &'static str {
            "siphash24"
        }
        fn hash_bytes(&self, input: &[u8]) -> u64 {
            let mut h = Sip24::new_with_keys(self.k0, self.k1);
            h.write(input);
            h.finish()
        }
    }
}

mod xxh3_impl {
    use super::*;

    pub struct Xxh3;

    impl HashFn for Xxh3 {
        fn name(&self) -> &'static str {
            "xxh3"
        }
        fn hash_bytes(&self, input: &[u8]) -> u64 {
            xxhash_rust::xxh3::xxh3_64(input)
        }
    }

    pub struct Xxh3Seeded {
        pub seed: u64,
    }

    impl HashFn for Xxh3Seeded {
        fn name(&self) -> &'static str {
            "xxh3_seeded"
        }
        fn hash_bytes(&self, input: &[u8]) -> u64 {
            xxhash_rust::xxh3::xxh3_64_with_seed(input, self.seed)
        }
    }
}

mod wyhash_impl {
    use super::*;

    pub struct WyHash {
        pub seed: u64,
    }

    impl HashFn for WyHash {
        fn name(&self) -> &'static str {
            "wyhash"
        }
        fn hash_bytes(&self, input: &[u8]) -> u64 {
            wyhash::wyhash(input, self.seed)
        }
    }
}

mod crc32_impl {
    use super::*;

    pub struct Crc32Fast;

    impl HashFn for Crc32Fast {
        fn name(&self) -> &'static str {
            "crc32c"
        }
        fn hash_bytes(&self, input: &[u8]) -> u64 {
            crc32fast::hash(input) as u64
        }
    }
}

mod ahash_impl {
    use super::*;
    use std::hash::{BuildHasher, Hasher};

    pub struct AHash {
        pub state: ahash::RandomState,
    }

    impl HashFn for AHash {
        fn name(&self) -> &'static str {
            "ahash"
        }
        fn hash_bytes(&self, input: &[u8]) -> u64 {
            let mut h = self.state.build_hasher();
            h.write(input);
            h.finish()
        }
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod gxhash_impl {
    use super::*;

    pub struct GxHash {
        pub seed: i64,
    }

    impl HashFn for GxHash {
        fn name(&self) -> &'static str {
            "gxhash"
        }
        fn hash_bytes(&self, input: &[u8]) -> u64 {
            gxhash::gxhash64(input, self.seed)
        }
    }
}

mod foldhash_impl {
    use super::*;
    use std::hash::{BuildHasher, Hasher};

    pub struct FoldHash {
        pub state: foldhash::fast::FixedState,
    }

    impl HashFn for FoldHash {
        fn name(&self) -> &'static str {
            "foldhash"
        }
        fn hash_bytes(&self, input: &[u8]) -> u64 {
            let mut h = self.state.build_hasher();
            h.write(input);
            h.finish()
        }
    }
}

/// Returns all hash function instances with deterministic keys derived from `seed`.
pub fn all_hashfns(seed: u64) -> Vec<Box<dyn HashFn>> {
    let mut fns: Vec<Box<dyn HashFn>> = Vec::new();

    fns.push(Box::new(blake3_impl::Blake3Hash));
    fns.push(Box::new(blake3_keyed_impl::Blake3Keyed {
        key: blake3_key_from_seed(seed),
    }));

    fns.push(Box::new(highway_impl::Highway {
        key: highway::Key(highway_key_from_seed(seed)),
    }));

    let (k0, k1) = siphash_keys_from_seed(seed);
    fns.push(Box::new(siphash_impl::SipHash13 { k0, k1 }));
    fns.push(Box::new(siphash_impl::SipHash24 { k0, k1 }));

    fns.push(Box::new(xxh3_impl::Xxh3));
    fns.push(Box::new(xxh3_impl::Xxh3Seeded { seed }));

    fns.push(Box::new(wyhash_impl::WyHash { seed }));

    fns.push(Box::new(crc32_impl::Crc32Fast));

    let (s0, s1, s2, s3) = four_seeds_from_seed(seed);
    fns.push(Box::new(ahash_impl::AHash {
        state: ahash::RandomState::with_seeds(s0, s1, s2, s3),
    }));

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fns.push(Box::new(gxhash_impl::GxHash { seed: seed as i64 }));

    fns.push(Box::new(foldhash_impl::FoldHash {
        state: foldhash::fast::FixedState::with_seed(seed),
    }));

    fns
}

/// Detect available CPU SIMD features at runtime.
pub fn detect_cpu_features() -> Vec<String> {
    let mut features = Vec::new();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            features.push("sse2".to_string());
        }
        if is_x86_feature_detected!("sse4.1") {
            features.push("sse4.1".to_string());
        }
        if is_x86_feature_detected!("sse4.2") {
            features.push("sse4.2".to_string());
        }
        if is_x86_feature_detected!("avx2") {
            features.push("avx2".to_string());
        }
        if is_x86_feature_detected!("avx512f") {
            features.push("avx512f".to_string());
        }
        if is_x86_feature_detected!("aes") {
            features.push("aes".to_string());
        }
        if is_x86_feature_detected!("pclmulqdq") {
            features.push("pclmulqdq".to_string());
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            features.push("neon".to_string());
        }
        if std::arch::is_aarch64_feature_detected!("aes") {
            features.push("aes".to_string());
        }
        if std::arch::is_aarch64_feature_detected!("crc") {
            features.push("crc".to_string());
        }
    }

    features
}
