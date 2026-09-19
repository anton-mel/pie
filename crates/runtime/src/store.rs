//! Automatic prefix sharing. Every full page of KV a working set computes
//! is recorded under a chain hash: the hash of the page's tokens together
//! with the chain hash of the page before it. Two sequences that start with
//! the same tokens get the same chain, like two paths through a trie that
//! share their start, so a new sequence can take the pages of the longest
//! recorded prefix of its tokens instead of computing them again.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

/// The chain hash of every full page of `tokens`.
pub fn chain(tokens: &[u32], page_size: usize) -> Vec<u64> {
    let mut previous = 0u64;
    tokens
        .chunks_exact(page_size)
        .map(|page| {
            let mut h = DefaultHasher::new();
            (previous, page).hash(&mut h);
            previous = h.finish();
            previous
        })
        .collect()
}

/// Recorded pages by chain hash, with when each was last used. It holds one
/// reference to every page in it.
#[derive(Default)]
pub struct Prefixes {
    pub pages: HashMap<u64, (u32, u64)>,
    pub clock: u64,
}
