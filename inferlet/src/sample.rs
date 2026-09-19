//! Sampling, written in the inferlet: the engine only returns the top-k of
//! the next-token distribution, and everything else happens here.

use crate::Distribution;

/// Temperature, top-p and min-p sampling over a top-k distribution.
///
/// The distribution is already cut to its top k tokens, so these act within
/// those k. Ask `forward` for a k large enough that the cut does not matter
/// (64 is plenty at usual temperatures).
pub struct Sampler {
    /// 0 is greedy; below 1 sharpens the distribution, above 1 flattens it.
    pub temperature: f32,
    /// Keep the fewest most likely tokens whose probability adds up to this.
    pub top_p: f32,
    /// Drop tokens less likely than this fraction of the most likely one.
    pub min_p: f32,
    rng: u64,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32) -> Self {
        let mut seed = [0u8; 8];
        getrandom::fill(&mut seed).expect("no randomness");
        Self {
            temperature,
            top_p,
            min_p: 0.0,
            rng: u64::from_le_bytes(seed),
        }
    }

    /// Use a fixed seed, so the same run gives the same tokens.
    pub fn seed(mut self, seed: u64) -> Self {
        self.rng = seed;
        self
    }

    pub fn sample(&mut self, d: &Distribution) -> u32 {
        if self.temperature <= 0.0 {
            return d.ids[0];
        }
        // softmax(logits / T) is proportional to p^(1/T).
        let mut w: Vec<f32> = d.probs.iter().map(|p| p.powf(1.0 / self.temperature)).collect();
        let total: f32 = w.iter().sum();
        w.iter_mut().for_each(|x| *x /= total);

        // `probs` are sorted, most likely first, so both cuts keep a prefix.
        let mut keep = 0;
        let mut mass = 0.0;
        while keep < w.len() && (keep == 0 || (mass < self.top_p && w[keep] >= self.min_p * w[0])) {
            mass += w[keep];
            keep += 1;
        }

        let mut u = self.next_f32() * mass;
        for i in 0..keep {
            u -= w[i];
            if u <= 0.0 {
                return d.ids[i];
            }
        }
        d.ids[keep - 1]
    }

    /// Uniform in [0, 1), from splitmix64.
    fn next_f32(&mut self) -> f32 {
        self.rng = self.rng.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
    }
}
