//! Sampling on the GPU, in a kernel of our own: one threadgroup per row of
//! logits picks a token with the Gumbel-max trick. Each thread scores its
//! share of the vocabulary as `logit / T + g`, where `g` is Gumbel noise made
//! from a hash of (seed, row, token), and the threadgroup keeps the best.
//! candle's own GPU random numbers are not uniform enough for this: the
//! trick depends on the noise's tails, and they skewed the samples.

use candle_core::{CustomOp1, Layout, MetalStorage, Result, Shape, bail};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLSize;
use std::sync::OnceLock;

const THREADS: usize = 256;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint THREADS = 256;

// SplitMix64: a well-mixed 64-bit hash, one per (seed, row, token).
static inline ulong mix(ulong z) {
    z += 0x9e3779b97f4a7c15UL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
    return z ^ (z >> 31);
}

kernel void gumbel_argmax(
    device const float *logits [[buffer(0)]], device uint *out [[buffer(1)]],
    constant uint &vocab [[buffer(2)]], constant float &inv_t [[buffer(3)]],
    constant float &cut_log_ratio [[buffer(4)]], constant ulong &seed [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    device const float *l = logits + ulong(row) * vocab;
    threadgroup float best_score[THREADS];
    threadgroup uint best_index[THREADS];

    // The largest logit, for the min-p cut: keep l >= max + T * ln(min_p).
    float m = -INFINITY;
    for (uint i = tid; i < vocab; i += THREADS) m = max(m, l[i]);
    best_score[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = THREADS / 2; off > 0; off >>= 1) {
        if (tid < off) best_score[tid] = max(best_score[tid], best_score[tid + off]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float floor = best_score[0] + cut_log_ratio / inv_t;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float s = -INFINITY;
    uint idx = 0;
    for (uint i = tid; i < vocab; i += THREADS) {
        if (l[i] < floor) continue;
        const ulong h = mix(seed ^ mix((ulong(row) << 32) | i));
        const float u = (float(h >> 40) + 0.5f) * (1.0f / 16777216.0f);  // in (0, 1)
        const float score = l[i] * inv_t - log(-log(u));
        if (score > s) { s = score; idx = i; }
    }
    best_score[tid] = s;
    best_index[tid] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = THREADS / 2; off > 0; off >>= 1) {
        if (tid < off && best_score[tid + off] > best_score[tid]) {
            best_score[tid] = best_score[tid + off];
            best_index[tid] = best_index[tid + off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[row] = best_index[0];
}
"#;

/// Pick one token per row of f32 logits `[rows, vocab]`, at temperature
/// `temperature` (> 0), keeping only tokens at least `min_p` times as likely
/// as the most likely one.
pub struct GumbelArgmax {
    pub temperature: f32,
    pub min_p: f32,
    pub seed: u64,
}

fn pipeline(device: &candle_core::MetalDevice) -> Result<ComputePipeline> {
    static PIPELINE: OnceLock<ComputePipeline> = OnceLock::new();
    if let Some(p) = PIPELINE.get() {
        return Ok(p.clone());
    }
    let library = device
        .device()
        .new_library_with_source(SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library.get_function("gumbel_argmax", None).map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    Ok(PIPELINE.get_or_init(|| pipeline).clone())
}

impl CustomOp1 for GumbelArgmax {
    fn name(&self) -> &'static str {
        "gumbel-argmax"
    }

    fn cpu_fwd(&self, _: &candle_core::CpuStorage, _: &Layout) -> Result<(candle_core::CpuStorage, Shape)> {
        bail!("gumbel-argmax runs on Metal only")
    }

    fn metal_fwd(&self, logits: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        let (rows, vocab) = layout.shape().dims2()?;
        if !layout.is_contiguous() || logits.dtype() != candle_core::DType::F32 {
            bail!("gumbel-argmax needs contiguous f32 logits");
        }
        let device = logits.device();
        let pipeline = pipeline(device)?;
        let out = device.new_buffer_builder().with_size_for(rows, candle_core::DType::U32).build()?;
        let (vocab32, inv_t) = (vocab as u32, 1.0f32 / self.temperature);
        let cut = if self.min_p > 0.0 { self.min_p.ln() } else { f32::NEG_INFINITY };

        let encoder = device.command_encoder()?;
        let e: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        e.set_compute_pipeline_state(&pipeline);
        e.set_input_buffer(0, Some(logits.buffer()), layout.start_offset() * 4);
        e.set_output_buffer(1, Some(&out), 0);
        e.set_bytes(2, &vocab32);
        e.set_bytes(3, &inv_t);
        e.set_bytes(4, &cut);
        e.set_bytes(5, &self.seed);
        let groups = MTLSize { width: rows, height: 1, depth: 1 };
        let threads = MTLSize { width: THREADS, height: 1, depth: 1 };
        e.dispatch_thread_groups(groups, threads);
        drop(encoder);

        let storage = MetalStorage::new(out, device.clone(), rows, candle_core::DType::U32);
        Ok((storage, Shape::from(rows)))
    }
}
