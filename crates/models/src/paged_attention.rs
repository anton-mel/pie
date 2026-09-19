//! Our own GPU kernel: decode attention that reads the KV cache where it is.
//!
//! One threadgroup per (sequence, query head). It walks the sequence's page
//! table and reads each key and value straight from its cache slot, so
//! nothing is gathered or copied. Scores are taken in blocks of `THREADS`
//! positions with a running maximum and sum (online softmax), so memory
//! does not grow with the length of the context. Metal only; elsewhere the
//! caller uses the gather-based path.

use candle_core::{CustomOp3, Layout, MetalStorage, Result, Shape, bail};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLSize;
use std::sync::OnceLock;

/// Threads per threadgroup: positions per block, and at least `head_dim`.
const THREADS: usize = 128;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint THREADS = 128;

template <typename T>
void paged_decode(
    device const T *q, device const T *kc, device const T *vc,
    device const uint *pages, device const uint *page_start, device const uint *kv_len,
    device T *out, uint nh, uint nkv, uint hd, uint ps, float scale,
    uint b, uint h, uint tid, threadgroup float *s, threadgroup float *red)
{
    const uint kvh = h / (nh / nkv);
    const uint len = kv_len[b];
    device const uint *table = pages + page_start[b];
    device const T *qh = q + (b * nh + h) * hd;

    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (uint base = 0; base < len; base += THREADS) {
        // One score per thread: position base + tid.
        const uint i = base + tid;
        float score = -INFINITY;
        if (i < len) {
            const uint slot = table[i / ps] * ps + i % ps;
            device const T *k = kc + (slot * nkv + kvh) * hd;
            float dot = 0.0f;
            for (uint e = 0; e < hd; e++) dot += float(qh[e]) * float(k[e]);
            score = dot * scale;
        }
        s[tid] = score;
        red[tid] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = THREADS / 2; off > 0; off >>= 1) {
            if (tid < off) red[tid] = max(red[tid], red[tid + off]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float m_new = max(m, red[0]);
        const float rescale = exp(m - m_new);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Thread `tid` owns output dimension `tid`.
        const uint n = min(THREADS, len - base);
        if (tid < hd) {
            acc *= rescale;
            for (uint j = 0; j < n; j++) {
                const uint ii = base + j;
                const uint slot = table[ii / ps] * ps + ii % ps;
                acc += exp(s[j] - m_new) * float(vc[(slot * nkv + kvh) * hd + tid]);
            }
        }
        red[tid] = tid < n ? exp(s[tid] - m_new) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = THREADS / 2; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        l = l * rescale + red[0];
        m = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < hd) out[(b * nh + h) * hd + tid] = T(acc / l);
}

#define PAGED_DECODE(NAME, T)                                                           \
kernel void NAME(                                                                       \
    device const T *q [[buffer(0)]], device const T *kc [[buffer(1)]],                  \
    device const T *vc [[buffer(2)]], device const uint *pages [[buffer(3)]],           \
    device const uint *page_start [[buffer(4)]], device const uint *kv_len [[buffer(5)]],\
    device T *out [[buffer(6)]], constant uint &nh [[buffer(7)]],                       \
    constant uint &nkv [[buffer(8)]], constant uint &hd [[buffer(9)]],                  \
    constant uint &ps [[buffer(10)]], constant float &scale [[buffer(11)]],             \
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) \
{                                                                                       \
    threadgroup float s[THREADS];                                                       \
    threadgroup float red[THREADS];                                                     \
    paged_decode<T>(q, kc, vc, pages, page_start, kv_len, out, nh, nkv, hd, ps, scale,  \
                    group.y, group.x, tid, s, red);                                     \
}

PAGED_DECODE(paged_decode_f32, float)
PAGED_DECODE(paged_decode_bf16, bfloat)
"#;

/// Decode attention for `b` sequences of one new token each, over their
/// pages. Inputs: `q` `[b, nh, hd]`, and the K and V caches
/// `[slots, nkv, hd]`. Output: `[b, nh, hd]`.
pub struct PagedDecode {
    /// Every sequence's pages, one after another.
    pub pages: Vec<u32>,
    /// Where each sequence's pages start in `pages`.
    pub page_start: Vec<u32>,
    pub kv_len: Vec<u32>,
    pub kv_heads: usize,
    pub page_size: usize,
}

fn pipeline(device: &candle_core::MetalDevice, name: &'static str) -> Result<ComputePipeline> {
    static F32: OnceLock<ComputePipeline> = OnceLock::new();
    static BF16: OnceLock<ComputePipeline> = OnceLock::new();
    let cell = if name.ends_with("f32") { &F32 } else { &BF16 };
    if let Some(p) = cell.get() {
        return Ok(p.clone());
    }
    let library = device
        .device()
        .new_library_with_source(SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library.get_function(name, None).map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    Ok(cell.get_or_init(|| pipeline).clone())
}

impl CustomOp3 for PagedDecode {
    fn name(&self) -> &'static str {
        "paged-decode"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        bail!("paged-decode runs on Metal only")
    }

    fn metal_fwd(
        &self,
        q: &MetalStorage,
        lq: &Layout,
        kc: &MetalStorage,
        lk: &Layout,
        vc: &MetalStorage,
        lv: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        let (b, nh, hd) = lq.shape().dims3()?;
        for l in [lq, lk, lv] {
            if !l.is_contiguous() || l.start_offset() != 0 {
                bail!("paged-decode needs contiguous inputs");
            }
        }
        if hd > THREADS || nh % self.kv_heads != 0 {
            bail!("paged-decode supports head_dim up to {THREADS}");
        }
        let name = match q.dtype() {
            candle_core::DType::F32 => "paged_decode_f32",
            candle_core::DType::BF16 => "paged_decode_bf16",
            dtype => bail!("paged-decode does not support {dtype:?}"),
        };
        let device = q.device();
        let pipeline = pipeline(device, name)?;
        let out = device
            .new_buffer_builder()
            .with_size_for(b * nh * hd, q.dtype())
            .build()?;
        let pages = device.new_buffer_with_data(&self.pages)?;
        let page_start = device.new_buffer_with_data(&self.page_start)?;
        let kv_len = device.new_buffer_with_data(&self.kv_len)?;
        let (nh32, nkv32, hd32, ps32) = (nh as u32, self.kv_heads as u32, hd as u32, self.page_size as u32);
        let scale = 1.0f32 / (hd as f32).sqrt();

        let encoder = device.command_encoder()?;
        let e: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        e.set_compute_pipeline_state(&pipeline);
        e.set_input_buffer(0, Some(q.buffer()), 0);
        e.set_input_buffer(1, Some(kc.buffer()), 0);
        e.set_input_buffer(2, Some(vc.buffer()), 0);
        e.set_input_buffer(3, Some(&pages), 0);
        e.set_input_buffer(4, Some(&page_start), 0);
        e.set_input_buffer(5, Some(&kv_len), 0);
        e.set_output_buffer(6, Some(&out), 0);
        e.set_bytes(7, &nh32);
        e.set_bytes(8, &nkv32);
        e.set_bytes(9, &hd32);
        e.set_bytes(10, &ps32);
        e.set_bytes(11, &scale);
        let groups = MTLSize {
            width: nh,
            height: b,
            depth: 1,
        };
        let threads = MTLSize {
            width: THREADS,
            height: 1,
            depth: 1,
        };
        e.dispatch_thread_groups(groups, threads);
        drop(encoder);

        let storage = MetalStorage::new(out, device.clone(), b * nh * hd, q.dtype());
        Ok((storage, Shape::from((b, nh, hd))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    /// Plain attention for one sequence and head, computed on the CPU.
    fn reference(q: &[f32], k: &[f32], v: &[f32], op: &PagedDecode, nh: usize, hd: usize, b: usize, h: usize) -> Vec<f32> {
        let (nkv, ps) = (op.kv_heads, op.page_size);
        let kvh = h / (nh / nkv);
        let len = op.kv_len[b] as usize;
        let table = &op.pages[op.page_start[b] as usize..];
        let slot = |i: usize| table[i / ps] as usize * ps + i % ps;
        let qh = &q[(b * nh + h) * hd..][..hd];
        let scores: Vec<f32> = (0..len)
            .map(|i| {
                let kr = &k[(slot(i) * nkv + kvh) * hd..][..hd];
                qh.iter().zip(kr).map(|(a, b)| a * b).sum::<f32>() / (hd as f32).sqrt()
            })
            .collect();
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let w: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let sum: f32 = w.iter().sum();
        (0..hd)
            .map(|d| (0..len).map(|i| w[i] * v[(slot(i) * nkv + kvh) * hd + d]).sum::<f32>() / sum)
            .collect()
    }

    #[test]
    fn matches_plain_attention() -> Result<()> {
        let device = Device::new_metal(0)?;
        let (nh, nkv, hd, ps, slots) = (8, 2, 64, 16, 64 * 16);
        // Three sequences: short, one page and a bit, and many pages, with
        // their pages scattered across the cache.
        let op = PagedDecode {
            pages: vec![5, 40, 3, 17, 60, 2, 9, 33, 12, 50, 1, 7, 22, 45, 30],
            page_start: vec![0, 1, 3],
            kv_len: vec![7, 21, 190],
            kv_heads: nkv,
            page_size: ps,
        };
        let b = op.kv_len.len();
        let q = Tensor::randn(0f32, 1.0, (b, nh, hd), &Device::Cpu)?;
        let k = Tensor::randn(0f32, 1.0, (slots, nkv, hd), &Device::Cpu)?;
        let v = Tensor::randn(0f32, 1.0, (slots, nkv, hd), &Device::Cpu)?;
        let out = q
            .to_device(&device)?
            .apply_op3_no_bwd(&k.to_device(&device)?, &v.to_device(&device)?, &op)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let (q, k, v) = (q.flatten_all()?.to_vec1()?, k.flatten_all()?.to_vec1()?, v.flatten_all()?.to_vec1()?);
        let mut worst = 0f32;
        for bi in 0..b {
            for h in 0..nh {
                let expected = reference(&q, &k, &v, &op, nh, hd, bi, h);
                let got = &out[(bi * nh + h) * hd..][..hd];
                for (e, g) in expected.iter().zip(got) {
                    worst = worst.max((e - g).abs());
                }
            }
        }
        assert!(worst < 1e-4, "largest difference {worst}");
        Ok(())
    }
}
