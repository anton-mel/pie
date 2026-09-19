//! The model itself: a Qwen2/Qwen3 transformer that turns tokens into
//! next-token scores (logits).

use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder, rotary_emb::rope};
use engine::Seq;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: Option<usize>,
    rms_norm_eps: f64,
    rope_theta: f64,
    vocab_size: usize,
    #[serde(default)]
    tie_word_embeddings: bool,
}

struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    gate: Linear,
    up: Linear,
    down: Linear,
    ln1: RmsNorm,
    ln2: RmsNorm,
    // Paged cache: [pages * page_size, kv_heads, head_dim].
    k_cache: Tensor,
    v_cache: Tensor,
}

pub struct Model {
    embed: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    cos: Tensor,
    sin: Tensor,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    pub page_size: usize,
    pub device: Device,
}

fn linear(i: usize, o: usize, vb: VarBuilder) -> Result<Linear> {
    // Qwen2 has q/k/v biases, Qwen3 does not: take them if the checkpoint does.
    let bias = vb.contains_tensor("bias");
    Ok(candle_nn::linear_b(i, o, bias, vb)?)
}

impl Model {
    pub fn load(cfg: &Config, vb: VarBuilder, pages: usize, page_size: usize) -> Result<Self> {
        let (h, hd) = (
            cfg.hidden_size,
            cfg.head_dim.unwrap_or(cfg.hidden_size / cfg.num_attention_heads),
        );
        let (nh, nkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
        let (dtype, device) = (vb.dtype(), vb.device().clone());
        let m = vb.pp("model");
        let norm = |n: usize, vb: VarBuilder| candle_nn::rms_norm(n, cfg.rms_norm_eps, vb);
        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            let l = m.pp(format!("layers.{i}"));
            let (a, mlp) = (l.pp("self_attn"), l.pp("mlp"));
            let cache = || Tensor::zeros((pages * page_size, nkv, hd), dtype, &device);
            layers.push(Layer {
                q: linear(h, nh * hd, a.pp("q_proj"))?,
                k: linear(h, nkv * hd, a.pp("k_proj"))?,
                v: linear(h, nkv * hd, a.pp("v_proj"))?,
                o: linear(nh * hd, h, a.pp("o_proj"))?,
                q_norm: a
                    .contains_tensor("q_norm.weight")
                    .then(|| norm(hd, a.pp("q_norm")))
                    .transpose()?,
                k_norm: a
                    .contains_tensor("k_norm.weight")
                    .then(|| norm(hd, a.pp("k_norm")))
                    .transpose()?,
                gate: linear(h, cfg.intermediate_size, mlp.pp("gate_proj"))?,
                up: linear(h, cfg.intermediate_size, mlp.pp("up_proj"))?,
                down: linear(cfg.intermediate_size, h, mlp.pp("down_proj"))?,
                ln1: norm(h, l.pp("input_layernorm"))?,
                ln2: norm(h, l.pp("post_attention_layernorm"))?,
                k_cache: cache()?,
                v_cache: cache()?,
            });
        }
        let embed = candle_nn::embedding(cfg.vocab_size, h, m.pp("embed_tokens"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed.embeddings().clone(), None)
        } else {
            candle_nn::linear_no_bias(h, cfg.vocab_size, vb.pp("lm_head"))?
        };
        // RoPE tables, indexed by position at runtime.
        let max_pos = 32768;
        let inv: Vec<f32> = (0..hd / 2)
            .map(|i| 1.0 / cfg.rope_theta.powf(2.0 * i as f64 / hd as f64) as f32)
            .collect();
        let t = Tensor::arange(0u32, max_pos as u32, &device)?.to_dtype(DType::F32)?;
        let freqs = t.unsqueeze(1)?.matmul(&Tensor::new(inv, &device)?.unsqueeze(0)?)?;
        Ok(Self {
            norm: norm(h, m.pp("norm"))?,
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
            embed,
            layers,
            lm_head,
            heads: nh,
            kv_heads: nkv,
            head_dim: hd,
            page_size,
            device,
        })
    }

    /// Returns f32 logits, one row per entry of each sequence's `outputs`.
    /// Attention now follows a plan made once per step (`Plan`).
    pub fn forward(&mut self, seqs: &[Seq]) -> Result<Tensor> {
        let (nh, nkv, hd, ps) = (self.heads, self.kv_heads, self.head_dim, self.page_size);
        let tokens: Vec<u32> = seqs.iter().flat_map(|s| s.tokens.iter().copied()).collect();
        let positions: Vec<u32> = seqs.iter().flat_map(|s| s.positions.iter().copied()).collect();
        let n = tokens.len();
        let pos = Tensor::new(positions, &self.device)?;
        let (cos, sin) = (self.cos.index_select(&pos, 0)?, self.sin.index_select(&pos, 0)?);

        let mut offsets = Vec::new();
        let mut off = 0;
        for s in seqs {
            offsets.push(off);
            off += s.tokens.len();
        }
        let plan = Plan::new(seqs, &offsets, (nkv, hd, nh / nkv), ps, &self.device)?;

        let mut x = self.embed.forward(&Tensor::new(tokens, &self.device)?)?;
        for l in &mut self.layers {
            let h = l.ln1.forward(&x)?;
            let heads = |t: Tensor, k: usize, norm: &Option<RmsNorm>| -> Result<Tensor> {
                let t = t.reshape((n, k, hd))?;
                let t = match norm {
                    Some(nm) => nm.forward(&t)?,
                    None => t,
                };
                Ok(rope(&t.transpose(0, 1)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?
                    .squeeze(0)?
                    .transpose(0, 1)?)
            };
            let q = heads(l.q.forward(&h)?, nh, &l.q_norm)?; // [n, nh, hd]
            let k = heads(l.k.forward(&h)?, nkv, &l.k_norm)?;
            let v = l.v.forward(&h)?.reshape((n, nkv, hd))?;

            // Copy-on-write pages first, then every new token's K/V into its
            // slot: one gather and one scatter per cache for the whole batch.
            if let Some((from, to)) = &plan.copy {
                for cache in [&l.k_cache, &l.v_cache] {
                    cache.scatter_set(to, &cache.index_select(from, 0)?, 0)?;
                }
            }
            l.k_cache.scatter_set(&plan.write, &k.contiguous()?, 0)?;
            l.v_cache.scatter_set(&plan.write, &v.contiguous()?, 0)?;

            let mut outs = Vec::new();
            if let Some(d) = &plan.decode {
                outs.push(attend_decode(&q, &l.k_cache, &l.v_cache, d, nkv)?);
            }
            for p in &plan.prefill {
                outs.push(attend(&q.narrow(0, p.offset, p.len)?, &l.k_cache, &l.v_cache, p, nkv)?);
            }
            let attn = l.o.forward(&Tensor::cat(&outs, 0)?.index_select(&plan.order, 0)?)?;
            x = (x + attn)?;
            let h = l.ln2.forward(&x)?;
            let mlp = l
                .down
                .forward(&(candle_nn::ops::silu(&l.gate.forward(&h)?)? * l.up.forward(&h)?)?)?;
            x = (x + mlp)?;
        }

        // Only the requested rows go through the final norm and the head.
        let rows: Vec<u32> = seqs
            .iter()
            .zip(&offsets)
            .flat_map(|(s, &o)| s.outputs.iter().map(move |&i| o as u32 + i))
            .collect();

        if rows.is_empty() {
            return Ok(Tensor::zeros((0, 1), DType::F32, &self.device)?);
        }

        let x = self
            .norm
            .forward(&x.index_select(&Tensor::new(rows, &self.device)?, 0)?)?;
        Ok(self.lm_head.forward(&x)?.to_dtype(DType::F32)?)
    }
}

/// The batch's attention, worked out once per step and used by every layer.
struct Plan {
    /// Cache row of every new token, repeated to `[n, kv_heads, head_dim]`:
    /// the index that writes all new K/V in one `scatter_set`.
    write: Tensor,
    /// Copy-on-write for the whole batch: the cache rows of every page to
    /// copy, and where they go (repeated to `[rows, kv_heads, head_dim]`).
    copy: Option<(Tensor, Tensor)>,
    /// Sequences with one new token, attended together.
    decode: Option<Decode>,
    /// Sequences with more (prefill, verify), attended one at a time.
    prefill: Vec<Prefill>,
    /// Puts the attention outputs, decodes first, back in token order.
    order: Tensor,
}

struct Decode {
    /// Their rows of `q`.
    rows: Tensor,
    /// Their cache slots, each padded with slot 0 to the longest: `[b * len]`.
    slots: Tensor,
    /// 0 for real slots, -inf for padding: `[b, 1, 1, len]`.
    mask: Tensor,
    b: usize,
    len: usize,
}

struct Prefill {
    offset: usize,
    len: usize,
    slots: Tensor,
    /// Causal mask of the sequence's new tokens: `[group * len, kv_len]`.
    mask: Tensor,
}

impl Plan {
    fn new(
        seqs: &[Seq],
        offsets: &[usize],
        (nkv, hd, group): (usize, usize, usize),
        ps: usize,
        device: &Device,
    ) -> Result<Self> {
        let slot = |s: &Seq, i: usize| s.pages[i / ps] * ps as u32 + (i % ps) as u32;
        let n: usize = seqs.iter().map(|s| s.tokens.len()).sum();

        let write: Vec<u32> = seqs
            .iter()
            .flat_map(|s| (s.kv_len - s.tokens.len()..s.kv_len).map(move |i| slot(s, i)))
            .collect();
        let write = Tensor::from_vec(write, (n, 1, 1), device)?
            .broadcast_as((n, nkv, hd))?
            .contiguous()?;

        let rows = |page: u32| (0..ps as u32).map(move |i| page * ps as u32 + i);
        let pairs: Vec<&(u32, u32)> = seqs.iter().flat_map(|s| &s.copies).collect();
        let copy = if pairs.is_empty() {
            None
        } else {
            let from: Vec<u32> = pairs.iter().flat_map(|&&(f, _)| rows(f)).collect();
            let to: Vec<u32> = pairs.iter().flat_map(|&&(_, t)| rows(t)).collect();
            let m = to.len();
            let to = Tensor::from_vec(to, (m, 1, 1), device)?
                .broadcast_as((m, nkv, hd))?
                .contiguous()?;
            Some((Tensor::new(from, device)?, to))
        };

        let (singles, longer): (Vec<usize>, Vec<usize>) = (0..seqs.len()).partition(|&i| seqs[i].tokens.len() == 1);
        let mut order_src: Vec<usize> = vec![];
        let decode = if singles.is_empty() {
            None
        } else {
            let (b, len) = (singles.len(), singles.iter().map(|&i| seqs[i].kv_len).max().unwrap());
            let (mut slots, mut mask) = (vec![0u32; b * len], vec![f32::NEG_INFINITY; b * len]);
            for (j, &i) in singles.iter().enumerate() {
                for c in 0..seqs[i].kv_len {
                    slots[j * len + c] = slot(&seqs[i], c);
                    mask[j * len + c] = 0.0;
                }
                order_src.push(offsets[i]);
            }
            let rows: Vec<u32> = singles.iter().map(|&i| offsets[i] as u32).collect();
            Some(Decode {
                rows: Tensor::new(rows, device)?,
                slots: Tensor::new(slots, device)?,
                mask: Tensor::from_vec(mask, (b, 1, 1, len), device)?,
                b,
                len,
            })
        };
        let mut prefill = vec![];
        for &i in &longer {
            let s = &seqs[i];
            let (len, kv_len) = (s.tokens.len(), s.kv_len);
            ensure!(len <= kv_len, "more new tokens than sequence length");
            let past = kv_len - len;
            let mask: Vec<f32> = (0..group * len)
                .flat_map(|r| (0..kv_len).map(move |c| if c <= past + r % len { 0.0 } else { f32::NEG_INFINITY }))
                .collect();
            prefill.push(Prefill {
                offset: offsets[i],
                len,
                slots: Tensor::new((0..kv_len).map(|c| slot(s, c)).collect::<Vec<_>>(), device)?,
                mask: Tensor::from_vec(mask, (group * len, kv_len), device)?,
            });
            order_src.extend(offsets[i]..offsets[i] + len);
        }
        // `order_src[k]` is the token at output row `k`; invert it.
        let mut order = vec![0u32; n];
        for (k, &t) in order_src.iter().enumerate() {
            order[t] = k as u32;
        }
        Ok(Self {
            write,
            copy,
            decode,
            prefill,
            order: Tensor::new(order, device)?,
        })
    }
}

/// Attention of all single-token sequences at once: one gather of their
/// cache slots, one batched matmul, one softmax.
fn attend_decode(q: &Tensor, kc: &Tensor, vc: &Tensor, d: &Decode, nkv: usize) -> Result<Tensor> {
    let (_, nh, hd) = q.dims3()?;
    let (b, len, group) = (d.b, d.len, nh / nkv);
    let q = q.contiguous()?.index_select(&d.rows, 0)?.reshape((b, nkv, group, hd))?;
    let gather = |c: &Tensor| -> Result<Tensor> {
        Ok(c.index_select(&d.slots, 0)?
            .reshape((b, len, nkv, hd))?
            .permute((0, 2, 1, 3))?
            .contiguous()?)
    };
    let (k, v) = (gather(kc)?, gather(vc)?);
    let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?)? / (hd as f64).sqrt())?.to_dtype(DType::F32)?;
    let p = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&d.mask)?)?.to_dtype(v.dtype())?;
    Ok(p.matmul(&v)?.reshape((b, nh * hd))?)
}

/// Causal attention of one sequence's new tokens `q` (`[len, nh, hd]`) over
/// its cache slots. Its slots and mask now come from the step's plan.
fn attend(q: &Tensor, kc: &Tensor, vc: &Tensor, p: &Prefill, nkv: usize) -> Result<Tensor> {
    let (len, nh, hd) = q.dims3()?;
    let group = nh / nkv;
    // [nkv, group, len, hd] against [nkv, kv_len, hd] handles GQA without copying K/V.
    let q = q
        .reshape((len, nkv, group, hd))?
        .permute((1, 2, 0, 3))?
        .reshape((nkv, group * len, hd))?;
    let k = kc.index_select(&p.slots, 0)?.transpose(0, 1)?.contiguous()?;
    let v = vc.index_select(&p.slots, 0)?.transpose(0, 1)?.contiguous()?;
    let scores = (q.contiguous()?.matmul(&k.t()?)? / (hd as f64).sqrt())?.to_dtype(DType::F32)?;
    let p = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&p.mask)?)?.to_dtype(v.dtype())?;
    let out = p.matmul(&v)?.reshape((nkv, group, len, hd))?.permute((2, 0, 1, 3))?;
    Ok(out.reshape((len, nh * hd))?)
}

/// NEW
/// The model as an engine: the runtime reaches it only through this.
impl engine::Engine for Model {
    fn page_size(&self) -> usize {
        self.page_size
    }

    fn forward(&mut self, seqs: &[Seq]) -> Result<Vec<Vec<f32>>> {
        Ok(Model::forward(self, seqs)?.to_vec2::<f32>()?)
    }
}
