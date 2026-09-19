//! The model itself: a Qwen2/Qwen3 transformer that turns tokens into
//! next-token scores (logits).

use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder, rotary_emb::rope};
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

/// One sequence's share of a batched forward. `tokens` are the last
/// `tokens.len()` entries of a `kv_len`-long sequence stored in `pages`.
pub struct Seq {
    pub tokens: Vec<u32>,
    pub positions: Vec<u32>,
    pub pages: Vec<u32>,
    pub kv_len: usize,
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

    /// Returns `[seqs.len(), vocab]` f32 logits for each sequence's last token.
    pub fn forward(&mut self, seqs: &[Seq]) -> Result<Tensor> {
        let (nh, nkv, hd, ps) = (self.heads, self.kv_heads, self.head_dim, self.page_size);
        let tokens: Vec<u32> = seqs.iter().flat_map(|s| s.tokens.iter().copied()).collect();
        let positions: Vec<u32> = seqs.iter().flat_map(|s| s.positions.iter().copied()).collect();
        let n = tokens.len();
        let pos = Tensor::new(positions, &self.device)?;
        let (cos, sin) = (self.cos.index_select(&pos, 0)?, self.sin.index_select(&pos, 0)?);

        let mut offsets = Vec::new();
        let mut slots = Vec::new();
        let mut off = 0;
        for s in seqs {
            let slot = |i: usize| s.pages[i / ps] * ps as u32 + (i % ps) as u32;
            slots.push(Tensor::new((0..s.kv_len).map(slot).collect::<Vec<_>>(), &self.device)?);
            offsets.push(off);
            off += s.tokens.len();
        }

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

            let mut outs = Vec::new();
            for (i, s) in seqs.iter().enumerate() {
                let (o, len) = (offsets[i], s.tokens.len());
                // Write the new tokens' K/V into their slots, one page run at a time.
                let mut j = 0;
                while j < len {
                    let p = s.kv_len - len + j;
                    let run = (ps - p % ps).min(len - j);
                    let at = s.pages[p / ps] as usize * ps + p % ps;
                    l.k_cache.slice_set(&k.narrow(0, o + j, run)?.contiguous()?, 0, at)?;
                    l.v_cache.slice_set(&v.narrow(0, o + j, run)?.contiguous()?, 0, at)?;
                    j += run;
                }
                outs.push(attend(
                    &q.narrow(0, o, len)?,
                    &l.k_cache,
                    &l.v_cache,
                    &slots[i],
                    s.kv_len,
                    nkv,
                )?);
            }
            let attn = l.o.forward(&Tensor::cat(&outs, 0)?)?;
            x = (x + attn)?;
            let h = l.ln2.forward(&x)?;
            let mlp = l
                .down
                .forward(&(candle_nn::ops::silu(&l.gate.forward(&h)?)? * l.up.forward(&h)?)?)?;
            x = (x + mlp)?;
        }
        let last: Vec<u32> = seqs
            .iter()
            .zip(&offsets)
            .map(|(s, o)| (o + s.tokens.len() - 1) as u32)
            .collect();
        let x = self
            .norm
            .forward(&x.index_select(&Tensor::new(last, &self.device)?, 0)?)?;
        Ok(self.lm_head.forward(&x)?.to_dtype(DType::F32)?)
    }
}

/// Causal attention of `q` ([len, nh, hd], the sequence's last `len` tokens)
/// over the `kv_len` cache slots in `slots`.
fn attend(q: &Tensor, kc: &Tensor, vc: &Tensor, slots: &Tensor, kv_len: usize, nkv: usize) -> Result<Tensor> {
    let (len, nh, hd) = q.dims3()?;
    ensure!(len <= kv_len, "more new tokens than sequence length");
    let group = nh / nkv;
    // [nkv, group, len, hd] against [nkv, kv_len, hd] handles GQA without copying K/V.
    let q = q
        .reshape((len, nkv, group, hd))?
        .permute((1, 2, 0, 3))?
        .reshape((nkv, group * len, hd))?;
    let k = kc.index_select(slots, 0)?.transpose(0, 1)?.contiguous()?;
    let v = vc.index_select(slots, 0)?.transpose(0, 1)?.contiguous()?;
    let scores = (q.contiguous()?.matmul(&k.t()?)? / (hd as f64).sqrt())?.to_dtype(DType::F32)?;
    let past = kv_len - len;
    let mask: Vec<f32> = (0..group * len)
        .flat_map(|r| (0..kv_len).map(move |c| if c <= past + r % len { 0.0 } else { f32::NEG_INFINITY }))
        .collect();
    let mask = Tensor::from_vec(mask, (group * len, kv_len), q.device())?;
    let p = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&mask)?)?.to_dtype(v.dtype())?;
    let out = p.matmul(&v)?.reshape((nkv, group, len, hd))?.permute((2, 0, 1, 3))?;
    Ok(out.reshape((len, nh * hd))?)
}
