//! What goes into each model step.
//!
//! Every step runs at most `step_tokens` tokens. Requests with the fewest
//! tokens left go first: most are one inferlet waiting for its next token.
//! A long prefill fills what is left of the budget and is split across
//! steps, and its inferlet gets the result once the last piece has run. So a
//! long prompt slows everyone a little instead of stopping them.

use crate::engine::{Distribution, Request, top_k};
use ::engine::{Engine, Seq};
use std::collections::HashSet;
use tokio::sync::mpsc;

/// A request, and how far through it the model is.
struct Job {
    request: Request,
    /// Arrival order.
    id: u64,
    /// Tokens already run.
    done: usize,
    rows: Vec<Distribution>,
    error: Option<String>,
}

impl Job {
    fn left(&self) -> usize {
        self.request.seq.tokens.len() - self.done
    }

    /// The next `n` of its tokens, as a sequence of their own: the same
    /// pages, a shorter `kv_len`, and only the outputs that fall inside.
    /// Pages its remaining tokens will write.
    fn writes(&self, page_size: usize) -> impl Iterator<Item = u32> + '_ {
        let s = &self.request.seq;
        let first = (s.kv_len - self.left()) / page_size;
        let last = (s.kv_len - 1) / page_size;
        s.pages[first..=last].iter().copied()
    }

    fn chunk(&self, n: usize) -> Seq {
        let s = &self.request.seq;
        let (from, to) = (self.done, self.done + n);
        Seq {
            copies: if from == 0 { s.copies.clone() } else { vec![] },
            tokens: s.tokens[from..to].to_vec(),
            positions: s.positions[from..to].to_vec(),
            outputs: s
                .outputs
                .iter()
                .filter(|&&o| (from..to).contains(&(o as usize)))
                .map(|&o| o - from as u32)
                .collect(),
            pages: s.pages.clone(),
            kv_len: s.kv_len - (s.tokens.len() - to),
        }
    }
}

/// Runs its steps on an `Engine`.
/// Orders jobs by pipeline instead of by the pages they share.
pub fn run(mut model: Box<dyn Engine>, mut rx: mpsc::UnboundedReceiver<Vec<Request>>, step_tokens: usize) {
    let ps = model.page_size();
    let mut jobs: Vec<Job> = vec![];
    let mut next_id = 0;
    let mut add = |jobs: &mut Vec<Job>, batch: Vec<Request>| {
        for request in batch {
            let job = Job {
                request,
                id: next_id,
                done: 0,
                rows: vec![],
                error: None,
            };
            jobs.push(job);
            next_id += 1;
        }
    };

    loop {
        // Block for work only when nothing is left over from last step.
        if jobs.is_empty() {
            match rx.blocking_recv() {
                Some(batch) => add(&mut jobs, batch),
                None => return,
            }
        }
        while let Ok(batch) = rx.try_recv() {
            add(&mut jobs, batch);
        }

        // Fewest tokens left first, within the budget. A job runs only after
        // every earlier job on its pipeline: finished, or running in full in
        // this same step, where arrival order puts it first.
        jobs.sort_by_key(|j| (j.left(), j.id));
        let mut budget = step_tokens;
        let mut picked = vec![];
        let mut whole = HashSet::new();
        for (i, job) in jobs.iter().enumerate() {
            if budget == 0 {
                break;
            }
            let pipeline = job.request.pipeline;
            let mut waiting = jobs
                .iter()
                .any(|e| e.request.pipeline == pipeline && e.id < job.id && !whole.contains(&e.id));
            // A step copies pages before it writes any, so a job that copies
            // a page waits while an earlier job still writes that page.
            if job.done == 0 && !job.request.seq.copies.is_empty() {
                let sources: HashSet<u32> = job.request.seq.copies.iter().map(|&(from, _)| from).collect();
                waiting |= jobs
                    .iter()
                    .any(|e| e.id < job.id && e.writes(ps).any(|p| sources.contains(&p)));
            }
            if !waiting {
                let n = job.left().min(budget);
                if n == job.left() {
                    whole.insert(job.id);
                }
                picked.push((i, n));
                budget -= n;
            }
        }
        // Run in arrival order, so writes land before later reads.
        picked.sort_by_key(|&(i, _)| jobs[i].id);

        let seqs: Vec<Seq> = picked.iter().map(|&(i, n)| jobs[i].chunk(n)).collect();
        match model.forward(&seqs) {
            Ok(logits) => {
                let mut rows = logits.into_iter();
                for (&(i, n), seq) in picked.iter().zip(&seqs) {
                    let job = &mut jobs[i];
                    let (k, allowed) = (job.request.top_k, job.request.allowed.as_deref());
                    let new: Vec<_> = rows
                        .by_ref()
                        .take(seq.outputs.len())
                        .map(|r| top_k(r, k, allowed))
                        .collect();
                    job.rows.extend(new);
                    job.done += n;
                }
            }
            Err(e) => {
                for &(i, _) in &picked {
                    jobs[i].error = Some(e.to_string());
                    jobs[i].done = jobs[i].request.seq.tokens.len();
                }
            }
        }

        // Answer finished jobs. Dropping them releases the pages they held.
        let (finished, left): (Vec<Job>, Vec<Job>) = jobs.drain(..).partition(|j| j.left() == 0);
        jobs = left;
        for job in finished {
            let result = match job.error {
                Some(e) => Err(e),
                None => {
                    if let Some(on_done) = job.request.on_done {
                        on_done();
                    }
                    Ok(job.rows)
                }
            };
            let _ = job.request.reply.send(result);
        }
    }
}
