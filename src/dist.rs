//! Per-position output of a model under teacher forcing, and the math that
//! turns a full logit row into it.

use serde::{Deserialize, Serialize};

/// What a backend reports for one position `t` of a token window, i.e. the
/// model's distribution over the token at `t + 1`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PosOut {
    /// The model's own top-k tokens as `(token id, log-prob)`, sorted by
    /// descending probability.
    pub top: Vec<(u32, f32)>,
    /// Log-probs of the token ids the caller asked for (usually the
    /// reference model's top-k at this position), in the same order.
    pub gathered: Vec<f32>,
    /// Log-prob of the token that actually comes next in the text.
    pub target: f32,
}

/// Numerically stable log-softmax normaliser: returns `logsumexp(logits)`.
pub fn log_sum_exp(logits: &[f32]) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    if !max.is_finite() {
        return max;
    }
    let sum: f64 = logits.iter().map(|&x| ((x as f64) - max).exp()).sum();
    max + sum.ln()
}

/// Turn one row of raw logits into a [`PosOut`].
///
/// * `top_k` - how many of the model's own most likely tokens to keep.
/// * `gather` - extra token ids whose log-probs we need (may be empty).
/// * `target` - the id of the token that really follows.
pub fn summarize_logits(logits: &[f32], top_k: usize, gather: &[u32], target: u32) -> PosOut {
    let lse = log_sum_exp(logits);
    let lp = |id: u32| -> f32 {
        logits
            .get(id as usize)
            .map(|&x| ((x as f64) - lse) as f32)
            .unwrap_or(f32::NEG_INFINITY)
    };

    let k = top_k.min(logits.len()).max(1);
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    let by_logit_desc = |a: &u32, b: &u32| {
        logits[*b as usize]
            .partial_cmp(&logits[*a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    if k < idx.len() {
        idx.select_nth_unstable_by(k - 1, by_logit_desc);
        idx.truncate(k);
    }
    idx.sort_unstable_by(by_logit_desc);

    PosOut {
        top: idx.into_iter().map(|id| (id, lp(id))).collect(),
        gathered: gather.iter().map(|&id| lp(id)).collect(),
        target: lp(target),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let logits = [1.0_f32, 2.0, 3.0, -1.0, 0.5];
        let lse = log_sum_exp(&logits);
        let total: f64 = logits.iter().map(|&x| ((x as f64) - lse).exp()).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn top_k_is_sorted_and_correct() {
        let logits = [0.1_f32, 5.0, -2.0, 3.0, 4.0];
        let out = summarize_logits(&logits, 3, &[2, 0], 3);
        let ids: Vec<u32> = out.top.iter().map(|t| t.0).collect();
        assert_eq!(ids, vec![1, 4, 3]);
        assert!(out.top[0].1 > out.top[1].1 && out.top[1].1 > out.top[2].1);
        assert_eq!(out.gathered.len(), 2);
        assert!((out.target - out.top[2].1).abs() < 1e-6);
    }
}
