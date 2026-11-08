//! Divergence metrics between a reference model and a quantized candidate.
//!
//! The reference keeps its top-k tokens per position. The candidate reports
//! its log-probs for exactly those k tokens. Everything outside the top-k is
//! folded into one "tail" bucket on both sides, so we compare two proper
//! (k+1)-bucket distributions. By the data-processing inequality this
//! coarsened KL is a *lower bound* on the full-vocabulary KL, and with k = 64
//! the reference tail is typically < 1% of the mass, so the bound is tight.

use crate::dist::PosOut;
use serde::{Deserialize, Serialize};

const EPS: f64 = 1e-10;

/// Divergence of the candidate from the reference at one position.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TokenDiv {
    /// KL(P_ref || Q_cand) in nats, over top-k + tail buckets.
    pub kl: f64,
    /// Both models put the same token first.
    pub top1_agree: bool,
    /// They disagree on the top token although the reference was confident
    /// about it (p > 0.5) - these are the flips a user actually notices.
    pub confident_flip: bool,
    /// Negative log-likelihood of the real next token, reference and candidate.
    pub nll_ref: f64,
    pub nll_cand: f64,
}

pub fn token_divergence(reference: &PosOut, cand: &PosOut) -> TokenDiv {
    let mut kl = 0.0_f64;
    let mut p_mass = 0.0_f64;
    let mut q_mass = 0.0_f64;
    for (i, &(_, lp_ref)) in reference.top.iter().enumerate() {
        let lp_ref = lp_ref as f64;
        let lq = cand.gathered.get(i).copied().unwrap_or(f32::NEG_INFINITY) as f64;
        let p = lp_ref.exp();
        let q = lq.exp();
        p_mass += p;
        q_mass += q;
        if p > 0.0 {
            kl += p * (lp_ref - lq.max(EPS.ln()));
        }
    }
    let p_tail = (1.0 - p_mass).max(0.0);
    let q_tail = (1.0 - q_mass).max(EPS);
    if p_tail > EPS {
        kl += p_tail * (p_tail.ln() - q_tail.ln());
    }

    let ref_top = reference.top.first().copied();
    let cand_top = cand.top.first().map(|t| t.0);
    let top1_agree = ref_top.map(|t| t.0) == cand_top;
    let confident_flip = !top1_agree && ref_top.map(|t| (t.1 as f64).exp() > 0.5).unwrap_or(false);

    TokenDiv {
        kl: kl.max(0.0),
        top1_agree,
        confident_flip,
        nll_ref: -(reference.target as f64),
        nll_cand: -(cand.target as f64),
    }
}

/// Aggregate statistics over many positions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Summary {
    pub n_tokens: usize,
    pub kl_mean: f64,
    pub kl_median: f64,
    pub kl_p90: f64,
    pub kl_p99: f64,
    pub kl_max: f64,
    pub top1_agreement: f64,
    pub confident_flips: usize,
    pub ppl_ref: f64,
    pub ppl_cand: f64,
    /// Relative perplexity change in percent, (ppl_cand / ppl_ref - 1) * 100.
    pub ppl_delta_pct: f64,
}

pub fn summarize(divs: &[TokenDiv]) -> Summary {
    let n = divs.len();
    if n == 0 {
        return Summary::default();
    }
    let mut kls: Vec<f64> = divs.iter().map(|d| d.kl).collect();
    kls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pct = |q: f64| -> f64 {
        let i = ((q * (n as f64 - 1.0)).round() as usize).min(n - 1);
        kls[i]
    };
    let mean = |f: &dyn Fn(&TokenDiv) -> f64| divs.iter().map(f).sum::<f64>() / n as f64;
    let ppl_ref = mean(&|d| d.nll_ref).exp();
    let ppl_cand = mean(&|d| d.nll_cand).exp();
    Summary {
        n_tokens: n,
        kl_mean: kls.iter().sum::<f64>() / n as f64,
        kl_median: pct(0.5),
        kl_p90: pct(0.9),
        kl_p99: pct(0.99),
        kl_max: kls[n - 1],
        top1_agreement: divs.iter().filter(|d| d.top1_agree).count() as f64 / n as f64,
        confident_flips: divs.iter().filter(|d| d.confident_flip).count(),
        ppl_ref,
        ppl_cand,
        ppl_delta_pct: (ppl_cand / ppl_ref - 1.0) * 100.0,
    }
}

/// Human-facing quality tier for a mean KL value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    NearLossless,
    Excellent,
    Noticeable,
    Degraded,
}

impl Tier {
    pub fn from_kl(kl_mean: f64, t: &crate::config::Thresholds) -> Self {
        if kl_mean < t.near_lossless {
            Tier::NearLossless
        } else if kl_mean < t.excellent {
            Tier::Excellent
        } else if kl_mean < t.noticeable {
            Tier::Noticeable
        } else {
            Tier::Degraded
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Tier::NearLossless => "near-lossless",
            Tier::Excellent => "excellent",
            Tier::Noticeable => "noticeable",
            Tier::Degraded => "degraded",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pos(top: &[(u32, f64)], gathered: &[f64]) -> PosOut {
        PosOut {
            top: top.iter().map(|&(i, p)| (i, p.ln() as f32)).collect(),
            gathered: gathered.iter().map(|&p| p.ln() as f32).collect(),
            target: 0.5_f32.ln(),
        }
    }

    #[test]
    fn identical_distributions_have_zero_kl() {
        let r = pos(&[(1, 0.6), (2, 0.3)], &[]);
        let c = pos(&[(1, 0.6), (2, 0.3)], &[0.6, 0.3]);
        let d = token_divergence(&r, &c);
        assert!(d.kl.abs() < 1e-6, "kl = {}", d.kl);
        assert!(d.top1_agree);
        assert!(!d.confident_flip);
    }

    #[test]
    fn matches_closed_form_kl() {
        // P = (0.7, 0.2, tail 0.1), Q = (0.5, 0.3, tail 0.2)
        let r = pos(&[(1, 0.7), (2, 0.2)], &[]);
        let c = pos(&[(1, 0.5), (2, 0.3)], &[0.5, 0.3]);
        let expected = 0.7 * (0.7_f64 / 0.5).ln() + 0.2 * (0.2_f64 / 0.3).ln() + 0.1 * (0.1_f64 / 0.2).ln();
        let d = token_divergence(&r, &c);
        assert!((d.kl - expected).abs() < 1e-6, "kl = {} expected {}", d.kl, expected);
    }

    #[test]
    fn detects_confident_flip() {
        let r = pos(&[(1, 0.9), (2, 0.05)], &[]);
        let c = pos(&[(2, 0.6), (1, 0.3)], &[0.3, 0.6]);
        let d = token_divergence(&r, &c);
        assert!(!d.top1_agree);
        assert!(d.confident_flip);
        assert!(d.kl > 0.5);
    }

    #[test]
    fn summary_percentiles() {
        let divs: Vec<TokenDiv> = (0..100)
            .map(|i| TokenDiv { kl: i as f64 / 100.0, top1_agree: i % 2 == 0, ..Default::default() })
            .collect();
        let s = summarize(&divs);
        assert_eq!(s.n_tokens, 100);
        assert!((s.kl_mean - 0.495).abs() < 1e-9);
        assert!((s.top1_agreement - 0.5).abs() < 1e-9);
        assert!((s.kl_max - 0.99).abs() < 1e-9);
    }
}
