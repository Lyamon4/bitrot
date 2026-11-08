//! How much of this is noise? Bootstrap confidence intervals.
//!
//! Token KLs are not independent (neighbouring tokens share context), so the
//! prompt tokens are resampled in contiguous blocks; tasks are resampled one
//! by one. Every candidate is measured on the same tokens and tasks, so the
//! comparisons are *paired*: each replicate draws one resample and evaluates
//! all candidates on it. Each replicate also re-fits the calibration and
//! re-runs the recommendation, which tells us how stable the final advice is.

use crate::calibrate::{self, Calibration};
use crate::config::Thresholds;
use crate::metrics::TokenDiv;
use crate::recommend::{self, Cand};
use crate::tasks::TaskRun;
use serde::{Deserialize, Serialize};

/// Tokens per resampling block.
pub const BLOCK: usize = 32;
const LEVEL: f64 = 0.95;

/// splitmix64: tiny, fast and good enough for resampling.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Interval {
    pub est: f64,
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    fn from(est: f64, mut samples: Vec<f64>) -> Self {
        samples.retain(|v| v.is_finite());
        if samples.is_empty() {
            return Self { est, lo: est, hi: est };
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| samples[((p * (samples.len() - 1) as f64).round() as usize).min(samples.len() - 1)];
        let tail = (1.0 - LEVEL) / 2.0;
        Self { est, lo: q(tail), hi: q(1.0 - tail) }
    }

    /// The interval excludes zero: the difference is unlikely to be noise.
    pub fn significant(&self) -> bool {
        self.lo > 0.0 || self.hi < 0.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandStats {
    pub name: String,
    pub kl_mean: Interval,
    pub top1: Interval,
    pub accuracy: Option<Interval>,
    /// Share of the reference's correct answers that this candidate broke.
    pub broken_rate: Option<Interval>,
    /// Calibrated expected share of changed answers at this KL.
    pub p_change: Option<Interval>,
}

/// `smaller` minus `larger`: positive KL difference = the smaller quant
/// drifts more.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairStats {
    pub larger: String,
    pub smaller: String,
    pub kl_diff: Interval,
    pub acc_diff: Option<Interval>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stats {
    pub replicates: usize,
    pub block_tokens: usize,
    pub level: f64,
    pub candidates: Vec<CandStats>,
    pub pairs: Vec<PairStats>,
    /// Calibrated tier boundaries with their uncertainty.
    pub kl_at: Option<[Interval; 3]>,
    /// Replicates in which the calibration could not be re-fitted.
    pub calibration_failures: usize,
    /// How often each candidate was the recommendation, best first.
    pub stability: Vec<(String, f64)>,
}

pub struct Input<'a> {
    pub names: Vec<String>,
    pub sizes: Vec<u64>,
    /// `divs[c][t]`: candidate c at token t, same token order for everyone.
    pub divs: Vec<&'a [TokenDiv]>,
    /// Number of scored tokens in each prompt document, in order.
    pub doc_lens: Vec<usize>,
    pub ref_runs: Option<&'a [TaskRun]>,
    pub cand_runs: Vec<&'a [TaskRun]>,
    pub calibration: Option<&'a Calibration>,
    /// Thresholds before calibration (the user's / default ones).
    pub base: &'a Thresholds,
    pub replicates: usize,
    pub seed: u64,
}

/// Per-block sums so a replicate costs O(blocks), not O(tokens).
struct Block {
    n: f64,
    kl: Vec<f64>,
    top1: Vec<f64>,
}

fn blocks(inp: &Input) -> Vec<Block> {
    let mut out = Vec::new();
    let mut start = 0;
    for &len in &inp.doc_lens {
        let mut s = start;
        while s < start + len {
            let e = (s + BLOCK).min(start + len);
            out.push(Block {
                n: (e - s) as f64,
                kl: inp.divs.iter().map(|d| d[s..e].iter().map(|x| x.kl).sum()).collect(),
                top1: inp.divs.iter().map(|d| d[s..e].iter().filter(|x| x.top1_agree).count() as f64).collect(),
            });
            s = e;
        }
        start += len;
    }
    out
}

struct Rep {
    kl: Vec<f64>,
    top1: Vec<f64>,
    acc: Vec<f64>,
    broken: Vec<f64>,
    p_change: Vec<f64>,
    kl_at: Option<[f64; 3]>,
    pick: Option<usize>,
}

/// One evaluation on a given resample (weights = how often each block / task
/// was drawn; all ones = the original data).
fn evaluate(inp: &Input, blocks: &[Block], block_w: &[f64], task_w: &[f64], cal_fail: &mut usize) -> Rep {
    let nc = inp.names.len();
    let n_tok: f64 = blocks.iter().zip(block_w).map(|(b, w)| b.n * w).sum::<f64>().max(1.0);
    let kl: Vec<f64> = (0..nc).map(|c| blocks.iter().zip(block_w).map(|(b, w)| b.kl[c] * w).sum::<f64>() / n_tok).collect();
    let top1: Vec<f64> = (0..nc).map(|c| blocks.iter().zip(block_w).map(|(b, w)| b.top1[c] * w).sum::<f64>() / n_tok).collect();

    let (mut acc, mut broken, mut p_change) = (Vec::new(), Vec::new(), Vec::new());
    let mut kl_at = None;
    let mut thresholds = inp.base.clone();
    let mut regression: Vec<Option<f64>> = vec![None; nc];

    if let Some(ref_runs) = inp.ref_runs {
        let n_tasks: f64 = task_w.iter().sum::<f64>().max(1.0);
        let ref_correct: f64 = ref_runs.iter().zip(task_w).filter(|(r, _)| r.correct).map(|(_, w)| w).sum();
        for (c, runs) in inp.cand_runs.iter().enumerate() {
            let correct: f64 = runs.iter().zip(task_w).filter(|(r, _)| r.correct).map(|(_, w)| w).sum();
            let reg: f64 =
                runs.iter().zip(ref_runs).zip(task_w).filter(|((r, rr), _)| rr.correct && !r.correct).map(|(_, w)| w).sum();
            acc.push(correct / n_tasks);
            let rate = if ref_correct > 0.0 { reg / ref_correct } else { 0.0 };
            broken.push(rate);
            regression[c] = Some(rate);
        }

        if let Some(cal) = inp.calibration {
            let (mut x, mut y) = (Vec::new(), Vec::new());
            for runs in &inp.cand_runs {
                for ((r, rr), &w) in runs.iter().zip(ref_runs).zip(task_w) {
                    let Some(k) = r.answer_kl else { continue };
                    let changed = if cal.uses_verdict() { r.correct != rr.correct } else { r.diverge_at.is_some() };
                    for _ in 0..w as usize {
                        x.push(k);
                        y.push(changed);
                    }
                }
            }
            let (a, b, at) = match calibrate::fit_points(&x, &y) {
                Some(fit) => fit,
                None => {
                    *cal_fail += 1;
                    (cal.a, cal.b, cal.kl_at)
                }
            };
            kl_at = Some(at);
            thresholds = Thresholds { near_lossless: at[0], excellent: at[1], noticeable: at[2], ..inp.base.clone() };
            p_change = kl.iter().map(|&k| calibrate::predict(a, b, k)).collect();
        }
    }

    let cands: Vec<Cand> = (0..nc)
        .map(|c| Cand { size: inp.sizes[c], kl_mean: kl[c], top1: top1[c], regression_rate: regression[c] })
        .collect();
    let pick = recommend::pick(&cands, &thresholds).map(|(i, _)| i);
    Rep { kl, top1, acc, broken, p_change, kl_at, pick }
}

/// Weights for drawing `n` items with replacement.
fn draw(rng: &mut Rng, n: usize) -> Vec<f64> {
    let mut w = vec![0.0; n];
    for _ in 0..n {
        w[rng.below(n)] += 1.0;
    }
    w
}

pub fn bootstrap(inp: &Input) -> Stats {
    let bl = blocks(inp);
    let n_tasks = inp.ref_runs.map(|r| r.len()).unwrap_or(0);
    let mut fails = 0usize;
    let full = evaluate(inp, &bl, &vec![1.0; bl.len()], &vec![1.0; n_tasks], &mut fails);
    fails = 0;

    let mut rng = Rng::new(inp.seed);
    let reps: Vec<Rep> = (0..inp.replicates)
        .map(|_| {
            let bw = draw(&mut rng, bl.len());
            let tw = if n_tasks > 0 { draw(&mut rng, n_tasks) } else { Vec::new() };
            evaluate(inp, &bl, &bw, &tw, &mut fails)
        })
        .collect();

    let col = |f: &dyn Fn(&Rep) -> Option<f64>| reps.iter().filter_map(f).collect::<Vec<f64>>();
    let nc = inp.names.len();
    let candidates = (0..nc)
        .map(|c| CandStats {
            name: inp.names[c].clone(),
            kl_mean: Interval::from(full.kl[c], col(&|r| Some(r.kl[c]))),
            top1: Interval::from(full.top1[c], col(&|r| Some(r.top1[c]))),
            accuracy: full.acc.get(c).map(|&e| Interval::from(e, col(&|r| r.acc.get(c).copied()))),
            broken_rate: full.broken.get(c).map(|&e| Interval::from(e, col(&|r| r.broken.get(c).copied()))),
            p_change: full.p_change.get(c).map(|&e| Interval::from(e, col(&|r| r.p_change.get(c).copied()))),
        })
        .collect();

    // Every pair, larger model first.
    let mut order: Vec<usize> = (0..nc).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(inp.sizes[i]));
    let mut pairs = Vec::new();
    for (x, &a) in order.iter().enumerate() {
        for &b in &order[x + 1..] {
            pairs.push(PairStats {
                larger: inp.names[a].clone(),
                smaller: inp.names[b].clone(),
                kl_diff: Interval::from(full.kl[b] - full.kl[a], col(&|r| Some(r.kl[b] - r.kl[a]))),
                acc_diff: (!full.acc.is_empty()).then(|| {
                    Interval::from(full.acc[b] - full.acc[a], col(&|r| Some(r.acc[b] - r.acc[a])))
                }),
            });
        }
    }

    let kl_at = full.kl_at.map(|at| {
        [0, 1, 2].map(|i| Interval::from(at[i], col(&|r| r.kl_at.map(|k| k[i]))))
    });

    let mut counts = vec![0usize; nc + 1];
    for r in &reps {
        counts[r.pick.unwrap_or(nc)] += 1;
    }
    let total = reps.len().max(1) as f64;
    let mut stability: Vec<(String, f64)> = counts
        .iter()
        .enumerate()
        .filter(|(_, &n)| n > 0)
        .map(|(i, &n)| (inp.names.get(i).cloned().unwrap_or_else(|| "none".into()), n as f64 / total))
        .collect();
    stability.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    Stats {
        replicates: inp.replicates,
        block_tokens: BLOCK,
        level: LEVEL,
        candidates,
        pairs,
        kl_at,
        calibration_failures: fails,
        stability,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn divs(kls: &[f64]) -> Vec<TokenDiv> {
        kls.iter().map(|&kl| TokenDiv { kl, top1_agree: kl < 0.5, ..Default::default() }).collect()
    }

    #[test]
    fn interval_contains_estimate_and_detects_difference() {
        // Candidate B drifts clearly more than A on every token.
        let a: Vec<f64> = (0..640).map(|i| 0.01 + (i % 7) as f64 * 0.001).collect();
        let b: Vec<f64> = a.iter().map(|x| x * 10.0).collect();
        let (da, db) = (divs(&a), divs(&b));
        let t = Thresholds::default();
        let inp = Input {
            names: vec!["A".into(), "B".into()],
            sizes: vec![200, 100],
            divs: vec![&da, &db],
            doc_lens: vec![320, 320],
            ref_runs: None,
            cand_runs: vec![],
            calibration: None,
            base: &t,
            replicates: 300,
            seed: 7,
        };
        let s = bootstrap(&inp);
        let ka = s.candidates[0].kl_mean;
        assert!(ka.lo <= ka.est && ka.est <= ka.hi);
        assert_eq!(s.pairs.len(), 1);
        assert_eq!((s.pairs[0].larger.as_str(), s.pairs[0].smaller.as_str()), ("A", "B"));
        assert!(s.pairs[0].kl_diff.significant());
        let total: f64 = s.stability.iter().map(|x| x.1).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn identical_candidates_are_not_significant() {
        let a: Vec<f64> = (0..640).map(|i| ((i * 37) % 11) as f64 * 0.003).collect();
        let da = divs(&a);
        let t = Thresholds::default();
        let inp = Input {
            names: vec!["A".into(), "B".into()],
            sizes: vec![200, 100],
            divs: vec![&da, &da],
            doc_lens: vec![640],
            ref_runs: None,
            cand_runs: vec![],
            calibration: None,
            base: &t,
            replicates: 200,
            seed: 1,
        };
        assert!(!bootstrap(&inp).pairs[0].kl_diff.significant());
    }
}
