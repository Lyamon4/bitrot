//! Which quant to recommend. A pure function of a few numbers per candidate,
//! so the bootstrap can re-run the exact same decision on resampled data.

use crate::config::Thresholds;

#[derive(Debug, Clone)]
pub struct Cand {
    pub size: u64,
    pub kl_mean: f64,
    pub top1: f64,
    /// Share of the reference's correct answers this candidate broke
    /// (None when no tasks were run).
    pub regression_rate: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Why {
    /// Smallest candidate in the excellent tier that keeps the answers.
    Excellent,
    /// Nothing is excellent; smallest that is at most noticeable.
    Noticeable,
    /// Everything degrades; the one closest to the reference.
    Closest,
}

/// Index of the recommended candidate and the rule that picked it.
pub fn pick(cands: &[Cand], t: &Thresholds) -> Option<(usize, Why)> {
    let keeps = |c: &Cand| c.regression_rate.is_none_or(|r| r <= t.max_regression);
    let smallest = |pred: &dyn Fn(&Cand) -> bool| {
        cands.iter().enumerate().filter(|(_, c)| pred(c)).min_by_key(|(_, c)| c.size).map(|(i, _)| i)
    };
    if let Some(i) = smallest(&|c| c.kl_mean < t.excellent && c.top1 >= t.min_top1 && keeps(c)) {
        return Some((i, Why::Excellent));
    }
    if let Some(i) = smallest(&|c| c.kl_mean < t.noticeable && keeps(c)) {
        return Some((i, Why::Noticeable));
    }
    cands
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.kl_mean.partial_cmp(&b.1.kl_mean).unwrap())
        .map(|(i, _)| (i, Why::Closest))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(size: u64, kl: f64, top1: f64, reg: Option<f64>) -> Cand {
        Cand { size, kl_mean: kl, top1, regression_rate: reg }
    }

    #[test]
    fn prefers_smallest_excellent() {
        let t = Thresholds::default();
        let cands = [c(100, 0.004, 0.97, None), c(60, 0.03, 0.93, None), c(40, 0.2, 0.7, None)];
        assert_eq!(pick(&cands, &t), Some((1, Why::Excellent)));
    }

    #[test]
    fn broken_answers_disqualify() {
        let t = Thresholds::default();
        let cands = [c(100, 0.004, 0.97, Some(0.0)), c(60, 0.03, 0.93, Some(0.3))];
        assert_eq!(pick(&cands, &t), Some((0, Why::Excellent)));
    }

    #[test]
    fn falls_back_to_closest() {
        let t = Thresholds::default();
        let cands = [c(100, 0.5, 0.6, None), c(60, 0.9, 0.5, None)];
        assert_eq!(pick(&cands, &t), Some((0, Why::Closest)));
    }
}
