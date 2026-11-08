//! Calibrate the quality tiers from behaviour instead of picking numbers.
//!
//! For every (candidate, task) pair we know two things:
//!   x = mean per-token KL of the candidate on the reference's own answer
//!       (teacher-forced, same tokens for everyone), and
//!   y = whether the candidate's answer actually changed.
//! A logistic regression  P(change) = σ(a + b·ln x)  fitted on those pairs
//! tells us, for *this* model, how much KL it takes before answers start to
//! change. The tier boundaries are then the KL values at which the fitted
//! probability reaches 2%, 5% and 15%, instead of fixed constants.

use crate::config::Thresholds;
use crate::tasks::TaskSection;
use serde::{Deserialize, Serialize};

/// P(change) levels that bound the near-lossless / excellent / noticeable tiers.
pub const LEVELS: [f64; 3] = [0.02, 0.05, 0.15];
/// Minimum number of changed and unchanged answers needed for a fit.
const MIN_CLASS: usize = 5;
const KL_FLOOR: f64 = 1e-6;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    pub candidate: String,
    pub task: usize,
    pub kl: f64,
    pub changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    /// What counts as "changed": the verdict (right/wrong) or the text.
    pub outcome: String,
    pub n: usize,
    pub positives: usize,
    /// logit P(change) = a + b · ln KL
    pub a: f64,
    pub b: f64,
    /// KL at which P(change) reaches each of [`LEVELS`].
    pub kl_at: [f64; 3],
    pub levels: [f64; 3],
    pub points: Vec<Point>,
}

impl Calibration {
    pub fn p_change(&self, kl: f64) -> f64 {
        sigmoid(self.a + self.b * kl.max(KL_FLOOR).ln())
    }

    pub fn apply(&self, base: &Thresholds) -> Thresholds {
        Thresholds { near_lossless: self.kl_at[0], excellent: self.kl_at[1], noticeable: self.kl_at[2], ..base.clone() }
    }
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Fit logit P(y) = a + b·x by Newton's method with a tiny ridge penalty
/// (keeps the fit finite if the classes happen to be separable).
pub fn fit_logistic(x: &[f64], y: &[bool]) -> Option<(f64, f64)> {
    const RIDGE: f64 = 1e-3;
    let (mut a, mut b) = (0.0_f64, 0.0_f64);
    for _ in 0..100 {
        let (mut ga, mut gb) = (0.0, -RIDGE * b);
        let (mut haa, mut hab, mut hbb) = (0.0, 0.0, RIDGE);
        for (&xi, &yi) in x.iter().zip(y) {
            let p = sigmoid(a + b * xi);
            let r = (yi as u8 as f64) - p;
            let w = (p * (1.0 - p)).max(1e-12);
            ga += r;
            gb += r * xi;
            haa += w;
            hab += w * xi;
            hbb += w * xi * xi;
        }
        let det = haa * hbb - hab * hab;
        if det.abs() < 1e-12 {
            return None;
        }
        let da = (hbb * ga - hab * gb) / det;
        let db = (haa * gb - hab * ga) / det;
        a += da;
        b += db;
        if da.abs() < 1e-9 && db.abs() < 1e-9 {
            break;
        }
    }
    (a.is_finite() && b.is_finite()).then_some((a, b))
}

/// KL at which the fitted probability equals `p`.
fn kl_for(a: f64, b: f64, p: f64) -> f64 {
    (((p / (1.0 - p)).ln() - a) / b).exp()
}

/// Build the calibration from task results, or explain why it is not possible.
pub fn calibrate(sec: &TaskSection, ref_name: &str) -> Result<Calibration, String> {
    let ref_runs = sec.runs.get(ref_name).ok_or("no reference task runs")?;
    let mut verdict = Vec::new();
    let mut text = Vec::new();
    for (name, runs) in &sec.runs {
        if name == ref_name {
            continue;
        }
        for (i, (run, r)) in runs.iter().zip(ref_runs).enumerate() {
            let Some(kl) = run.answer_kl else { continue };
            let base = Point { candidate: name.clone(), task: i, kl: kl.max(KL_FLOOR), changed: false };
            verdict.push(Point { changed: run.correct != r.correct, ..base.clone() });
            text.push(Point { changed: run.diverge_at.is_some(), ..base });
        }
    }

    let enough = |pts: &[Point]| {
        let pos = pts.iter().filter(|p| p.changed).count();
        pos >= MIN_CLASS && pts.len() - pos >= MIN_CLASS
    };
    // Prefer the outcome users care about (a right answer turning wrong or
    // vice versa); fall back to "the text changed" when quants are so good
    // that verdicts almost never flip.
    let (outcome, points) = if enough(&verdict) {
        ("answer verdict changed", verdict)
    } else if enough(&text) {
        ("answer text changed", text)
    } else {
        return Err(format!(
            "need at least {MIN_CLASS} changed and {MIN_CLASS} unchanged answers to calibrate; add more tasks or candidates"
        ));
    };

    let x: Vec<f64> = points.iter().map(|p| p.kl.ln()).collect();
    let y: Vec<bool> = points.iter().map(|p| p.changed).collect();
    let (a, b) = fit_logistic(&x, &y).ok_or("logistic fit did not converge")?;
    if b <= 0.0 {
        return Err("answers did not change more often at higher KL; calibration would be meaningless".into());
    }
    let kl_at = LEVELS.map(|p| kl_for(a, b, p));
    if kl_at.iter().any(|k| !k.is_finite() || *k <= 0.0) {
        return Err("calibrated thresholds are out of range".into());
    }
    Ok(Calibration {
        outcome: outcome.into(),
        n: points.len(),
        positives: y.iter().filter(|&&v| v).count(),
        a,
        b,
        kl_at,
        levels: LEVELS,
        points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_known_parameters() {
        // Deterministic "data": at each x, put round(1000·p) positives.
        let (a_true, b_true) = (2.0, 1.5);
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..40 {
            let xi = -6.0 + i as f64 * 0.2;
            let k = (sigmoid(a_true + b_true * xi) * 1000.0).round() as usize;
            for j in 0..1000 {
                x.push(xi);
                y.push(j < k);
            }
        }
        let (a, b) = fit_logistic(&x, &y).unwrap();
        assert!((a - a_true).abs() < 0.05, "a = {a}");
        assert!((b - b_true).abs() < 0.05, "b = {b}");
    }

    #[test]
    fn inverse_is_consistent() {
        let (a, b) = (3.0, 1.2);
        for p in LEVELS {
            let kl = kl_for(a, b, p);
            assert!((sigmoid(a + b * kl.ln()) - p).abs() < 1e-9);
        }
        // Higher tolerated change rate -> higher KL boundary.
        assert!(kl_for(a, b, 0.02) < kl_for(a, b, 0.05));
        assert!(kl_for(a, b, 0.05) < kl_for(a, b, 0.15));
    }
}
