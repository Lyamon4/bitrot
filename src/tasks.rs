//! Behavioural check: do the quants still *answer* the same?
//!
//! KL tells you how far the next-token distributions moved. Tasks tell you
//! whether that movement changes what the model actually says: every model
//! generates a short answer greedily from the same prompt, answers are graded
//! against a known solution, and each quant's output is compared with the
//! reference's (does it break answers the original got right, and how many
//! tokens in does its text start to differ).

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Check {
    /// The start of the answer contains the expected text (case-insensitive).
    #[default]
    Contains,
    /// The last number in the answer equals the expected number (models
    /// often restate the question first, so the final number is the answer,
    /// as in GSM8K-style grading).
    Number,
    /// The first line of the answer equals the expected text.
    Exact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    #[serde(default)]
    pub category: String,
    pub prompt: String,
    pub answer: String,
    #[serde(default)]
    pub check: Check,
    /// Other spellings that also count as correct.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub accept: Vec<String>,
}

/// One model's attempt at one task.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskRun {
    pub text: String,
    pub extracted: Option<String>,
    pub correct: bool,
    pub n_tokens: usize,
    /// Token index where the output first differs from the reference's
    /// (None = identical, and always None for the reference itself).
    pub diverge_at: Option<usize>,
    /// Length in characters of the text shared with the reference's output.
    pub same_prefix_chars: usize,
    /// Mean per-token KL of this model on the *reference's* answer
    /// (teacher-forced). None for the reference and for empty answers.
    #[serde(default)]
    pub answer_kl: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskSummary {
    pub n: usize,
    pub correct: usize,
    pub accuracy: f64,
    /// Tasks the reference got right and this model got wrong.
    pub regressions: usize,
    /// Tasks the reference got wrong and this model got right (luck).
    pub fixes: usize,
    /// Outputs token-for-token identical to the reference's.
    pub identical: usize,
    /// Median token index of the first difference, over non-identical outputs.
    pub median_diverge: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskSection {
    pub items: Vec<Task>,
    pub gen_tokens: usize,
    pub runs: BTreeMap<String, Vec<TaskRun>>,
    pub summary: BTreeMap<String, TaskSummary>,
}

/// Load every `*.jsonl` file in `dir`; one task per line. The category
/// defaults to the file name.
pub fn load(dir: &Path) -> Result<Vec<Task>> {
    if !dir.is_dir() {
        bail!("task directory not found: {}", dir.display());
    }
    let mut tasks = Vec::new();
    let pattern = dir.join("*.jsonl");
    let mut files: Vec<_> = glob::glob(&pattern.to_string_lossy())?.collect::<std::result::Result<_, _>>()?;
    files.sort();
    for path in files {
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("//") {
                continue;
            }
            let mut t: Task = serde_json::from_str(line)
                .with_context(|| format!("{}:{}: invalid task", path.display(), i + 1))?;
            if t.category.is_empty() {
                t.category = stem.clone();
            }
            tasks.push(t);
        }
    }
    Ok(tasks)
}

fn normalize(s: &str) -> String {
    s.to_lowercase().replace('ё', "е").split_whitespace().collect::<Vec<_>>().join(" ")
}

/// First number in `s`, e.g. "The answer is 1,250.5 km" -> 1250.5.
#[cfg(test)]
pub fn first_number(s: &str) -> Option<f64> {
    numbers(s).into_iter().next()
}

/// Last number in `s`.
pub fn last_number(s: &str) -> Option<f64> {
    numbers(s).pop()
}

/// All numbers in `s`, in order. Handles signs, decimals and thousands
/// separators ("1,250.5"), but not "3,4" lists.
pub fn numbers(s: &str) -> Vec<f64> {
    let chars: Vec<char> = s.chars().collect();
    let mut found = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let neg = (c == '-' || c == '−') && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit());
        if c.is_ascii_digit() || neg {
            let mut num = String::new();
            if neg {
                num.push('-');
                i += 1;
            }
            while i < chars.len() {
                let ch = chars[i];
                let next_digit = chars.get(i + 1).is_some_and(|d| d.is_ascii_digit());
                if ch.is_ascii_digit() {
                    num.push(ch);
                } else if ch == '.' && next_digit && !num.contains('.') {
                    num.push('.');
                } else if ch == ',' && next_digit && chars.get(i + 4).is_none_or(|d| !d.is_ascii_digit()) {
                    // thousands separator: "1,250" (but not "3,4" lists)
                    let group: String = chars[i + 1..(i + 4).min(chars.len())].iter().collect();
                    if group.len() == 3 && group.chars().all(|d| d.is_ascii_digit()) {
                        // skip the comma
                    } else {
                        break;
                    }
                } else {
                    break;
                }
                i += 1;
            }
            if let Ok(v) = num.parse() {
                found.push(v);
            }
            continue;
        }
        i += 1;
    }
    found
}

/// Grade one output. Returns (what was extracted, whether it is correct).
pub fn grade(task: &Task, output: &str) -> (Option<String>, bool) {
    let head: String = output.trim().chars().take(160).collect();
    match task.check {
        Check::Number => {
            let want: Option<f64> = task.answer.replace(',', "").parse().ok();
            match (last_number(output), want) {
                (Some(got), Some(want)) => {
                    let ok = (got - want).abs() <= 1e-6 * want.abs().max(1.0);
                    (Some(trim_num(got)), ok)
                }
                (got, _) => (got.map(trim_num), false),
            }
        }
        Check::Contains => {
            let h = normalize(&head);
            let hit = std::iter::once(&task.answer)
                .chain(task.accept.iter())
                .find(|a| h.contains(&normalize(a)));
            (Some(head.lines().next().unwrap_or("").trim().to_string()), hit.is_some())
        }
        Check::Exact => {
            let first = head.lines().next().unwrap_or("").trim().trim_end_matches('.').to_string();
            let ok = std::iter::once(&task.answer).chain(task.accept.iter()).any(|a| normalize(a) == normalize(&first));
            (Some(first), ok)
        }
    }
}

fn trim_num(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

/// Where two token sequences first differ (None if identical).
pub fn diverge_at(reference: &[u32], cand: &[u32]) -> Option<usize> {
    match reference.iter().zip(cand).position(|(a, b)| a != b) {
        Some(i) => Some(i),
        None if reference.len() == cand.len() => None,
        None => Some(reference.len().min(cand.len())),
    }
}

pub fn common_prefix_chars(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

pub fn summarize(runs: &[TaskRun], reference: &[TaskRun]) -> TaskSummary {
    let n = runs.len();
    let correct = runs.iter().filter(|r| r.correct).count();
    let pairs = || runs.iter().zip(reference);
    let mut div: Vec<f64> = runs.iter().filter_map(|r| r.diverge_at.map(|d| d as f64)).collect();
    div.sort_by(|a, b| a.partial_cmp(b).unwrap());
    TaskSummary {
        n,
        correct,
        accuracy: if n > 0 { correct as f64 / n as f64 } else { 0.0 },
        regressions: pairs().filter(|(c, r)| r.correct && !c.correct).count(),
        fixes: pairs().filter(|(c, r)| !r.correct && c.correct).count(),
        identical: runs.iter().filter(|r| r.diverge_at.is_none()).count(),
        median_diverge: if div.is_empty() { None } else { Some(div[div.len() / 2]) },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(check: Check, answer: &str) -> Task {
        Task { id: "t".into(), category: "c".into(), prompt: "p".into(), answer: answer.into(), check, accept: vec![] }
    }

    #[test]
    fn extracts_numbers() {
        assert_eq!(first_number("It is 48."), Some(48.0));
        assert_eq!(first_number("x = -3 and 5"), Some(-3.0));
        assert_eq!(first_number("about 1,250.5 km"), Some(1250.5));
        assert_eq!(first_number("3,4 and 5"), Some(3.0));
        assert_eq!(first_number("no digits"), None);
        assert_eq!(last_number("1,250,000 and 7 + 5 = 12."), Some(12.0));
        assert_eq!(numbers("1,250,000 and 3,4"), vec![1_250_000.0, 3.0, 4.0]);
    }

    #[test]
    fn grading() {
        assert!(grade(&task(Check::Number, "48"), " 48").1);
        assert!(grade(&task(Check::Number, "12"), "To find 7 + 5 we add them: 7 + 5 = 12.").1);
        assert!(!grade(&task(Check::Number, "48"), "47").1);
        assert!(grade(&task(Check::Contains, "Париж"), "Столица Франции — париж.").1);
        assert!(grade(&task(Check::Contains, "ёлка"), "Это елка").1);
        assert!(grade(&task(Check::Exact, "def"), "def.\nmore text").1);
        assert!(!grade(&task(Check::Exact, "def"), "define").1);
    }

    #[test]
    fn divergence() {
        assert_eq!(diverge_at(&[1, 2, 3], &[1, 2, 3]), None);
        assert_eq!(diverge_at(&[1, 2, 3], &[1, 9, 3]), Some(1));
        assert_eq!(diverge_at(&[1, 2, 3], &[1, 2]), Some(2));
    }

    #[test]
    fn regressions_and_fixes() {
        let r = |c: bool, d: Option<usize>| TaskRun { correct: c, diverge_at: d, ..Default::default() };
        let reference = vec![r(true, None), r(true, None), r(false, None)];
        let cand = vec![r(true, None), r(false, Some(3)), r(true, Some(0))];
        let s = summarize(&cand, &reference);
        assert_eq!((s.regressions, s.fixes, s.identical, s.correct), (1, 1, 1, 2));
        assert_eq!(s.median_diverge, Some(3.0));
    }
}
