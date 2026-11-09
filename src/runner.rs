//! The evaluation pipeline: reference pass, candidate passes, benchmarks,
//! and assembly of the result document.

use crate::backend::{self, Backend, BenchResult, ModelInfo};
use crate::config::{Config, ModelSpec, Settings, Thresholds};
use crate::dist::PosOut;
use crate::metrics::{self, Summary, Tier, TokenDiv};
use crate::sys::{self, Machine, PowerMeter};
use crate::calibrate::{self, Calibration};
use crate::recommend;
use crate::stats;
use crate::tasks::{self, Task, TaskRun, TaskSection};
use anyhow::{bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use crate::term::Paint;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

/// How many alternatives per token the report shows on hover.
const SHOW_TOP: usize = 5;
/// How many worst tokens per candidate the report lists.
const WORST_N: usize = 12;
/// Probe used to check that two models share a tokenizer.
const TOKENIZER_PROBE: &str = "Quantization check: The quick brown fox — Быстрая лиса 123 + fn main() { println!(\"{}\", 42); }";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Doc {
    pub name: String,
    pub category: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResult {
    pub spec: ModelSpec,
    pub info: ModelInfo,
    pub bench: Option<BenchResult>,
    pub ppl: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryStat {
    pub category: String,
    pub n_tokens: usize,
    pub kl_mean: f64,
    pub top1_agreement: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorstToken {
    pub doc: usize,
    /// Index into the document's token list of the predicted token.
    pub token: usize,
    pub kl: f64,
    pub context: String,
    pub actual: String,
    pub ref_top: Vec<(String, f32)>,
    pub cand_top: Vec<(String, f32)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateResult {
    pub model: ModelResult,
    pub summary: Summary,
    pub tier: Tier,
    pub pareto: bool,
    pub by_category: Vec<CategoryStat>,
    pub worst: Vec<WorstToken>,
}

/// Per-candidate token-level data for one document.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocCand {
    /// KL at each predicted token (index t is the prediction of token t+1).
    pub kl: Vec<f32>,
    /// 0 = same top token, 1 = different, 2 = confident flip.
    pub flip: Vec<u8>,
    pub top: Vec<Vec<(String, f32)>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocView {
    pub name: String,
    pub category: String,
    pub pieces: Vec<String>,
    pub ref_top: Vec<Vec<(String, f32)>>,
    pub cands: BTreeMap<String, DocCand>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub name: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Results {
    pub tool: String,
    pub version: String,
    pub created: String,
    pub machine: Machine,
    pub settings: Settings,
    /// Tier boundaries actually used: calibrated if `calibration` is set.
    pub thresholds: Thresholds,
    #[serde(default)]
    pub calibration: Option<Calibration>,
    /// Why calibration was skipped, if it was.
    #[serde(default)]
    pub calibration_note: Option<String>,
    pub energy_available: bool,
    pub idle_watts: Option<f64>,
    pub reference: ModelResult,
    pub candidates: Vec<CandidateResult>,
    pub docs: Vec<DocView>,
    #[serde(default)]
    pub tasks: Option<TaskSection>,
    /// Bootstrap confidence intervals and recommendation stability.
    #[serde(default)]
    pub stats: Option<stats::Stats>,
    pub recommendation: Option<Recommendation>,
}

pub fn load_docs(dir: &Path) -> Result<Vec<Doc>> {
    if !dir.is_dir() {
        bail!("prompt directory not found: {}", dir.display());
    }
    let mut docs = Vec::new();
    let pattern = dir.join("**").join("*.txt");
    for entry in glob::glob(&pattern.to_string_lossy())? {
        let path = entry?;
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        if text.trim().is_empty() {
            continue;
        }
        let category = path
            .parent()
            .filter(|p| *p != dir)
            .and_then(|p| p.file_name())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "general".into());
        let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        docs.push(Doc { name, category, text });
    }
    docs.sort_by(|a, b| (&a.category, &a.name).cmp(&(&b.category, &b.name)));
    if docs.is_empty() {
        bail!("no .txt prompts in {}", dir.display());
    }
    Ok(docs)
}

/// Split `ids` into windows of at most `ctx` tokens that overlap by one
/// token, so every token after the first is predicted exactly once.
fn windows(ids: &[u32], ctx: usize) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0;
    while start + 1 < ids.len() {
        let end = (start + ctx).min(ids.len());
        out.push(start..end);
        start = end - 1;
    }
    out
}

fn progress(len: usize, label: &str, unit: &str) -> ProgressBar {
    let pb = ProgressBar::new(len as u64);
    pb.set_style(
        ProgressStyle::with_template(&format!("  {{prefix:<14}} {{bar:28.cyan/blue}} {{pos}}/{{len}} {unit:<7}  {{msg}}"))
            .unwrap()
            .progress_chars("━╸─"),
    );
    pb.set_prefix(label.to_string());
    pb
}

fn eval_doc(
    model: &mut dyn Backend,
    ids: &[u32],
    ctx: usize,
    top_k: usize,
    gather: Option<&[Vec<u32>]>,
    pb: &ProgressBar,
) -> Result<Vec<PosOut>> {
    let mut out = Vec::with_capacity(ids.len().saturating_sub(1));
    for w in windows(ids, ctx) {
        let g = gather.map(|g| &g[w.start..w.end - 1]);
        out.extend(model.eval(&ids[w.clone()], top_k, g)?);
        pb.inc(1);
    }
    Ok(out)
}

fn show_top(model: &mut dyn Backend, pos: &[PosOut]) -> Result<Vec<Vec<(String, f32)>>> {
    let ids: Vec<u32> = pos.iter().flat_map(|p| p.top.iter().take(SHOW_TOP).map(|t| t.0)).collect();
    let mut pieces = model.pieces(&ids, false)?.into_iter();
    Ok(pos
        .iter()
        .map(|p| {
            p.top
                .iter()
                .take(SHOW_TOP)
                .map(|&(_, lp)| (pieces.next().unwrap_or_default(), lp.exp()))
                .collect()
        })
        .collect())
}

fn run_bench(model: &mut dyn Backend, prompt: &[u32], s: &Settings, meter: &mut PowerMeter, idle_w: Option<f64>) -> Result<BenchResult> {
    let mut noop = |_: bool| {};
    // Warm-up run: kernel compilation, page-in of mmapped weights.
    model.bench(&prompt[..prompt.len().min(32)], 4, &mut noop)?;

    let mut prefill = Vec::new();
    let mut decode = Vec::new();
    let mut energy_mj = Vec::new();
    let mut watts = Vec::new();
    for _ in 0..s.bench_repeats.max(1) {
        let mut started = None;
        let mut sample: Option<(f64, f64)> = None;
        let (p, d) = model.bench(prompt, s.bench_gen_tokens, &mut |start| {
            if start {
                meter.begin();
                started = Some(Instant::now());
            } else if let (Some(t0), Some(e)) = (started, meter.end()) {
                sample = Some((e.total_mj(), t0.elapsed().as_secs_f64()));
            }
        })?;
        prefill.push(p);
        decode.push(d);
        if let Some((mj, secs)) = sample {
            let idle_mj = idle_w.unwrap_or(0.0) * secs * 1000.0;
            energy_mj.push(((mj - idle_mj).max(0.0)) / s.bench_gen_tokens as f64);
            watts.push(mj / 1000.0 / secs);
        }
    }
    let median = |mut v: Vec<f64>| -> Option<f64> {
        if v.is_empty() {
            return None;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(v[v.len() / 2])
    };
    Ok(BenchResult {
        prompt_tokens: prompt.len(),
        gen_tokens: s.bench_gen_tokens,
        prefill_tok_s: median(prefill).unwrap_or(0.0),
        decode_tok_s: median(decode).unwrap_or(0.0),
        mem_bytes: model.mem_bytes(),
        energy_mj_per_tok: median(energy_mj),
        decode_watts: median(watts),
    })
}

fn bench_prompt(doc_ids: &[Vec<u32>], n: usize) -> Vec<u32> {
    let all: Vec<u32> = doc_ids.iter().flatten().copied().collect();
    all.iter().cycle().take(n.max(8)).copied().collect()
}

pub fn run(cfg: &Config) -> Result<Results> {
    let s = &cfg.settings;
    let docs = load_docs(&s.prompts)?;
    let machine = sys::machine();
    let mut meter = PowerMeter::new();

    println!(
        "{} {} prompts in {} categories · ctx {} · top-k {} · {}",
        "bitrot".bold().cyan(),
        docs.len(),
        docs.iter().map(|d| &d.category).collect::<std::collections::BTreeSet<_>>().len(),
        s.ctx,
        s.top_k,
        machine.chip.dimmed()
    );
    let idle_w = if meter.available() && !s.no_bench {
        let w = meter.idle_watts(1.5);
        if let Some(w) = w {
            println!("  idle package power {:.2} W (subtracted from energy figures)", w);
        }
        w
    } else {
        None
    };

    let task_list: Option<Vec<Task>> = if s.no_tasks {
        None
    } else if s.tasks.is_dir() {
        let t = tasks::load(&s.tasks)?;
        println!("  {} tasks with checkable answers, up to {} generated tokens each", t.len(), s.gen_tokens);
        Some(t).filter(|t| !t.is_empty())
    } else {
        println!("  {}", format!("no task directory at {} - skipping the answer check", s.tasks.display()).dimmed());
        None
    };

    // ---- reference pass -------------------------------------------------
    println!("\n{} {}  {}", "reference".bold(), cfg.reference.name.bold().green(), cfg.reference.path.dimmed());
    let mut reference = backend::open(&cfg.reference, s)?;
    let ref_probe = reference.tokenize(TOKENIZER_PROBE, false)?;
    let doc_ids: Vec<Vec<u32>> = docs
        .iter()
        .map(|d| {
            reference.tokenize(&d.text, true).map(|mut ids| {
                ids.truncate(s.max_tokens_per_doc);
                ids
            })
        })
        .collect::<Result<_>>()?;
    let n_windows: usize = doc_ids.iter().map(|ids| windows(ids, s.ctx).len()).sum();

    let pb = progress(n_windows, &cfg.reference.name, "windows");
    let t0 = Instant::now();
    let mut ref_out = Vec::with_capacity(docs.len());
    for ids in &doc_ids {
        ref_out.push(eval_doc(reference.as_mut(), ids, s.ctx, s.top_k, None, &pb)?);
    }
    pb.finish_with_message(format!("{:.1}s", t0.elapsed().as_secs_f64()));

    let mut views = Vec::with_capacity(docs.len());
    for ((doc, ids), out) in docs.iter().zip(&doc_ids).zip(&ref_out) {
        views.push(DocView {
            name: doc.name.clone(),
            category: doc.category.clone(),
            pieces: reference.pieces(ids, true)?,
            ref_top: show_top(reference.as_mut(), out)?,
            cands: BTreeMap::new(),
        });
    }
    let ref_nll: Vec<f64> = ref_out.iter().flatten().map(|p| -(p.target as f64)).collect();
    let ref_ppl = (ref_nll.iter().sum::<f64>() / ref_nll.len().max(1) as f64).exp();
    let bench_ids = bench_prompt(&doc_ids, s.bench_prompt_tokens);
    let ref_bench = if s.no_bench { None } else { Some(run_bench(reference.as_mut(), &bench_ids, s, &mut meter, idle_w)?) };
    let ref_result = ModelResult { spec: cfg.reference.clone(), info: reference.info().clone(), bench: ref_bench, ppl: ref_ppl };

    // Prompts are formatted once, with the reference's chat template, and the
    // same ids go to every model.
    let mut task_section = None;
    let mut task_prompts: Vec<Vec<u32>> = Vec::new();
    let mut ref_gen: Vec<Vec<u32>> = Vec::new();
    let mut ref_answers: Vec<Vec<PosOut>> = Vec::new();
    if let Some(list) = &task_list {
        task_prompts = list.iter().map(|t| reference.chat_prompt(&t.prompt)).collect::<Result<_>>()?;
        let (gen, runs) = run_tasks(reference.as_mut(), list, &task_prompts, s.gen_tokens, &cfg.reference.name, None)?;
        let correct = runs.iter().filter(|r| r.correct).count();
        println!("  tasks {}/{} correct", correct, runs.len());
        ref_gen = gen;
        ref_answers = score_answers(reference.as_mut(), &task_prompts, &ref_gen, s.top_k, None)?;
        let mut sec = TaskSection { items: list.clone(), gen_tokens: s.gen_tokens, ..Default::default() };
        sec.summary.insert(cfg.reference.name.clone(), tasks::summarize(&runs, &runs));
        sec.runs.insert(cfg.reference.name.clone(), runs);
        task_section = Some(sec);
    }
    drop(reference);

    // Reference top-k ids per position: what each candidate must score.
    let gathers: Vec<Vec<Vec<u32>>> = ref_out
        .iter()
        .map(|doc| doc.iter().map(|p| p.top.iter().map(|t| t.0).collect()).collect())
        .collect();

    // ---- candidates -----------------------------------------------------
    let mut cands = Vec::new();
    let mut cand_divs: Vec<Vec<TokenDiv>> = Vec::new();
    for spec in &cfg.candidates {
        println!("\n{} {}  {}", "candidate".bold(), spec.name.bold().yellow(), spec.path.dimmed());
        let mut model = backend::open(spec, s)?;
        if model.tokenize(TOKENIZER_PROBE, false)? != ref_probe {
            bail!(
                "'{}' tokenizes text differently from the reference '{}' - KL is only meaningful for quants of the same model",
                spec.name,
                cfg.reference.name
            );
        }
        let pb = progress(n_windows, &spec.name, "windows");
        let t0 = Instant::now();
        let mut all_divs: Vec<TokenDiv> = Vec::new();
        let mut cat_divs: BTreeMap<String, Vec<TokenDiv>> = BTreeMap::new();
        let mut worst: Vec<(f64, usize, usize, Vec<(String, f32)>)> = Vec::new();

        for (d, ids) in doc_ids.iter().enumerate() {
            let out = eval_doc(model.as_mut(), ids, s.ctx, s.top_k, Some(&gathers[d]), &pb)?;
            let divs: Vec<TokenDiv> =
                ref_out[d].iter().zip(&out).map(|(r, c)| metrics::token_divergence(r, c)).collect();
            let tops = show_top(model.as_mut(), &out)?;
            for (t, div) in divs.iter().enumerate() {
                worst.push((div.kl, d, t, tops[t].clone()));
            }
            worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            worst.truncate(WORST_N);

            views[d].cands.insert(
                spec.name.clone(),
                DocCand {
                    kl: divs.iter().map(|x| x.kl as f32).collect(),
                    flip: divs.iter().map(|x| if x.confident_flip { 2 } else if x.top1_agree { 0 } else { 1 }).collect(),
                    top: tops,
                },
            );
            cat_divs.entry(docs[d].category.clone()).or_default().extend(divs.iter().copied());
            all_divs.extend(divs);
        }
        pb.finish_with_message(format!("{:.1}s", t0.elapsed().as_secs_f64()));

        let bench = if s.no_bench { None } else { Some(run_bench(model.as_mut(), &bench_ids, s, &mut meter, idle_w)?) };
        let summary = metrics::summarize(&all_divs);
        cand_divs.push(all_divs);
        let mut task_line = String::new();
        if let (Some(sec), Some(list)) = (task_section.as_mut(), &task_list) {
            let ref_runs = sec.runs[&cfg.reference.name].clone();
            let (_, mut runs) = run_tasks(model.as_mut(), list, &task_prompts, s.gen_tokens, &spec.name, Some((&ref_gen, &ref_runs)))?;
            // Score the reference's own answers with this model: the KL the
            // calibration relates to whether the answer changed.
            let ans = score_answers(model.as_mut(), &task_prompts, &ref_gen, s.top_k, Some(&ref_answers))?;
            for ((run, r), c) in runs.iter_mut().zip(&ref_answers).zip(&ans) {
                if !r.is_empty() {
                    let divs: Vec<TokenDiv> = r.iter().zip(c).map(|(r, c)| metrics::token_divergence(r, c)).collect();
                    run.answer_kl = Some(metrics::summarize(&divs).kl_mean);
                }
            }
            let ts = tasks::summarize(&runs, &ref_runs);
            task_line = format!("  ·  tasks {}/{} ({} broken, {} identical)", ts.correct, ts.n, ts.regressions, ts.identical);
            sec.summary.insert(spec.name.clone(), ts);
            sec.runs.insert(spec.name.clone(), runs);
        }
        let by_category = cat_divs
            .into_iter()
            .map(|(category, divs)| {
                let sm = metrics::summarize(&divs);
                CategoryStat { category, n_tokens: sm.n_tokens, kl_mean: sm.kl_mean, top1_agreement: sm.top1_agreement }
            })
            .collect();
        let worst = worst
            .into_iter()
            .map(|(kl, d, t, cand_top)| {
                let v = &views[d];
                let from = (t + 1).saturating_sub(16);
                WorstToken {
                    doc: d,
                    token: t + 1,
                    kl,
                    context: v.pieces[from..=t].concat(),
                    actual: v.pieces[t + 1].clone(),
                    ref_top: v.ref_top[t].clone(),
                    cand_top,
                }
            })
            .collect();

        println!(
            "  KL mean {}  ·  top-1 {:.1}%  ·  ΔPPL {:+.2}%{}",
            format!("{:.4}", summary.kl_mean).bold(),
            summary.top1_agreement * 100.0,
            summary.ppl_delta_pct,
            task_line
        );
        cands.push(CandidateResult {
            model: ModelResult { spec: spec.clone(), info: model.info().clone(), bench, ppl: summary.ppl_cand },
            tier: Tier::from_kl(summary.kl_mean, &cfg.thresholds),
            summary,
            pareto: false,
            by_category,
            worst,
        });
    }

    // ---- calibration ----------------------------------------------------
    let mut thresholds = cfg.thresholds.clone();
    let mut calibration = None;
    let mut calibration_note = None;
    if let Some(sec) = &task_section {
        match calibrate::calibrate(sec, &cfg.reference.name) {
            Ok(cal) => {
                thresholds = cal.apply(&cfg.thresholds);
                for c in cands.iter_mut() {
                    c.tier = Tier::from_kl(c.summary.kl_mean, &thresholds);
                }
                calibration = Some(cal);
            }
            Err(why) => calibration_note = Some(why),
        }
    }

    mark_pareto(&mut cands);
    let recommendation = recommend(&cands, &thresholds, task_section.as_ref(), &cfg.reference.name);

    // ---- uncertainty ----------------------------------------------------
    let stats = (s.bootstrap > 0).then(|| {
        let t0 = Instant::now();
        let ref_runs = task_section.as_ref().map(|sec| sec.runs[&cfg.reference.name].as_slice());
        let input = stats::Input {
            names: cands.iter().map(|c| c.model.spec.name.clone()).collect(),
            sizes: cands.iter().map(|c| c.model.info.size_bytes).collect(),
            divs: cand_divs.iter().map(|d| d.as_slice()).collect(),
            doc_lens: ref_out.iter().map(|d| d.len()).collect(),
            ref_runs,
            cand_runs: match &task_section {
                Some(sec) => cands.iter().map(|c| sec.runs[&c.model.spec.name].as_slice()).collect(),
                None => Vec::new(),
            },
            calibration: calibration.as_ref(),
            base: &cfg.thresholds,
            replicates: s.bootstrap,
            seed: 0x6b6c_6472_6966_7421,
        };
        let st = stats::bootstrap(&input);
        println!("
  {} bootstrap resamples in {:.1}s", st.replicates, t0.elapsed().as_secs_f64());
        st
    });

    Ok(Results {
        tool: "bitrot".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        created: chrono::Local::now().format("%Y-%m-%d %H:%M").to_string(),
        machine,
        settings: s.clone(),
        thresholds,
        calibration,
        calibration_note,
        energy_available: meter.available(),
        idle_watts: idle_w,
        reference: ref_result,
        candidates: cands,
        docs: views,
        tasks: task_section,
        stats,
        recommendation,
    })
}

/// A candidate is Pareto-optimal if no other one is both smaller and closer
/// to the reference.
fn mark_pareto(cands: &mut [CandidateResult]) {
    let pts: Vec<(u64, f64)> = cands.iter().map(|c| (c.model.info.size_bytes, c.summary.kl_mean)).collect();
    for (i, c) in cands.iter_mut().enumerate() {
        let (s, k) = pts[i];
        c.pareto = !pts
            .iter()
            .enumerate()
            .any(|(j, &(s2, k2))| j != i && s2 <= s && k2 <= k && (s2 < s || k2 < k));
    }
}

/// Share of the reference's correct answers this candidate broke.
fn regression_rate(sec: Option<&TaskSection>, ref_name: &str, name: &str) -> Option<f64> {
    let sec = sec?;
    let ref_correct = sec.summary.get(ref_name)?.correct;
    let s = sec.summary.get(name)?;
    Some(if ref_correct == 0 { 0.0 } else { s.regressions as f64 / ref_correct as f64 })
}

fn rec_inputs(cands: &[CandidateResult], sec: Option<&TaskSection>, ref_name: &str) -> Vec<recommend::Cand> {
    cands
        .iter()
        .map(|c| recommend::Cand {
            size: c.model.info.size_bytes,
            kl_mean: c.summary.kl_mean,
            top1: c.summary.top1_agreement,
            regression_rate: regression_rate(sec, ref_name, &c.model.spec.name),
        })
        .collect()
}

fn recommend(cands: &[CandidateResult], t: &Thresholds, sec: Option<&TaskSection>, ref_name: &str) -> Option<Recommendation> {
    let task_note = if sec.is_some() {
        format!(", breaking at most {:.0}% of the answers the original gets right", t.max_regression * 100.0)
    } else {
        String::new()
    };
    let (i, why) = recommend::pick(&rec_inputs(cands, sec, ref_name), t)?;
    let reason = match why {
        recommend::Why::Excellent => format!(
            "smallest model with mean KL < {:.4} and ≥ {:.0}% top-1 agreement on your prompts{}",
            t.excellent,
            t.min_top1 * 100.0,
            task_note
        ),
        recommend::Why::Noticeable => {
            format!("no candidate is excellent; this is the smallest with mean KL < {:.4}{}", t.noticeable, task_note)
        }
        recommend::Why::Closest => "every candidate degrades noticeably; this one is closest to the reference".into(),
    };
    Some(Recommendation { name: cands[i].model.spec.name.clone(), reason })
}

/// Teacher-force `prompt ++ answer` for every task and return the outputs at
/// the answer positions only. With `reference` (the reference's own outputs
/// from this function) the model also scores the reference's top-k ids.
fn score_answers(
    model: &mut dyn Backend,
    prompts: &[Vec<u32>],
    answers: &[Vec<u32>],
    top_k: usize,
    reference: Option<&[Vec<PosOut>]>,
) -> Result<Vec<Vec<PosOut>>> {
    let mut all = Vec::with_capacity(prompts.len());
    for (i, (prompt, answer)) in prompts.iter().zip(answers).enumerate() {
        if answer.is_empty() || prompt.is_empty() {
            all.push(Vec::new());
            continue;
        }
        let seq: Vec<u32> = prompt.iter().chain(answer).copied().collect();
        // Position t predicts token t+1; the answer starts at prompt.len().
        let first = prompt.len() - 1;
        let gather = reference.map(|r| {
            let mut g = vec![Vec::new(); seq.len() - 1];
            for (j, p) in r[i].iter().enumerate() {
                g[first + j] = p.top.iter().map(|t| t.0).collect();
            }
            g
        });
        let out = model.eval(&seq, top_k, gather.as_deref())?;
        all.push(out.into_iter().skip(first).collect());
    }
    Ok(all)
}

/// Greedy-generate an answer for every task and grade it. With `against`
/// (the reference's token outputs and runs) also record where each output
/// starts to differ from the reference's.
fn run_tasks(
    model: &mut dyn Backend,
    list: &[Task],
    prompts: &[Vec<u32>],
    max_new: usize,
    label: &str,
    against: Option<(&[Vec<u32>], &[TaskRun])>,
) -> Result<(Vec<Vec<u32>>, Vec<TaskRun>)> {
    let pb = progress(list.len(), label, "tasks");
    let t0 = Instant::now();
    let mut gens = Vec::with_capacity(list.len());
    let mut runs = Vec::with_capacity(list.len());
    for (i, (task, prompt)) in list.iter().zip(prompts).enumerate() {
        let ids = model.generate(prompt, max_new)?;
        let text = model.pieces(&ids, true)?.concat();
        let (extracted, correct) = tasks::grade(task, &text);
        let (diverge_at, same_prefix_chars) = match against {
            Some((ref_ids, ref_runs)) => {
                (tasks::diverge_at(&ref_ids[i], &ids), tasks::common_prefix_chars(&ref_runs[i].text, &text))
            }
            None => (None, text.chars().count()),
        };
        runs.push(TaskRun { n_tokens: ids.len(), text, extracted, correct, diverge_at, same_prefix_chars, answer_kl: None });
        gens.push(ids);
        pb.inc(1);
    }
    pb.finish_with_message(format!("{:.1}s", t0.elapsed().as_secs_f64()));
    Ok((gens, runs))
}

#[cfg(test)]
mod tests {
    use super::windows;

    #[test]
    fn windows_cover_every_prediction_once() {
        let ids: Vec<u32> = (0..1000).collect();
        let ws = windows(&ids, 256);
        let predicted: usize = ws.iter().map(|w| w.len() - 1).sum();
        assert_eq!(predicted, ids.len() - 1);
        for pair in ws.windows(2) {
            assert_eq!(pair[0].end - 1, pair[1].start);
        }
        assert!(ws.iter().all(|w| w.len() <= 256));
    }

    #[test]
    fn short_inputs() {
        assert!(windows(&[1], 8).is_empty());
        assert_eq!(windows(&[1, 2], 8), vec![0..2]);
    }
}
