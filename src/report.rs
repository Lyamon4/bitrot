//! Terminal summary and the self-contained HTML report.

use crate::metrics::Tier;
use crate::runner::Results;
use anyhow::Result;
use owo_colors::OwoColorize;
use std::path::Path;

const TEMPLATE: &str = include_str!("../assets/report.html");
const HERO_JPG: &[u8] = include_bytes!("../assets/hero.jpg");
const DATA_MARKER: &str = "/*__KLDRIFT_DATA__*/null";
const HERO_MARKER: &str = "__KLDRIFT_HERO__";

fn base64(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(A[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

pub fn write_html(results: &Results, path: &Path) -> Result<()> {
    // Keep `</script>` and friends inside strings from closing the tag.
    let json = serde_json::to_string(results)?.replace("</", "<\\/").replace("<!--", "<\\!--");
    let hero = format!("data:image/jpeg;base64,{}", base64(HERO_JPG));
    let html = TEMPLATE.replacen(DATA_MARKER, &json, 1).replacen(HERO_MARKER, &hero, 1);
    std::fs::write(path, html)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn base64_matches_rfc4648() {
        assert_eq!(super::base64(b""), "");
        assert_eq!(super::base64(b"f"), "Zg==");
        assert_eq!(super::base64(b"fo"), "Zm8=");
        assert_eq!(super::base64(b"foo"), "Zm9v");
        assert_eq!(super::base64(b"foobar"), "Zm9vYmFy");
    }
}

pub fn human_bytes(b: u64) -> String {
    let b = b as f64;
    if b >= 1e9 {
        format!("{:.2} GB", b / 1e9)
    } else {
        format!("{:.0} MB", b / 1e6)
    }
}

fn tier_cell(t: Tier) -> String {
    let s = format!("{:<13}", t.label());
    match t {
        Tier::NearLossless => s.green().bold().to_string(),
        Tier::Excellent => s.green().to_string(),
        Tier::Noticeable => s.yellow().to_string(),
        Tier::Degraded => s.red().bold().to_string(),
    }
}

fn ci_pct(i: &crate::stats::Interval) -> String {
    format!("{:.1}% [{:.1}–{:.1}]", i.est * 100.0, i.lo * 100.0, i.hi * 100.0)
}

fn verdict(i: &crate::stats::Interval) -> String {
    if i.significant() {
        "real difference".red().to_string()
    } else {
        "could be noise".dimmed().to_string()
    }
}

fn print_stats(st: &crate::stats::Stats) {
    println!(
        "\n{}",
        format!(
            "  how sure are we? ({:.0}% bootstrap intervals, {} resamples, {}-token blocks)",
            st.level * 100.0,
            st.replicates,
            st.block_tokens
        )
        .bold()
    );
    for c in &st.candidates {
        let mut line = format!(
            "  {:<16} KL {:.4} [{:.4}–{:.4}] · top-1 {}",
            c.name,
            c.kl_mean.est,
            c.kl_mean.lo,
            c.kl_mean.hi,
            ci_pct(&c.top1)
        );
        if let Some(a) = &c.accuracy {
            line += &format!(" · correct {}", ci_pct(a));
        }
        if let Some(p) = &c.p_change {
            line += &format!(" · answers changed ~{}", ci_pct(p));
        }
        println!("{line}");
    }
    for p in &st.pairs {
        let mut line = format!(
            "  {:<22} KL {:+.4} [{:+.4}, {:+.4}] {}",
            format!("{} vs {}", p.smaller, p.larger),
            p.kl_diff.est,
            p.kl_diff.lo,
            p.kl_diff.hi,
            verdict(&p.kl_diff)
        );
        if let Some(a) = &p.acc_diff {
            line += &format!(
                " · correct {:+.0}% [{:+.0}, {:+.0}] {}",
                a.est * 100.0,
                a.lo * 100.0,
                a.hi * 100.0,
                verdict(a)
            );
        }
        println!("{line}");
    }
    if let Some(at) = &st.kl_at {
        println!(
            "  {}",
            format!(
                "tier boundaries: excellent < {:.4} [{:.4}–{:.4}], noticeable < {:.4} [{:.4}–{:.4}]{}",
                at[1].est,
                at[1].lo,
                at[1].hi,
                at[2].est,
                at[2].lo,
                at[2].hi,
                if st.calibration_failures > 0 {
                    format!(" ({} resamples could not be re-fitted)", st.calibration_failures)
                } else {
                    String::new()
                }
            )
            .dimmed()
        );
    }
    println!(
        "  recommendation stability: {}",
        st.stability
            .iter()
            .map(|(n, f)| format!("{} {:.0}%", n, f * 100.0))
            .collect::<Vec<_>>()
            .join(" · ")
    );
}

pub fn print_summary(r: &Results) {
    let ref_size = r.reference.info.size_bytes.max(1) as f64;
    println!("\n{}", "  results".bold());
    println!(
        "  {}",
        format!(
            "{:<16} {:>9} {:>7} {:>9} {:>9} {:>8} {:>8} {:>10}  {:<13}",
            "model", "size", "of ref", "KL mean", "KL p99", "top-1", "ΔPPL", "decode t/s", "quality"
        )
        .dimmed()
    );
    let decode = |b: &Option<crate::backend::BenchResult>| {
        b.as_ref().map(|b| format!("{:.1}", b.decode_tok_s)).unwrap_or_else(|| "-".into())
    };
    println!(
        "  {:<16} {:>9} {:>7} {:>9} {:>9} {:>8} {:>8} {:>10}  {}",
        r.reference.spec.name.bold(),
        human_bytes(r.reference.info.size_bytes),
        "100%",
        "0",
        "0",
        "100%",
        "0",
        decode(&r.reference.bench),
        "reference".dimmed()
    );
    for c in &r.candidates {
        let s = &c.summary;
        let name = if c.pareto { format!("{} ◆", c.model.spec.name) } else { c.model.spec.name.clone() };
        println!(
            "  {:<16} {:>9} {:>6.0}% {:>9.4} {:>9.3} {:>7.1}% {:>+7.2}% {:>10}  {}",
            name,
            human_bytes(c.model.info.size_bytes),
            c.model.info.size_bytes as f64 / ref_size * 100.0,
            s.kl_mean,
            s.kl_p99,
            s.top1_agreement * 100.0,
            s.ppl_delta_pct,
            decode(&c.model.bench),
            tier_cell(c.tier)
        );
    }
    println!("  {}", "◆ = Pareto-optimal (nothing else is both smaller and closer to the reference)".dimmed());

    if let Some(t) = &r.tasks {
        println!("\n{}", format!("  answers on {} tasks (greedy, up to {} tokens)", t.items.len(), t.gen_tokens).bold());
        println!(
            "  {}",
            format!("{:<16} {:>9} {:>8} {:>7} {:>11} {:>16}", "model", "correct", "broken", "fixed", "identical", "differs after").dimmed()
        );
        let names = std::iter::once(&r.reference.spec.name).chain(r.candidates.iter().map(|c| &c.model.spec.name));
        for (i, name) in names.enumerate() {
            let Some(s) = t.summary.get(name) else { continue };
            if i == 0 {
                println!("  {:<16} {:>9} {:>8} {:>7} {:>11} {:>16}", name.bold(), format!("{}/{}", s.correct, s.n), "-", "-", "-", "-");
                continue;
            }
            let broken = format!("{:>8}", s.regressions);
            let broken = if s.regressions == 0 { broken.green().to_string() } else { broken.red().to_string() };
            println!(
                "  {:<16} {:>9} {} {:>7} {:>10.0}% {:>16}",
                name,
                format!("{}/{}", s.correct, s.n),
                broken,
                s.fixes,
                s.identical as f64 / s.n.max(1) as f64 * 100.0,
                s.median_diverge.map(|d| format!("~{d:.0} tokens")).unwrap_or_else(|| "never".into())
            );
        }
        println!("  {}", "broken = the original answered correctly, this quant did not".dimmed());
    }

    match (&r.calibration, &r.calibration_note) {
        (Some(c), _) => {
            println!(
                "\n  {} on {} answers ({}): {}",
                "calibrated tiers".bold(),
                c.n,
                c.outcome,
                c.levels
                    .iter()
                    .zip(c.kl_at)
                    .map(|(p, kl)| format!("{:.0}% change at KL {}", p * 100.0, format!("{kl:.4}").bold()))
                    .collect::<Vec<_>>()
                    .join(" · ")
            );
            println!(
                "  {}",
                format!(
                    "tiers: near-lossless < {:.4} < excellent < {:.4} < noticeable < {:.4} < degraded",
                    c.kl_at[0], c.kl_at[1], c.kl_at[2]
                )
                .dimmed()
            );
            for cand in &r.candidates {
                println!(
                    "  {:<16} KL {:.4} → ~{:.0}% of answers expected to change",
                    cand.model.spec.name,
                    cand.summary.kl_mean,
                    c.p_change(cand.summary.kl_mean) * 100.0
                );
            }
        }
        (None, Some(note)) => println!("\n  {}", format!("tiers use default thresholds: {note}").dimmed()),
        _ => {}
    }

    if let Some(st) = &r.stats {
        print_stats(st);
    }

    // Where each candidate breaks first.
    for c in &r.candidates {
        if let Some(w) = c
            .by_category
            .iter()
            .max_by(|a, b| a.kl_mean.partial_cmp(&b.kl_mean).unwrap())
            .filter(|_| c.by_category.len() > 1)
        {
            println!(
                "  {} drifts most on {} (KL {:.4})",
                c.model.spec.name.yellow(),
                w.category.bold(),
                w.kl_mean
            );
        }
    }
    if let Some(rec) = &r.recommendation {
        println!("\n  {} {}  {}", "→ use".bold().green(), rec.name.bold().green(), format!("({})", rec.reason).dimmed());
    }
}
