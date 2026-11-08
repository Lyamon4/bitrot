//! Terminal summary and the self-contained HTML report.

use crate::metrics::Tier;
use crate::runner::Results;
use anyhow::Result;
use owo_colors::OwoColorize;
use std::path::Path;

const TEMPLATE: &str = include_str!("../assets/report.html");
const DATA_MARKER: &str = "/*__KLDRIFT_DATA__*/null";

pub fn write_html(results: &Results, path: &Path) -> Result<()> {
    // Keep `</script>` and friends inside strings from closing the tag.
    let json = serde_json::to_string(results)?.replace("</", "<\\/").replace("<!--", "<\\!--");
    let html = TEMPLATE.replacen(DATA_MARKER, &json, 1);
    std::fs::write(path, html)?;
    Ok(())
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
