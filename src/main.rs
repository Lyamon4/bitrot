//! bitrot - how much did quantization actually change your model?

mod backend;
mod banner;
mod calibrate;
mod config;
mod dist;
mod metrics;
mod recommend;
mod report;
mod runner;
mod stats;
mod sys;
mod tasks;
mod term;
mod tui;
mod ui;

use anyhow::{bail, Context, Result};
use clap::{CommandFactory, FromArgMatches, Parser, Subcommand};
use config::{guess_name, BackendKind, Config, ModelSpec, Settings, Thresholds};
use crate::term::Paint;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "bitrot",
    version,
    about = "Measure how far quantized LLMs drift from the original - per token, on your prompts, on your hardware.",
    long_about = None
)]
struct Cli {
    /// Without a command, bitrot opens the terminal interface.
    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compare quants straight from the command line.
    ///
    /// Models are GGUF paths, or `mlx:<dir or hf-repo>` for MLX models.
    Compare {
        /// The full-precision reference (e.g. an F16 GGUF).
        reference: String,
        /// One or more quantized versions of the same model.
        #[arg(required = true)]
        candidates: Vec<String>,
        #[command(flatten)]
        opts: RunOpts,
    },
    /// Run a comparison described by a TOML config file.
    Run {
        /// Path to the config (see examples/).
        #[arg(short, long, default_value = "bitrot.toml")]
        config: PathBuf,
        #[command(flatten)]
        opts: RunOpts,
    },
    /// Open the terminal interface: pick models, run, browse results.
    Tui {
        /// Folder with the GGUF files to compare.
        #[arg(short, long, default_value = "models")]
        models: PathBuf,
        /// Output directory for results.json and report.html.
        #[arg(short, long, default_value = "bitrot-out")]
        out: PathBuf,
    },
    /// Browse a saved results.json in the terminal interface.
    View {
        #[arg(default_value = "bitrot-out/results.json")]
        results: PathBuf,
        /// Instead of opening the interface, render it to an SVG file.
        #[arg(long, hide = true)]
        svg: Option<PathBuf>,
        /// Screen for --svg: setup, or a results tab 1-4.
        #[arg(long, hide = true, default_value = "1")]
        shot: String,
    },
    /// Re-render the HTML report from a saved results.json.
    Report {
        results: PathBuf,
        #[arg(short, long)]
        out: Option<PathBuf>,
        #[arg(long)]
        open: bool,
    },
}

#[derive(clap::Args)]
struct RunOpts {
    /// Prompt directory (`<dir>/<category>/<name>.txt`).
    #[arg(short, long)]
    prompts: Option<PathBuf>,
    /// Output directory for results.json and report.html.
    #[arg(short, long, default_value = "bitrot-out")]
    out: PathBuf,
    /// Tokens per evaluation window.
    #[arg(long)]
    ctx: Option<usize>,
    /// Reference tokens kept per position for the KL estimate.
    #[arg(long)]
    top_k: Option<usize>,
    /// Cap on tokens taken from each prompt file.
    #[arg(long)]
    max_tokens: Option<usize>,
    /// Skip the speed / memory / energy benchmark.
    #[arg(long)]
    no_bench: bool,
    /// Task directory (`*.jsonl` with checkable answers).
    #[arg(long)]
    tasks: Option<PathBuf>,
    /// Max tokens each model generates per task.
    #[arg(long)]
    gen_tokens: Option<usize>,
    /// Skip the answer check on tasks.
    #[arg(long)]
    no_tasks: bool,
    /// Bootstrap resamples for confidence intervals (0 = off).
    #[arg(long)]
    bootstrap: Option<usize>,
    /// CPU threads for llama.cpp (default: one per physical core).
    #[arg(long)]
    threads: Option<usize>,
    /// Python with mlx-lm installed, for MLX models.
    #[arg(long)]
    python: Option<String>,
    /// Open the report in the browser when done.
    #[arg(long)]
    open: bool,
}

impl RunOpts {
    fn apply(&self, s: &mut Settings) {
        if let Some(p) = &self.prompts {
            s.prompts = p.clone();
        }
        if let Some(v) = self.ctx {
            s.ctx = v;
        }
        if let Some(v) = self.top_k {
            s.top_k = v;
        }
        if let Some(v) = self.max_tokens {
            s.max_tokens_per_doc = v;
        }
        if let Some(v) = self.threads {
            s.threads = v;
        }
        if let Some(v) = &self.python {
            s.python = v.clone();
        }
        if let Some(p) = &self.tasks {
            s.tasks = p.clone();
        }
        if let Some(v) = self.gen_tokens {
            s.gen_tokens = v;
        }
        s.no_bench |= self.no_bench;
        s.no_tasks |= self.no_tasks;
        if let Some(v) = self.bootstrap {
            s.bootstrap = v;
        }
    }
}

fn spec_from_arg(arg: &str) -> ModelSpec {
    match arg.strip_prefix("mlx:") {
        Some(path) => ModelSpec {
            name: Path::new(path).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| path.into()),
            backend: BackendKind::Mlx,
            path: path.into(),
        },
        None => ModelSpec { name: guess_name(arg), backend: BackendKind::Llama, path: arg.into() },
    }
}

/// Make model names unique so they can key the report.
fn dedupe_names(cfg: &mut Config) {
    let mut seen = std::collections::HashMap::<String, usize>::new();
    for m in std::iter::once(&mut cfg.reference).chain(cfg.candidates.iter_mut()) {
        let n = seen.entry(m.name.clone()).or_insert(0);
        *n += 1;
        if *n > 1 {
            m.name = format!("{} ({})", m.name, n);
        }
    }
}

fn tui_settings() -> Settings {
    Settings { prompts: bundled_dir("prompts"), tasks: bundled_dir("tasks"), ..Settings::default() }
}

/// Prefer `./<name>`, then the copy shipped next to the crate's sources.
fn bundled_dir(name: &str) -> PathBuf {
    let local = PathBuf::from(name);
    if local.is_dir() {
        return local;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(name)
}

fn execute(mut cfg: Config, opts: &RunOpts) -> Result<()> {
    opts.apply(&mut cfg.settings);
    cfg.validate()?;
    if std::io::IsTerminal::is_terminal(&std::io::stdout()) {
        banner::print();
    }
    let results = runner::run(&cfg)?;
    let (json_path, html_path) = report::save(&results, &opts.out)?;
    report::print_summary(&results);
    println!("\n  {}  {}", "report".bold(), html_path.display().underline());
    println!("  {}    {}", "data".bold(), json_path.display());
    if opts.open {
        open_in_browser(&html_path);
    }
    Ok(())
}

pub(crate) fn open_in_browser(path: &Path) {
    let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let _ = if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(&path).status()
    } else if cfg!(windows) {
        std::process::Command::new("cmd").args(["/C", "start", ""]).arg(&path).status()
    } else {
        std::process::Command::new("xdg-open").arg(&path).status()
    };
}

fn main() -> Result<()> {
    term::init();
    let matches = Cli::command().before_help(banner::plain()).get_matches();
    let cli = Cli::from_arg_matches(&matches)?;
    let Some(command) = cli.command else {
        return tui::run(PathBuf::from("models"), tui_settings(), PathBuf::from("bitrot-out"));
    };
    match command {
        Cmd::Tui { models, out } => tui::run(models, tui_settings(), out),
        Cmd::View { results, svg: Some(svg), shot } => {
            let shot = match shot.as_str() {
                "setup" => tui::Shot::Setup,
                n => tui::Shot::Results(n.parse::<usize>().unwrap_or(1).saturating_sub(1)),
            };
            tui::snapshot_svg(&results, Path::new("models"), shot, &svg, 118, 26)
        }
        Cmd::View { results, .. } => tui::view(&results),
        Cmd::Compare { reference, candidates, opts } => {
            let mut cfg = Config {
                settings: Settings { prompts: bundled_dir("prompts"), tasks: bundled_dir("tasks"), ..Settings::default() },
                thresholds: Thresholds::default(),
                reference: spec_from_arg(&reference),
                candidates: candidates.iter().map(|c| spec_from_arg(c)).collect(),
            };
            dedupe_names(&mut cfg);
            execute(cfg, &opts)
        }
        Cmd::Run { config, opts } => {
            if !config.exists() {
                bail!("config {} not found (see examples/smollm2.toml)", config.display());
            }
            execute(Config::load(&config)?, &opts)
        }
        Cmd::Report { results, out, open } => {
            let data: runner::Results = serde_json::from_slice(&std::fs::read(&results)?)
                .with_context(|| format!("{} is not a bitrot results file", results.display()))?;
            let out = out.unwrap_or_else(|| results.with_file_name("report.html"));
            report::write_html(&data, &out)?;
            println!("wrote {}", out.display());
            if open {
                open_in_browser(&out);
            }
            Ok(())
        }
    }
}
