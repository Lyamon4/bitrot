use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    /// GGUF through an in-process llama.cpp.
    Llama,
    /// MLX model (local dir or Hugging Face repo id) through a Python worker.
    Mlx,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub name: String,
    pub backend: BackendKind,
    /// File path (GGUF) or directory / Hugging Face repo id (MLX).
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Directory with prompt texts: `<dir>/<category>/<name>.txt`.
    pub prompts: PathBuf,
    /// Tokens per evaluation window.
    pub ctx: usize,
    /// How many reference tokens to keep per position for the KL estimate.
    pub top_k: usize,
    /// Cap on tokens taken from each prompt file.
    pub max_tokens_per_doc: usize,
    pub bench_prompt_tokens: usize,
    pub bench_gen_tokens: usize,
    pub bench_repeats: usize,
    /// CPU threads for llama.cpp (0 = all physical cores).
    pub threads: usize,
    /// Layers to offload to the GPU in llama.cpp (Metal / CUDA builds).
    pub gpu_layers: u32,
    /// Python interpreter that has `mlx-lm` installed.
    pub python: String,
    /// Skip speed / energy measurement.
    pub no_bench: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            prompts: PathBuf::from("prompts"),
            ctx: 512,
            top_k: 64,
            max_tokens_per_doc: 1024,
            bench_prompt_tokens: 256,
            bench_gen_tokens: 64,
            bench_repeats: 3,
            threads: 0,
            gpu_layers: 999,
            python: if cfg!(windows) { "python".into() } else { "python3".into() },
            no_bench: false,
        }
    }
}

/// Mean-KL boundaries (nats) for the quality tiers in the report.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Thresholds {
    pub near_lossless: f64,
    pub excellent: f64,
    pub noticeable: f64,
    /// Minimum top-1 agreement for a candidate to be recommended.
    pub min_top1: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self { near_lossless: 0.01, excellent: 0.05, noticeable: 0.15, min_top1: 0.90 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub settings: Settings,
    #[serde(default)]
    pub thresholds: Thresholds,
    pub reference: ModelSpec,
    #[serde(rename = "candidate")]
    pub candidates: Vec<ModelSpec>,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read config {}", path.display()))?;
        let mut cfg: Config =
            toml::from_str(&text).with_context(|| format!("invalid config {}", path.display()))?;
        // Relative paths in a config file are relative to that file.
        let base = path.parent().unwrap_or(Path::new("."));
        let rel = |p: &str| crate::sys::clean_path(&base.join(p)).to_string_lossy().into_owned();
        if cfg.settings.prompts.is_relative() {
            cfg.settings.prompts = crate::sys::clean_path(&base.join(&cfg.settings.prompts));
        }
        for m in std::iter::once(&mut cfg.reference).chain(cfg.candidates.iter_mut()) {
            if matches!(m.backend, BackendKind::Llama) && Path::new(&m.path).is_relative() {
                m.path = rel(&m.path);
            }
            if matches!(m.backend, BackendKind::Mlx) && base.join(&m.path).exists() {
                m.path = rel(&m.path);
            }
        }
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        if self.candidates.is_empty() {
            bail!("config needs at least one [[candidate]]");
        }
        if self.settings.ctx < 16 {
            bail!("settings.ctx must be at least 16");
        }
        if self.settings.top_k == 0 {
            bail!("settings.top_k must be > 0");
        }
        let mut names = std::collections::HashSet::new();
        for m in std::iter::once(&self.reference).chain(self.candidates.iter()) {
            if !names.insert(m.name.as_str()) {
                bail!("model name '{}' is used twice", m.name);
            }
        }
        Ok(())
    }
}

/// Guess a short display name for a model file, e.g. `Q4_K_M` from
/// `SmolLM2-135M-Instruct-Q4_K_M.gguf`.
pub fn guess_name(path: &str) -> String {
    let stem = Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string());
    let upper = stem.to_uppercase();
    for marker in ["IQ", "Q", "F16", "BF16", "F32"] {
        if let Some(pos) = upper.rfind(&format!("-{marker}")).or_else(|| upper.rfind(&format!(".{marker}"))) {
            return stem[pos + 1..].to_string();
        }
    }
    stem
}

#[cfg(test)]
mod tests {
    use super::guess_name;

    #[test]
    fn guesses_quant_names() {
        assert_eq!(guess_name("models/SmolLM2-135M-Instruct-Q4_K_M.gguf"), "Q4_K_M");
        assert_eq!(guess_name("x/llama-3-8b.Q8_0.gguf"), "Q8_0");
        assert_eq!(guess_name("SmolLM2-135M-Instruct-f16.gguf"), "f16");
        assert_eq!(guess_name("model.gguf"), "model");
    }
}
