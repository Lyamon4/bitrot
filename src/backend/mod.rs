//! Model backends. Each one can run a fixed token sequence through a model
//! (teacher forcing) and report per-position distributions, plus a small
//! speed benchmark.

#[cfg(feature = "llama")]
pub mod llama;
pub mod mlx;

use crate::config::{BackendKind, ModelSpec, Settings};
use crate::dist::PosOut;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelInfo {
    pub n_vocab: usize,
    /// Size of the weights on disk / in memory, bytes.
    pub size_bytes: u64,
    pub n_params: Option<u64>,
    /// Where the math runs, e.g. "Metal", "CPU", "MLX (gpu)".
    pub device: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchResult {
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    /// Memory the model needs while running (weights + KV + scratch), bytes.
    pub mem_bytes: Option<u64>,
    /// Package energy spent per generated token, millijoules (macOS only).
    pub energy_mj_per_tok: Option<f64>,
    /// Average package power while decoding, watts (macOS only).
    pub decode_watts: Option<f64>,
}

pub trait Backend {
    fn info(&self) -> &ModelInfo;

    /// Tokenize plain text. `add_special` adds BOS etc. as the model expects.
    fn tokenize(&mut self, text: &str, add_special: bool) -> Result<Vec<u32>>;

    /// Text of each token, used to draw the report.
    ///
    /// With `stream = true` the ids are a running text: a character whose
    /// UTF-8 bytes are split over several byte-level tokens (common for
    /// Cyrillic, CJK, emoji) is attributed to the token that completes it, so
    /// concatenating the pieces gives back the exact text. With
    /// `stream = false` each id is decoded on its own.
    fn pieces(&mut self, ids: &[u32], stream: bool) -> Result<Vec<String>>;

    /// Run `ids` through the model and return one [`PosOut`] for every
    /// position except the last (each predicts the following token).
    /// `gather[t]` lists token ids whose log-prob is needed at position `t`.
    fn eval(&mut self, ids: &[u32], top_k: usize, gather: Option<&[Vec<u32>]>) -> Result<Vec<PosOut>>;

    /// Time prefill of `prompt` and greedy decoding of `gen` tokens.
    /// `around_decode` is called right before and after the decode loop so
    /// the caller can sample power counters.
    fn bench(&mut self, prompt: &[u32], gen: usize, around_decode: &mut dyn FnMut(bool)) -> Result<(f64, f64)>;

    /// Memory currently held for this model, if the backend can tell.
    fn mem_bytes(&mut self) -> Option<u64>;

    /// Token ids for a single user message wrapped in the model's chat
    /// template (or the plain text if the model has none).
    fn chat_prompt(&mut self, user: &str) -> Result<Vec<u32>>;

    /// Greedy continuation of `prompt`, at most `max_new` tokens, stopping
    /// at the model's end-of-generation token (which is not returned).
    fn generate(&mut self, prompt: &[u32], max_new: usize) -> Result<Vec<u32>>;
}

pub fn open(spec: &ModelSpec, settings: &Settings) -> Result<Box<dyn Backend>> {
    match spec.backend {
        BackendKind::Llama => {
            #[cfg(feature = "llama")]
            {
                Ok(Box::new(llama::LlamaCpp::open(&spec.path, settings)?))
            }
            #[cfg(not(feature = "llama"))]
            {
                let _ = settings;
                bail!("this build has no llama.cpp backend (rebuild with --features llama)")
            }
        }
        BackendKind::Mlx => {
            if !cfg!(target_os = "macos") {
                bail!("model '{}': the MLX backend needs Apple Silicon macOS", spec.name);
            }
            Ok(Box::new(mlx::MlxWorker::open(&spec.path, settings)?))
        }
    }
}

/// Turn per-token raw bytes into per-token strings whose concatenation is the
/// decoded text. Bytes of an incomplete UTF-8 sequence are carried over to
/// the next token; invalid bytes become U+FFFD.
pub fn utf8_stream(pieces: Vec<Vec<u8>>) -> Vec<String> {
    let mut pending: Vec<u8> = Vec::new();
    let mut out = Vec::with_capacity(pieces.len());
    for bytes in pieces {
        pending.extend_from_slice(&bytes);
        let mut s = String::new();
        loop {
            match std::str::from_utf8(&pending) {
                Ok(valid) => {
                    s.push_str(valid);
                    pending.clear();
                    break;
                }
                Err(e) => {
                    let good = e.valid_up_to();
                    s.push_str(std::str::from_utf8(&pending[..good]).unwrap());
                    match e.error_len() {
                        // Truncated sequence: wait for the next token.
                        None => {
                            pending.drain(..good);
                            break;
                        }
                        Some(bad) => {
                            s.push('\u{FFFD}');
                            pending.drain(..good + bad);
                        }
                    }
                }
            }
        }
        out.push(s);
    }
    if !pending.is_empty() {
        if let Some(last) = out.last_mut() {
            last.push_str(&String::from_utf8_lossy(&pending));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::utf8_stream;

    #[test]
    fn split_cyrillic_is_reassembled() {
        let text = "Почему небо";
        let bytes = text.as_bytes();
        // Split "П" (2 bytes) across tokens: [D0], [9F, D0, BE ...]
        let pieces = vec![bytes[..1].to_vec(), bytes[1..5].to_vec(), bytes[5..].to_vec()];
        let out = utf8_stream(pieces);
        assert_eq!(out[0], "");
        assert_eq!(out.concat(), text);
        assert!(!out.concat().contains('\u{FFFD}'));
    }

    #[test]
    fn invalid_bytes_are_replaced() {
        let out = utf8_stream(vec![vec![b'a', 0xFF, b'b']]);
        assert_eq!(out, vec!["a\u{FFFD}b".to_string()]);
    }
}

/// Index of the maximum element.
pub fn argmax(xs: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in xs.iter().enumerate() {
        if x > xs[best] {
            best = i;
        }
    }
    best as u32
}
