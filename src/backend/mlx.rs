//! MLX models via a small Python worker (`workers/mlx_worker.py`, embedded in
//! the binary) that speaks JSON lines over stdin/stdout.

use super::{Backend, ModelInfo};
use crate::config::Settings;
use crate::dist::PosOut;
use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

const WORKER_SRC: &str = include_str!("../../workers/mlx_worker.py");

pub struct MlxWorker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    info: ModelInfo,
}

#[derive(Deserialize)]
struct LoadReply {
    n_vocab: usize,
    size_bytes: u64,
    n_params: Option<u64>,
    device: String,
}

#[derive(Deserialize)]
struct EvalReply {
    top_ids: Vec<Vec<u32>>,
    top_lp: Vec<Vec<f32>>,
    target: Vec<f32>,
    gathered: Option<Vec<Vec<f32>>>,
}

impl MlxWorker {
    pub fn open(path: &str, settings: &Settings) -> Result<Self> {
        let script = std::env::temp_dir().join(format!("kldrift_mlx_worker_{}.py", env!("CARGO_PKG_VERSION")));
        std::fs::write(&script, WORKER_SRC).context("cannot write MLX worker script")?;

        let mut child = Command::new(&settings.python)
            .arg("-u")
            .arg(&script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("cannot start '{}' for the MLX worker (pip install mlx-lm)", settings.python))?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("no worker stdin"))?;
        let stdout = BufReader::new(child.stdout.take().ok_or_else(|| anyhow!("no worker stdout"))?);

        let mut me = Self { child, stdin, stdout, info: ModelInfo::default() };
        let reply: LoadReply = serde_json::from_value(me.call(json!({"op": "load", "path": path}))?)?;
        me.info = ModelInfo {
            n_vocab: reply.n_vocab,
            size_bytes: reply.size_bytes,
            n_params: reply.n_params,
            device: reply.device,
        };
        Ok(me)
    }

    fn call(&mut self, req: Value) -> Result<Value> {
        writeln!(self.stdin, "{req}")?;
        self.stdin.flush()?;
        let mut line = String::new();
        if self.stdout.read_line(&mut line)? == 0 {
            bail!("MLX worker exited unexpectedly (is mlx-lm installed?)");
        }
        let v: Value = serde_json::from_str(&line).context("bad reply from MLX worker")?;
        if let Some(err) = v.get("error").and_then(Value::as_str) {
            bail!("MLX worker: {err}");
        }
        Ok(v)
    }
}

impl Drop for MlxWorker {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "{}", json!({"op": "quit"}));
        let _ = self.child.wait();
    }
}

impl Backend for MlxWorker {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn tokenize(&mut self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let v = self.call(json!({"op": "tokenize", "text": text, "add_special": add_special}))?;
        Ok(serde_json::from_value(v["ids"].clone())?)
    }

    fn pieces(&mut self, ids: &[u32], stream: bool) -> Result<Vec<String>> {
        let v = self.call(json!({"op": "pieces", "ids": ids, "stream": stream}))?;
        Ok(serde_json::from_value(v["pieces"].clone())?)
    }

    fn eval(&mut self, ids: &[u32], top_k: usize, gather: Option<&[Vec<u32>]>) -> Result<Vec<PosOut>> {
        let v = self.call(json!({"op": "eval", "ids": ids, "top_k": top_k, "gather": gather}))?;
        let r: EvalReply = serde_json::from_value(v)?;
        let n = r.target.len();
        let mut gathered = r.gathered.unwrap_or_default();
        gathered.resize(n, Vec::new());
        Ok((0..n)
            .map(|t| PosOut {
                top: r.top_ids[t].iter().copied().zip(r.top_lp[t].iter().copied()).collect(),
                gathered: std::mem::take(&mut gathered[t]),
                target: r.target[t],
            })
            .collect())
    }

    fn bench(&mut self, prompt: &[u32], gen: usize, around_decode: &mut dyn FnMut(bool)) -> Result<(f64, f64)> {
        // The worker reports its own timings; power is sampled around the
        // whole call, which is dominated by the decode loop.
        around_decode(true);
        let v = self.call(json!({"op": "bench", "ids": prompt, "gen": gen}));
        around_decode(false);
        let v = v?;
        let f = |k: &str| v[k].as_f64().ok_or_else(|| anyhow!("worker bench reply lacks {k}"));
        Ok((f("prefill_tok_s")?, f("decode_tok_s")?))
    }

    fn mem_bytes(&mut self) -> Option<u64> {
        self.call(json!({"op": "mem"})).ok()?.get("peak_bytes")?.as_u64()
    }
}
