//! Where progress goes. In the plain CLI it is printed (with indicatif bars);
//! inside the TUI the same calls become events on a channel that the
//! interface draws.

use indicatif::{ProgressBar, ProgressStyle};
use std::sync::mpsc::Sender;
use std::sync::Mutex;

#[derive(Debug, Clone)]
pub enum Event {
    /// A log line (may contain newlines).
    Line(String),
    /// A new progress stage starts.
    Stage { label: String, unit: String, total: usize },
    /// One more unit of the current stage is done.
    Tick,
    /// The current stage finished, with a short note (e.g. "3.1s").
    StageDone(String),
}

static SINK: Mutex<Option<Sender<Event>>> = Mutex::new(None);

/// Route all output to `tx` instead of the terminal.
pub fn set_sink(tx: Sender<Event>) {
    *SINK.lock().unwrap() = Some(tx);
}

fn send(ev: Event) -> bool {
    match SINK.lock().unwrap().as_ref() {
        Some(tx) => {
            let _ = tx.send(ev);
            true
        }
        None => false,
    }
}

pub fn line(s: String) {
    if !send(Event::Line(s.clone())) {
        println!("{s}");
    }
}

/// `println!`-style macro that goes through [`line`].
#[macro_export]
macro_rules! say {
    () => { $crate::ui::line(String::new()) };
    ($($t:tt)*) => { $crate::ui::line(format!($($t)*)) };
}

/// A progress bar that is either an indicatif bar or a stream of events.
pub struct Progress {
    bar: Option<ProgressBar>,
}

impl Progress {
    pub fn new(len: usize, label: &str, unit: &str) -> Self {
        if send(Event::Stage { label: label.to_string(), unit: unit.to_string(), total: len }) {
            return Self { bar: None };
        }
        let pb = ProgressBar::new(len as u64);
        pb.set_style(
            ProgressStyle::with_template(&format!("  {{prefix:<14}} {{bar:28.cyan/blue}} {{pos}}/{{len}} {unit:<7}  {{msg}}"))
                .unwrap()
                .progress_chars("━╸─"),
        );
        pb.set_prefix(label.to_string());
        Self { bar: Some(pb) }
    }

    pub fn inc(&self, n: u64) {
        match &self.bar {
            Some(pb) => pb.inc(n),
            None => {
                for _ in 0..n {
                    send(Event::Tick);
                }
            }
        }
    }

    pub fn finish_with_message(&self, msg: String) {
        match &self.bar {
            Some(pb) => pb.finish_with_message(msg),
            None => {
                send(Event::StageDone(msg));
            }
        }
    }
}
