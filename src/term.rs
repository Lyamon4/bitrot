//! Terminal colors that switch themselves off when they would be noise:
//! output piped to a file or a CI log, or `NO_COLOR` set (no-color.org).
//! Same method names as before (`.bold()`, `.dimmed()`, ...), so call sites
//! read naturally.

use std::fmt::Display;
use std::sync::atomic::{AtomicBool, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);

/// Decide once, at start-up, whether stdout gets ANSI colors.
pub fn init() {
    use std::io::IsTerminal;
    let on = std::io::stdout().is_terminal()
        && std::env::var_os("NO_COLOR").is_none_or(|v| v.is_empty())
        && std::env::var("TERM").map_or(true, |t| t != "dumb");
    ENABLED.store(on, Ordering::Relaxed);
}

fn paint(code: &str, text: String) -> String {
    if ENABLED.load(Ordering::Relaxed) {
        format!("\x1b[{code}m{text}\x1b[0m")
    } else {
        text
    }
}

pub trait Paint: Display {
    fn bold(&self) -> String {
        paint("1", self.to_string())
    }
    fn dimmed(&self) -> String {
        paint("2", self.to_string())
    }
    fn underline(&self) -> String {
        paint("4", self.to_string())
    }
    fn red(&self) -> String {
        paint("31", self.to_string())
    }
    fn green(&self) -> String {
        paint("32", self.to_string())
    }
    fn yellow(&self) -> String {
        paint("33", self.to_string())
    }
    fn cyan(&self) -> String {
        paint("36", self.to_string())
    }
}

impl<T: Display + ?Sized> Paint for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_when_disabled() {
        ENABLED.store(false, Ordering::Relaxed);
        assert_eq!("x".bold().green(), "x");
    }
}
