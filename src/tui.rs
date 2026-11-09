//! Terminal interface: pick models, watch the run, browse the results.
//!
//! `bitrot` with no arguments opens it; `bitrot view results.json` opens
//! the results screen for a saved run.

use crate::banner::{self, Role};
use crate::config::{guess_name, BackendKind, Config, ModelSpec, Settings, Thresholds};
use crate::metrics::Tier;
use crate::runner::Results;
use crate::ui::{self, Event as UiEvent};
use anyhow::{anyhow, Result};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Cell, Gauge, Padding, Paragraph, Row, Table, Tabs, Wrap};
use ratatui::{DefaultTerminal, Frame};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const INK: Color = Color::Rgb(242, 242, 240);
const INK2: Color = Color::Rgb(201, 201, 197);
const MUTED: Color = Color::Rgb(133, 133, 127);
const LINE: Color = Color::Rgb(52, 52, 50);
const ACCENT: Color = Color::Rgb(142, 197, 255);
const WARN: Color = Color::Rgb(217, 164, 65);
const BAD: Color = Color::Rgb(229, 72, 77);

fn tier_color(t: Tier) -> Color {
    match t {
        Tier::NearLossless => INK,
        Tier::Excellent => ACCENT,
        Tier::Noticeable => WARN,
        Tier::Degraded => BAD,
    }
}

fn tier_words(t: Tier) -> &'static str {
    match t {
        Tier::NearLossless => "practically identical",
        Tier::Excellent => "barely changed",
        Tier::Noticeable => "noticeably changed",
        Tier::Degraded => "badly damaged",
    }
}

fn human(b: u64) -> String {
    crate::report::human_bytes(b)
}

struct ModelFile {
    path: PathBuf,
    name: String,
    size: u64,
}

struct Stage {
    label: String,
    unit: String,
    total: usize,
    done: usize,
    note: Option<String>,
}

enum Screen {
    Setup,
    Running,
    Results,
}

const TABS: [&str; 4] = ["Overview", "Where it breaks", "Answers", "Is it luck?"];

struct App {
    screen: Screen,
    models_dir: PathBuf,
    files: Vec<ModelFile>,
    cursor: usize,
    reference: Option<usize>,
    cands: Vec<bool>,
    settings: Settings,
    out: PathBuf,
    note: Option<String>,
    rx: Option<Receiver<UiEvent>>,
    job: Option<JoinHandle<Result<(Results, PathBuf)>>>,
    stages: Vec<Stage>,
    log: Vec<String>,
    started: Instant,
    results: Option<Results>,
    report: Option<PathBuf>,
    tab: usize,
    scroll: u16,
    cand: usize,
    quit: bool,
}

fn scan(dir: &Path) -> Vec<ModelFile> {
    let mut files: Vec<ModelFile> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf")))
        .map(|e| {
            let path = e.path();
            ModelFile {
                name: guess_name(&path.to_string_lossy()),
                size: e.metadata().map(|m| m.len()).unwrap_or(0),
                path,
            }
        })
        .collect();
    files.sort_by(|a, b| b.size.cmp(&a.size));
    files
}

impl App {
    fn new(models_dir: PathBuf, settings: Settings, out: PathBuf) -> Self {
        let files = scan(&models_dir);
        // Pre-select: an F16/BF16/F32 file (or the largest) as the reference,
        // everything else as candidates.
        let reference = files
            .iter()
            .position(|f| matches!(f.name.to_uppercase().as_str(), "F16" | "BF16" | "F32"))
            .or(if files.is_empty() { None } else { Some(0) });
        let cands = (0..files.len()).map(|i| Some(i) != reference).collect();
        Self {
            screen: Screen::Setup,
            models_dir,
            files,
            cursor: 0,
            reference,
            cands,
            settings,
            out,
            note: None,
            rx: None,
            job: None,
            stages: Vec::new(),
            log: Vec::new(),
            started: Instant::now(),
            results: None,
            report: None,
            tab: 0,
            scroll: 0,
            cand: 0,
            quit: false,
        }
    }

    fn start(&mut self) {
        let Some(r) = self.reference else {
            self.note = Some("pick a reference with r".into());
            return;
        };
        let spec = |f: &ModelFile| ModelSpec {
            name: f.name.clone(),
            backend: BackendKind::Llama,
            path: f.path.to_string_lossy().into_owned(),
        };
        let candidates: Vec<ModelSpec> =
            self.files.iter().enumerate().filter(|(i, _)| self.cands[*i] && *i != r).map(|(_, f)| spec(f)).collect();
        if candidates.is_empty() {
            self.note = Some("select at least one quant to compare with space".into());
            return;
        }
        let cfg = Config {
            settings: self.settings.clone(),
            thresholds: Thresholds::default(),
            reference: spec(&self.files[r]),
            candidates,
        };
        if let Err(e) = cfg.validate() {
            self.note = Some(e.to_string());
            return;
        }
        let (tx, rx) = channel();
        ui::set_sink(tx);
        let out = self.out.clone();
        self.job = Some(std::thread::spawn(move || {
            let results = crate::runner::run(&cfg)?;
            let (_, html) = crate::report::save(&results, &out)?;
            Ok((results, html))
        }));
        self.rx = Some(rx);
        self.started = Instant::now();
        self.screen = Screen::Running;
    }

    fn pump(&mut self) {
        if let Some(rx) = &self.rx {
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    UiEvent::Line(s) => {
                        for l in s.lines() {
                            if !l.trim().is_empty() {
                                self.log.push(l.trim_end().to_string());
                            }
                        }
                    }
                    UiEvent::Stage { label, unit, total } => {
                        self.stages.push(Stage { label, unit, total, done: 0, note: None })
                    }
                    UiEvent::Tick => {
                        if let Some(s) = self.stages.last_mut() {
                            s.done += 1;
                        }
                    }
                    UiEvent::StageDone(n) => {
                        if let Some(s) = self.stages.last_mut() {
                            s.done = s.total;
                            s.note = Some(n);
                        }
                    }
                }
            }
        }
        if self.job.as_ref().is_some_and(|j| j.is_finished()) {
            match self.job.take().unwrap().join() {
                Ok(Ok((results, html))) => {
                    self.results = Some(results);
                    self.report = Some(html);
                    self.screen = Screen::Results;
                }
                Ok(Err(e)) => self.log.push(format!("error: {e:#}")),
                Err(_) => self.log.push("error: the run crashed".into()),
            }
        }
    }

    fn key(&mut self, code: KeyCode, mods: KeyModifiers) {
        if code == KeyCode::Char('c') && mods.contains(KeyModifiers::CONTROL) {
            self.quit = true;
            return;
        }
        match self.screen {
            Screen::Setup => match code {
                KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
                KeyCode::Up | KeyCode::Char('k') => self.cursor = self.cursor.saturating_sub(1),
                KeyCode::Down | KeyCode::Char('j') => {
                    if self.cursor + 1 < self.files.len() {
                        self.cursor += 1
                    }
                }
                KeyCode::Char('r') if !self.files.is_empty() => {
                    self.reference = Some(self.cursor);
                    self.cands[self.cursor] = false;
                }
                KeyCode::Char(' ') if !self.files.is_empty() => {
                    if Some(self.cursor) != self.reference {
                        self.cands[self.cursor] = !self.cands[self.cursor];
                    }
                }
                KeyCode::Char('t') => self.settings.no_tasks = !self.settings.no_tasks,
                KeyCode::Char('b') => self.settings.no_bench = !self.settings.no_bench,
                KeyCode::Enter => self.start(),
                _ => {}
            },
            Screen::Running => {
                if matches!(code, KeyCode::Char('q') | KeyCode::Esc) {
                    self.quit = true;
                }
            }
            Screen::Results => match code {
                KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
                KeyCode::Tab | KeyCode::Right | KeyCode::Char('l') => {
                    self.tab = (self.tab + 1) % TABS.len();
                    self.scroll = 0;
                }
                KeyCode::BackTab | KeyCode::Left | KeyCode::Char('h') => {
                    self.tab = (self.tab + TABS.len() - 1) % TABS.len();
                    self.scroll = 0;
                }
                KeyCode::Char(c @ '1'..='4') => {
                    self.tab = c as usize - '1' as usize;
                    self.scroll = 0;
                }
                KeyCode::Down | KeyCode::Char('j') => self.scroll = self.scroll.saturating_add(1),
                KeyCode::Up | KeyCode::Char('k') => self.scroll = self.scroll.saturating_sub(1),
                KeyCode::PageDown => self.scroll = self.scroll.saturating_add(10),
                KeyCode::PageUp => self.scroll = self.scroll.saturating_sub(10),
                KeyCode::Char('c') => {
                    let n = self.results.as_ref().map_or(1, |r| r.candidates.len().max(1));
                    self.cand = (self.cand + 1) % n;
                    self.scroll = 0;
                }
                KeyCode::Char('o') => {
                    if let Some(p) = &self.report {
                        crate::open_in_browser(p);
                    }
                }
                _ => {}
            },
        }
    }

    // ------------------------------------------------------------------ draw
    fn draw(&self, f: &mut Frame) {
        let area = f.area();
        let [head, body, foot] =
            Layout::vertical([Constraint::Length(6), Constraint::Min(5), Constraint::Length(1)]).areas(area);
        draw_banner(f, head);
        match self.screen {
            Screen::Setup => self.draw_setup(f, body),
            Screen::Running => self.draw_running(f, body),
            Screen::Results => self.draw_results(f, body),
        }
        let keys: &[(&str, &str)] = match self.screen {
            Screen::Setup => &[("↑↓", "move"), ("r", "reference"), ("space", "compare"), ("t", "tasks"), ("b", "speed"), ("enter", "run"), ("q", "quit")],
            Screen::Running => &[("q", "quit")],
            Screen::Results => &[("←→", "tabs"), ("↑↓", "scroll"), ("c", "next quant"), ("o", "open html report"), ("q", "quit")],
        };
        let mut spans = vec![Span::raw(" ")];
        for (k, v) in keys {
            spans.push(Span::styled(format!(" {k} "), Style::new().fg(Color::Black).bg(INK2)));
            spans.push(Span::styled(format!(" {v}   "), Style::new().fg(MUTED)));
        }
        f.render_widget(Paragraph::new(Line::from(spans)), foot);
    }

    fn draw_setup(&self, f: &mut Frame, area: Rect) {
        let [left, right] = Layout::horizontal([Constraint::Min(40), Constraint::Length(38)]).areas(area);
        let mut lines: Vec<Line> = Vec::new();
        if self.files.is_empty() {
            lines.push(Line::styled(
                format!("No .gguf files in {}.", self.models_dir.display()),
                Style::new().fg(INK),
            ));
            lines.push(Line::styled(
                "Put an F16 model and a few quants of it there, or run: bitrot tui --models <dir>",
                Style::new().fg(MUTED),
            ));
        }
        for (i, file) in self.files.iter().enumerate() {
            let sel = i == self.cursor;
            let is_ref = Some(i) == self.reference;
            let mark = if is_ref {
                Span::styled(" original ", Style::new().fg(Color::Black).bg(ACCENT))
            } else if self.cands[i] {
                Span::styled(" [x]      ", Style::new().fg(INK))
            } else {
                Span::styled(" [ ]      ", Style::new().fg(MUTED))
            };
            let name_style = if sel { Style::new().fg(INK).add_modifier(Modifier::BOLD) } else { Style::new().fg(INK2) };
            lines.push(Line::from(vec![
                Span::styled(if sel { "▸ " } else { "  " }, Style::new().fg(ACCENT)),
                mark,
                Span::styled(format!("  {:<12}", file.name), name_style),
                Span::styled(format!("{:>9}   ", human(file.size)), Style::new().fg(INK2)),
                Span::styled(
                    file.path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
                    Style::new().fg(MUTED),
                ),
            ]));
        }
        f.render_widget(
            Paragraph::new(lines).block(panel(&format!("Models in {}", self.models_dir.display()))),
            left,
        );

        let onoff = |b: bool| if b { Span::styled("on ", Style::new().fg(ACCENT)) } else { Span::styled("off", Style::new().fg(MUTED)) };
        let n_cands = self.cands.iter().filter(|&&c| c).count();
        let mut side = vec![
            Line::styled("Which version should you download?", Style::new().fg(INK)),
            Line::raw(""),
            Line::styled(
                "bitrot runs the same text through the original and each smaller version and compares them word by word.",
                Style::new().fg(INK2),
            ),
            Line::raw(""),
            Line::from(vec![Span::styled("comparing     ", Style::new().fg(MUTED)), Span::styled(format!("{n_cands} quants"), Style::new().fg(INK))]),
            Line::from(vec![Span::styled("test answers  ", Style::new().fg(MUTED)), onoff(!self.settings.no_tasks)]),
            Line::from(vec![Span::styled("speed         ", Style::new().fg(MUTED)), onoff(!self.settings.no_bench)]),
            Line::from(vec![Span::styled("report        ", Style::new().fg(MUTED)), Span::styled(self.out.join("report.html").display().to_string(), Style::new().fg(INK2))]),
        ];
        if let Some(n) = &self.note {
            side.push(Line::raw(""));
            side.push(Line::styled(n.clone(), Style::new().fg(WARN)));
        }
        f.render_widget(Paragraph::new(side).wrap(Wrap { trim: false }).block(panel("Run")), right);
    }

    fn draw_running(&self, f: &mut Frame, area: Rect) {
        let n = self.stages.len().min(((area.height as usize).saturating_sub(4)) / 2).max(1);
        let gauges_h = (n * 2 + 2) as u16;
        let [top, bottom] = Layout::vertical([Constraint::Length(gauges_h), Constraint::Min(3)]).areas(area);
        let block = panel(&format!("Running · {:.0}s", self.started.elapsed().as_secs_f64()));
        let inner = block.inner(top);
        f.render_widget(block, top);
        let shown = &self.stages[self.stages.len().saturating_sub(n)..];
        let rows = Layout::vertical(shown.iter().map(|_| Constraint::Length(2))).split(inner);
        for (s, r) in shown.iter().zip(rows.iter()) {
            let ratio = if s.total == 0 { 1.0 } else { s.done as f64 / s.total as f64 };
            let label = format!(
                "{}  {}/{} {}{}",
                s.label,
                s.done.min(s.total),
                s.total,
                s.unit,
                s.note.as_ref().map(|n| format!("  ·  {n}")).unwrap_or_default()
            );
            let [l, g] = Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(*r);
            f.render_widget(Paragraph::new(Span::styled(label, Style::new().fg(INK2))), l);
            let color = if s.note.is_some() { MUTED } else { ACCENT };
            f.render_widget(Gauge::default().ratio(ratio.min(1.0)).label("").gauge_style(Style::new().fg(color).bg(LINE)), g);
        }
        let tail = (bottom.height as usize).saturating_sub(2);
        let log: Vec<Line> = self.log[self.log.len().saturating_sub(tail)..]
            .iter()
            .map(|l| {
                let style = if l.starts_with("error") { Style::new().fg(BAD) } else { Style::new().fg(INK2) };
                Line::styled(l.clone(), style)
            })
            .collect();
        f.render_widget(Paragraph::new(log).block(panel("Log")), bottom);
    }

    fn draw_results(&self, f: &mut Frame, area: Rect) {
        let Some(r) = &self.results else { return };
        let [verdict, tabs, body] =
            Layout::vertical([Constraint::Length(3), Constraint::Length(1), Constraint::Min(3)]).areas(area);

        let mut v = Vec::new();
        if let Some(rec) = &r.recommendation {
            let mut l = vec![
                Span::styled(" Download ", Style::new().fg(INK).add_modifier(Modifier::BOLD)),
                Span::styled(rec.name.clone(), Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)),
            ];
            if let Some(st) = &r.stats {
                if let Some((_, share)) = st.stability.iter().find(|(n, _)| n == &rec.name) {
                    l.push(Span::styled(
                        format!("   picked in {:.0}% of 1,000 reshuffles of the data", share * 100.0),
                        Style::new().fg(MUTED),
                    ));
                }
            }
            v.push(Line::from(l));
            v.push(Line::styled(format!(" {}", rec.reason), Style::new().fg(MUTED)));
        }
        f.render_widget(Paragraph::new(v).wrap(Wrap { trim: false }), verdict);

        let titles: Vec<Line> = TABS.iter().enumerate().map(|(i, t)| Line::from(format!(" {} {t} ", i + 1))).collect();
        f.render_widget(
            Tabs::new(titles)
                .select(self.tab)
                .style(Style::new().fg(MUTED))
                .highlight_style(Style::new().fg(Color::Black).bg(INK))
                .divider(""),
            tabs,
        );
        match self.tab {
            0 => self.tab_overview(f, body, r),
            1 => self.tab_breaks(f, body, r),
            2 => self.tab_answers(f, body, r),
            _ => self.tab_luck(f, body, r),
        }
    }

    fn tab_overview(&self, f: &mut Frame, area: Rect, r: &Results) {
        let refsz = r.reference.info.size_bytes.max(1) as f64;
        let ts = |n: &str| r.tasks.as_ref().and_then(|t| t.summary.get(n));
        let header = Row::new(["version", "size", "verdict", "drift", "same word", "answers", "speed"])
            .style(Style::new().fg(MUTED))
            .bottom_margin(1);
        let speed = |m: &crate::runner::ModelResult| {
            m.bench.as_ref().map(|b| format!("{:.0} tok/s", b.decode_tok_s)).unwrap_or_else(|| "–".into())
        };
        let mut rows = vec![Row::new(vec![
            Cell::from(Span::styled(r.reference.spec.name.clone(), Style::new().fg(INK))),
            Cell::from(human(r.reference.info.size_bytes)),
            Cell::from(Span::styled("original", Style::new().fg(MUTED))),
            Cell::from(drift_bar(r, None, INK)),
            Cell::from("100%"),
            Cell::from(ts(&r.reference.spec.name).map(|s| format!("{}/{}", s.correct, s.n)).unwrap_or_else(|| "–".into())),
            Cell::from(speed(&r.reference)),
        ])
        .style(Style::new().fg(INK2))];
        let rec = r.recommendation.as_ref().map(|x| x.name.as_str());
        for c in &r.candidates {
            let name = c.model.spec.name.clone();
            let is_rec = Some(name.as_str()) == rec;
            let answers = ts(&name)
                .map(|s| {
                    if s.regressions > 0 {
                        format!("{}/{}  broke {}", s.correct, s.n, s.regressions)
                    } else {
                        format!("{}/{}", s.correct, s.n)
                    }
                })
                .unwrap_or_else(|| "–".into());
            rows.push(
                Row::new(vec![
                    Cell::from(Span::styled(
                        format!("{name}{}", if is_rec { "  ◂ pick" } else { "" }),
                        Style::new().fg(if is_rec { ACCENT } else { INK }),
                    )),
                    Cell::from(format!("{}  {:.0}%", human(c.model.info.size_bytes), c.model.info.size_bytes as f64 / refsz * 100.0)),
                    Cell::from(Span::styled(tier_words(c.tier), Style::new().fg(tier_color(c.tier)))),
                    Cell::from(drift_bar(r, Some(c.summary.kl_mean), tier_color(c.tier))),
                    Cell::from(format!("{:.1}%", c.summary.top1_agreement * 100.0)),
                    Cell::from(answers),
                    Cell::from(speed(&c.model)),
                ])
                .style(Style::new().fg(INK2)),
            );
        }
        let widths = [
            Constraint::Length(16),
            Constraint::Length(14),
            Constraint::Length(22),
            Constraint::Length(16),
            Constraint::Length(10),
            Constraint::Length(16),
            Constraint::Length(11),
        ];
        let mut text_rows: Vec<Line> = Vec::new();
        text_rows.push(Line::raw(""));
        text_rows.push(Line::styled(
            "drift: how far each version moved from the original, from 'identical' (left) to 'damaged' (right)",
            Style::new().fg(MUTED),
        ));
        text_rows.push(Line::styled(
            "same word: how often it picks the same next word as the original · answers: test questions it got right",
            Style::new().fg(MUTED),
        ));
        let [t, legend] = Layout::vertical([Constraint::Length(rows.len() as u16 + 2), Constraint::Min(0)]).areas(panel("").inner(area));
        f.render_widget(panel(""), area);
        f.render_widget(Table::new(rows, widths).header(header).column_spacing(2), t);
        f.render_widget(Paragraph::new(text_rows).wrap(Wrap { trim: false }), legend);
    }

    fn tab_breaks(&self, f: &mut Frame, area: Rect, r: &Results) {
        let mut lines = Vec::new();
        let Some(first) = r.candidates.first() else { return };
        let max = r.candidates.iter().flat_map(|c| c.by_category.iter().map(|b| b.kl_mean)).fold(1e-9, f64::max);
        let worst: Vec<&str> = r
            .candidates
            .iter()
            .filter_map(|c| c.by_category.iter().max_by(|a, b| a.kl_mean.partial_cmp(&b.kl_mean).unwrap()).map(|b| b.category.as_str()))
            .collect();
        if !worst.is_empty() && worst.iter().all(|w| *w == worst[0]) {
            lines.push(Line::from(vec![
                Span::styled("Every version drifts most on ", Style::new().fg(INK2)),
                Span::styled(worst[0].to_string(), Style::new().fg(WARN).add_modifier(Modifier::BOLD)),
                Span::styled(". If that's what you use the model for, pick a bigger version.", Style::new().fg(INK2)),
            ]));
            lines.push(Line::raw(""));
        }
        let bar_w = (area.width as usize).saturating_sub(40).clamp(10, 60);
        let rec = r.recommendation.as_ref().map(|x| x.name.as_str());
        for cat in first.by_category.iter().map(|b| &b.category) {
            lines.push(Line::styled(cat.clone(), Style::new().fg(INK).add_modifier(Modifier::BOLD)));
            for c in &r.candidates {
                let Some(b) = c.by_category.iter().find(|b| &b.category == cat) else { continue };
                let n = ((b.kl_mean / max) * bar_w as f64).round().max(1.0) as usize;
                let color = if Some(c.model.spec.name.as_str()) == rec { ACCENT } else { INK2 };
                lines.push(Line::from(vec![
                    Span::styled(format!("  {:<14}", c.model.spec.name), Style::new().fg(INK2)),
                    Span::styled("━".repeat(n), Style::new().fg(color)),
                    Span::styled("─".repeat(bar_w - n.min(bar_w)), Style::new().fg(LINE)),
                    Span::styled(format!("  {:.4}", b.kl_mean), Style::new().fg(MUTED)),
                ]));
            }
            lines.push(Line::raw(""));
        }
        f.render_widget(Paragraph::new(lines).scroll((self.scroll, 0)).block(panel("Longer bar = more drift on that kind of text")), area);
    }

    fn tab_answers(&self, f: &mut Frame, area: Rect, r: &Results) {
        let Some(t) = &r.tasks else {
            f.render_widget(Paragraph::new("No tasks were run (press t on the setup screen to turn them on).").block(panel("")), area);
            return;
        };
        let Some(c) = r.candidates.get(self.cand.min(r.candidates.len().saturating_sub(1))) else { return };
        let name = &c.model.spec.name;
        let refr = &t.runs[&r.reference.spec.name];
        let runs = &t.runs[name];
        let mut lines = Vec::new();
        let changed: Vec<usize> = (0..t.items.len()).filter(|&i| runs[i].correct != refr[i].correct).collect();
        if changed.is_empty() {
            lines.push(Line::styled(
                format!("{name} got exactly the same questions right and wrong as the original."),
                Style::new().fg(INK2),
            ));
        }
        let verdict = |ok: bool| if ok { Span::styled("right", Style::new().fg(ACCENT)) } else { Span::styled("wrong", Style::new().fg(BAD)) };
        let short = |s: &str| {
            let one: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
            if one.chars().count() > 110 { format!("{}…", one.chars().take(110).collect::<String>()) } else { one }
        };
        for i in changed {
            let item = &t.items[i];
            lines.push(Line::styled(item.prompt.clone(), Style::new().fg(INK).add_modifier(Modifier::BOLD)));
            lines.push(Line::from(vec![
                Span::styled("  expected ", Style::new().fg(MUTED)),
                Span::styled(item.answer.clone(), Style::new().fg(INK)),
            ]));
            lines.push(Line::from(vec![
                Span::styled(format!("  {:<10}", r.reference.spec.name), Style::new().fg(MUTED)),
                verdict(refr[i].correct),
                Span::styled(format!("  {}", short(&refr[i].text)), Style::new().fg(INK2)),
            ]));
            lines.push(Line::from(vec![
                Span::styled(format!("  {:<10}", name), Style::new().fg(MUTED)),
                verdict(runs[i].correct),
                Span::styled(format!("  {}", short(&runs[i].text)), Style::new().fg(INK2)),
            ]));
            lines.push(Line::raw(""));
        }
        let s = &t.summary[name];
        let title = format!(
            "{name}: {}/{} right, broke {}, same text as the original {:.0}%   (c = next quant)",
            s.correct,
            s.n,
            s.regressions,
            s.identical as f64 / s.n.max(1) as f64 * 100.0
        );
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }).scroll((self.scroll, 0)).block(panel(&title)), area);
    }

    fn tab_luck(&self, f: &mut Frame, area: Rect, r: &Results) {
        let Some(st) = &r.stats else {
            f.render_widget(Paragraph::new("Statistics were turned off for this run.").block(panel("")), area);
            return;
        };
        let mut lines = vec![
            Line::styled(
                "We reshuffled the data 1,000 times and kept only the conclusions that survive.",
                Style::new().fg(INK2),
            ),
            Line::raw(""),
        ];
        let n_tasks = r.tasks.as_ref().map(|t| t.items.len()).unwrap_or(0);
        for p in &st.pairs {
            let (sure, text) = match (&p.kl_diff, &p.acc_diff) {
                (k, _) if !k.significant() => (false, format!("{} and {} are indistinguishable on this data. Take the smaller one.", p.smaller, p.larger)),
                (_, Some(a)) if a.significant() && a.est < 0.0 => (true, format!("{} is clearly worse than {}: it drifts more and gets more answers wrong.", p.smaller, p.larger)),
                (_, Some(a)) if a.significant() => (true, format!("{} drifts more than {}, yet answered more questions right on this set.", p.smaller, p.larger)),
                (_, Some(_)) => (false, format!("{} drifts more than {}, that part is certain. Whether its answers are worse can't be told from {n_tasks} questions.", p.smaller, p.larger)),
                (_, None) => (true, format!("{} drifts more than {}.", p.smaller, p.larger)),
            };
            lines.push(Line::from(vec![
                if sure {
                    Span::styled(" certain      ", Style::new().fg(Color::Black).bg(INK))
                } else {
                    Span::styled(" could be luck", Style::new().fg(MUTED))
                },
                Span::styled(format!("  {text}"), Style::new().fg(INK2)),
            ]));
        }
        lines.push(Line::raw(""));
        lines.push(Line::styled("How often each version would be our pick:", Style::new().fg(INK2)));
        let w = (area.width as usize).saturating_sub(30).clamp(10, 60);
        for (name, share) in &st.stability {
            let n = (share * w as f64).round() as usize;
            lines.push(Line::from(vec![
                Span::styled(format!("  {:<14}", name), Style::new().fg(INK2)),
                Span::styled("█".repeat(n), Style::new().fg(ACCENT)),
                Span::styled("░".repeat(w - n.min(w)), Style::new().fg(LINE)),
                Span::styled(format!("  {:.0}%", share * 100.0), Style::new().fg(MUTED)),
            ]));
        }
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }).scroll((self.scroll, 0)).block(panel("Is it just luck?")), area);
    }
}

fn panel(title: &str) -> Block<'static> {
    let b = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(LINE))
        .padding(Padding::horizontal(1));
    if title.is_empty() {
        b
    } else {
        b.title(Span::styled(format!(" {title} "), Style::new().fg(INK2)))
    }
}

/// The drift line in text: where a version sits between identical and damaged.
fn drift_bar(r: &Results, kl: Option<f64>, color: Color) -> Line<'static> {
    const W: usize = 14;
    let th = &r.thresholds;
    let kls: Vec<f64> = r.candidates.iter().map(|c| c.summary.kl_mean.max(1e-6)).collect();
    let lo = kls.iter().copied().fold(th.near_lossless / 4.0, f64::min);
    let hi = kls.iter().map(|k| k * 1.6).fold(th.noticeable * 4.0, f64::max);
    let pos = kl.map_or(0, |k| {
        let x = (k.max(lo).ln() - lo.ln()) / (hi.ln() - lo.ln());
        ((x.clamp(0.0, 1.0)) * (W - 1) as f64).round() as usize
    });
    Line::from(vec![
        Span::styled("━".repeat(pos), Style::new().fg(color)),
        Span::styled("●", Style::new().fg(color)),
        Span::styled("─".repeat(W - 1 - pos), Style::new().fg(LINE)),
    ])
}

fn draw_banner(f: &mut Frame, area: Rect) {
    let lines: Vec<Line> = std::iter::once(Line::raw(""))
        .chain(banner::rows().into_iter().map(|row| {
            let mut spans = vec![Span::raw("  ")];
            spans.extend(row.into_iter().map(|(s, role)| {
                let style = match role {
                    Role::Ink => Style::new().fg(INK),
                    Role::Accent => Style::new().fg(ACCENT),
                    Role::Muted => Style::new().fg(MUTED),
                };
                Span::styled(s, style)
            }));
            Line::from(spans)
        }))
        .collect();
    f.render_widget(Paragraph::new(Text::from(lines)), area);
}

fn event_loop(mut terminal: DefaultTerminal, mut app: App) -> Result<()> {
    // For demos and recordings: start the run without pressing Enter.
    if matches!(app.screen, Screen::Setup) && std::env::var_os("BITROT_TUI_AUTOSTART").is_some() {
        app.start();
    }
    while !app.quit {
        app.pump();
        terminal.draw(|f| app.draw(f))?;
        if event::poll(Duration::from_millis(60))? {
            if let Event::Key(k) = event::read()? {
                if k.kind == KeyEventKind::Press {
                    app.key(k.code, k.modifiers);
                }
            }
        }
    }
    Ok(())
}

fn run_app(app: App) -> Result<()> {
    // Log lines are drawn by the TUI itself; ANSI codes would show up raw.
    crate::term::disable();
    let terminal = ratatui::init();
    let res = event_loop(terminal, app);
    ratatui::restore();
    res
}

/// The interactive flow: choose models, run, browse results.
pub fn run(models_dir: PathBuf, settings: Settings, out: PathBuf) -> Result<()> {
    run_app(App::new(models_dir, settings, out))
}

/// Open the results screen for a saved run.
pub fn view(results_path: &Path) -> Result<()> {
    let data: Results = serde_json::from_slice(&std::fs::read(results_path)?)
        .map_err(|e| anyhow!("{} is not a bitrot results file: {e}", results_path.display()))?;
    let html = results_path.with_file_name("report.html");
    let mut app = App::new(PathBuf::from("models"), Settings::default(), PathBuf::from("bitrot-out"));
    app.report = html.exists().then_some(html);
    app.results = Some(data);
    app.screen = Screen::Results;
    run_app(app)
}
