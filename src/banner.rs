//! The "bitrot" banner: the logo mark (a grid of bits whose corner decays)
//! next to the name drawn in a tiny pixel font with half-block characters.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Ink,
    Accent,
    Muted,
}

pub type Row = Vec<(String, Role)>;

/// 5-pixel-tall lowercase glyphs ('#' = on). Rows 0-1 are the ascender
/// zone, rows 2-4 the x-height.
fn glyph(c: char) -> &'static [&'static str] {
    match c {
        'b' => &["#...", "#...", "###.", "#..#", "###."],
        'i' => &["#", ".", "#", "#", "#"],
        't' => &["...", ".#.", "###", ".#.", ".##"],
        'r' => &["....", "....", "#.##", "##..", "#..."],
        'o' => &["....", "....", ".##.", "#..#", ".##."],
        _ => &["", "", "", "", ""],
    }
}

/// Render `word` as three rows of half-block characters.
pub fn pixel_word(word: &str) -> [String; 3] {
    let mut rows = [String::new(), String::new(), String::new()];
    for (n, c) in word.chars().enumerate() {
        let g = glyph(c);
        let w = g[0].len();
        for (r, out) in rows.iter_mut().enumerate() {
            if n > 0 {
                out.push(' ');
            }
            for x in 0..w {
                let px = |y: usize| g.get(y).and_then(|row| row.as_bytes().get(x)).is_some_and(|&b| b == b'#');
                out.push(match (px(2 * r), px(2 * r + 1)) {
                    (true, true) => '█',
                    (true, false) => '▀',
                    (false, true) => '▄',
                    (false, false) => ' ',
                });
            }
        }
    }
    rows
}

/// The full banner: mark, name, and a tagline row.
pub fn rows() -> Vec<Row> {
    let word = pixel_word("bitrot");
    let mark: [Vec<(&str, Role)>; 3] = [
        vec![("■ ■ ■    ", Role::Ink)],
        vec![("■ ■ ▪ ", Role::Ink), ("·  ", Role::Accent)],
        vec![("■ ▪  ", Role::Ink), ("▪ ", Role::Accent), ("· ", Role::Accent)],
    ];
    let mut out: Vec<Row> = Vec::new();
    for (m, w) in mark.into_iter().zip(word) {
        let mut row: Row = m.into_iter().map(|(s, r)| (s.to_string(), r)).collect();
        row.push((format!("  {w}"), Role::Ink));
        out.push(row);
    }
    out.push(vec![(
        format!("how much did quantization rot your llm?  v{}", env!("CARGO_PKG_VERSION")),
        Role::Muted,
    )]);
    out
}

/// Plain text version (for --help and non-terminals).
pub fn plain() -> String {
    rows()
        .into_iter()
        .map(|row| row.into_iter().map(|(s, _)| s).collect::<String>().trim_end().to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Print the banner with colors when stdout is a terminal.
pub fn print() {
    use crate::term::Paint;
    println!();
    for row in rows() {
        let line: String = row
            .into_iter()
            .map(|(s, role)| match role {
                Role::Ink => s.bold(),
                Role::Accent => s.cyan(),
                Role::Muted => s.dimmed(),
            })
            .collect();
        println!("  {line}");
    }
    println!();
}

#[cfg(test)]
mod tests {
    #[test]
    fn word_rows_have_equal_width() {
        let rows = super::pixel_word("bitrot");
        let w: Vec<usize> = rows.iter().map(|r| r.chars().count()).collect();
        assert!(w.iter().all(|&x| x == w[0]), "{w:?}");
        assert!(rows[1].contains('█'));
    }
}
