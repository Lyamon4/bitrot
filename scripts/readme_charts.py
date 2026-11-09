"""Render the README charts from a real bitrot run.

    python scripts/readme_charts.py bitrot-out/results.json assets/charts

Pure standard library: writes self-contained SVG files in the report's
visual language (dark card, one ice-blue accent for the recommended quant).
"""

import json
import math
import os
import sys

BG = "#0d0d0d"
INK = "#f2f2f0"
INK2 = "#c9c9c5"
MUTED = "#85857f"
LINE = "#262626"
ACCENT = "#8ec5ff"
WARN = "#d9a441"
BAD = "#e5484d"
NEUTRAL = ["#c9c9c5", "#7a7a75", "#4a4a47", "#9b9bff", "#4fd1c5"]
FONT = "Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
CATS = {"english": "English text", "russian": "Russian text", "code": "code", "math": "math", "structured": "tables & JSON"}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, w, h, title, subtitle):
        self.w, self.h = w, h
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">',
            f'<rect width="{w}" height="{h}" rx="14" fill="{BG}"/>',
            f'<text x="28" y="40" fill="{INK}" font-size="19" font-weight="500">{esc(title)}</text>',
            f'<text x="28" y="64" fill="{MUTED}" font-size="13.5">{esc(subtitle)}</text>',
        ]

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, fill=MUTED, size=12, anchor="start", mono=False, weight=400):
        fam = MONO if mono else FONT
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" font-family="{fam}" font-weight="{weight}">{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke=LINE, width=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{d}/>')

    def rect(self, x, y, w, h, fill, rx=3):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"/>')

    def save(self, path):
        self.add("</svg>")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.parts))


def colors(d):
    """The recommended quant gets the accent, the rest neutral steps."""
    rec = (d.get("recommendation") or {}).get("name")
    out, k = {}, 0
    for c in d["candidates"]:
        n = c["model"]["spec"]["name"]
        if n == rec:
            out[n] = ACCENT
        else:
            out[n] = NEUTRAL[k % len(NEUTRAL)]
            k += 1
    return out


def mb(b):
    return b / 1e6


def quality_vs_size(d, path):
    W, H = 860, 400
    s = Svg(W, H, "Quality vs size", "Lower is closer to the original. The line joins the quants nothing else beats on both axes.")
    L, R, T, B = 80, 40, 96, 60
    cand = d["candidates"]
    ref = d["reference"]
    xmax = mb(ref["info"]["size_bytes"]) * 1.12
    kls = [max(c["summary"]["kl_mean"], 1e-5) for c in cand]
    y0 = 10 ** math.floor(math.log10(min(kls)))
    y1 = 10 ** math.ceil(math.log10(max(kls)))
    X = lambda v: L + v / xmax * (W - L - R)
    Y = lambda v: T + (H - T - B) - (math.log10(v) - math.log10(y0)) / (math.log10(y1) - math.log10(y0)) * (H - T - B)
    th = d["thresholds"]
    # tier bands
    bands = [(y0, th["near_lossless"], INK, "near-lossless"), (th["near_lossless"], th["excellent"], ACCENT, "excellent"),
             (th["excellent"], th["noticeable"], WARN, "noticeable"), (th["noticeable"], y1, BAD, "degraded")]
    for a, b, col, label in bands:
        a, b = max(a, y0), min(b, y1)
        if b <= a:
            continue
        s.rect(L, Y(b), W - L - R, Y(a) - Y(b), col, rx=0)
        s.parts[-1] = s.parts[-1].replace("/>", ' opacity="0.06"/>')
        s.text(W - R - 6, Y(b) + 14, label, fill=col, size=11, anchor="end")
    e = math.log10(y0)
    while e <= math.log10(y1) + 1e-9:
        v = 10 ** e
        s.line(L, Y(v), W - R, Y(v), dash="2 4")
        s.text(L - 10, Y(v) + 4, f"{v:g}", anchor="end", mono=True)
        e += 1
    step = 50 if xmax < 400 else 100 if xmax < 1000 else 1000
    v = 0
    while v <= xmax:
        s.text(X(v), H - B + 22, f"{v:g} MB", anchor="middle", mono=True)
        v += step
    s.text(L, T - 10, "mean KL (log scale)", size=11)
    rx = X(mb(ref["info"]["size_bytes"]))
    s.line(rx, T, rx, H - B, stroke=MUTED, dash="4 4")
    s.text(rx - 6, T + 14, f'{ref["spec"]["name"]} (original)', anchor="end", size=11)
    par = sorted([c for c in cand if c["pareto"]], key=lambda c: c["model"]["info"]["size_bytes"])
    if len(par) > 1:
        pts = " ".join(f'{X(mb(c["model"]["info"]["size_bytes"])):.1f},{Y(max(c["summary"]["kl_mean"], 1e-5)):.1f}' for c in par)
        s.add(f'<polyline points="{pts}" fill="none" stroke="{INK2}" stroke-opacity="0.45" stroke-width="1.5"/>')
    col = colors(d)
    for c in cand:
        n = c["model"]["spec"]["name"]
        cx, cy = X(mb(c["model"]["info"]["size_bytes"])), Y(max(c["summary"]["kl_mean"], 1e-5))
        s.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{col[n]}" stroke="{BG}" stroke-width="2"/>')
        s.text(cx + 12, cy + 4, f'{n}  KL {c["summary"]["kl_mean"]:.4f}', fill=INK, size=12, mono=True)
    s.save(path)


def by_category(d, path):
    cand = d["candidates"]
    cats = [b["category"] for b in cand[0]["by_category"]]
    cats.sort(key=lambda k: -sum(next(b["kl_mean"] for b in c["by_category"] if b["category"] == k) for c in cand))
    worst = cats[0]
    bar_h, gap, group_gap = 12, 4, 18
    group_h = len(cand) * (bar_h + gap) + group_gap
    W = 860
    H = 100 + len(cats) * group_h + 20
    ratios = []
    for c in cand:
        kl = {b["category"]: b["kl_mean"] for b in c["by_category"]}
        rest = [v for k, v in kl.items() if k != worst]
        ratios.append(kl[worst] / (sum(rest) / len(rest)))
    same = all(max(c["by_category"], key=lambda b: b["kl_mean"])["category"] == worst for c in cand)
    sub = (f"Mean KL per kind of text. Every quant drifts most on {CATS.get(worst, worst)}: "
           f"{min(ratios):.1f}-{max(ratios):.1f}x the average of the other kinds.") if same else "Mean KL per kind of text."
    s = Svg(W, H, f"Where quantization breaks first: {CATS.get(worst, worst)}", sub)
    L, R, T = 150, 90, 92
    mx = max(b["kl_mean"] for c in cand for b in c["by_category"])
    col = colors(d)
    for gi, k in enumerate(cats):
        y = T + gi * group_h
        s.text(L - 16, y + group_h / 2 - 6, CATS.get(k, k), fill=INK if k == worst else INK2, size=13, anchor="end", weight=500 if k == worst else 400)
        for i, c in enumerate(cand):
            b = next(b for b in c["by_category"] if b["category"] == k)
            by = y + i * (bar_h + gap)
            s.rect(L, by, W - L - R, bar_h, "#161616")
            s.rect(L, by, (W - L - R) * b["kl_mean"] / mx, bar_h, col[c["model"]["spec"]["name"]])
            s.text(L + (W - L - R) * b["kl_mean"] / mx + 8, by + bar_h - 2, f'{b["kl_mean"]:.4f}', size=11, mono=True)
    # legend
    x = L
    for c in cand:
        n = c["model"]["spec"]["name"]
        s.rect(x, H - 26, 10, 10, col[n], rx=2)
        s.text(x + 16, H - 17, n, fill=INK2, size=12, mono=True)
        x += 110
    s.save(path)


def answers(d, path):
    t = d.get("tasks")
    if not t:
        return False
    cand = d["candidates"]
    refn = d["reference"]["spec"]["name"]
    rs = t["summary"][refn]
    W = 860
    row = 46
    H = 100 + len(cand) * row + 44
    s = Svg(W, H, "Does it still answer the same?", f'{len(t["items"])} questions with known answers, greedy decoding from identical prompt tokens. The original got {rs["correct"]} right.')
    L, R, T = 130, 200, 100
    segs = [("identical", INK), ("reworded", "#3a3a3a"), ("fixed", ACCENT), ("broken", BAD)]
    for i, c in enumerate(cand):
        n = c["model"]["spec"]["name"]
        sm = t["summary"][n]
        vals = {"identical": sm["identical"], "fixed": sm["fixes"], "broken": sm["regressions"]}
        vals["reworded"] = sm["n"] - vals["identical"] - vals["fixed"] - vals["broken"]
        y = T + i * row
        s.text(L - 16, y + 17, n, fill=INK, size=13, anchor="end", mono=True)
        x = L
        for key, colr in segs:
            w = (W - L - R) * vals[key] / sm["n"]
            if w > 0:
                s.rect(x, y + 4, w - 1, 18, colr, rx=2)
                if w > 34:
                    s.text(x + w / 2, y + 17, str(vals[key]), fill="#000" if colr in (INK, ACCENT) else INK, size=11, anchor="middle", mono=True)
            x += w
        s.text(W - R + 14, y + 12, f'{sm["correct"]}/{sm["n"]} correct', fill=INK2, size=12, mono=True)
        s.text(W - R + 14, y + 27, f'broke {sm["regressions"]} · same text {sm["identical"] / sm["n"]:.0%}', fill=BAD if sm["regressions"] else MUTED, size=11)
    y = T + len(cand) * row + 10
    x = L
    for key, colr in segs:
        label = {"identical": "word-for-word the same", "reworded": "different text, same verdict", "fixed": "now right", "broken": "was right, now wrong"}[key]
        s.rect(x, y, 10, 10, colr, rx=2)
        s.text(x + 16, y + 9, label, size=11.5, fill=INK2)
        x += 160
    s.save(path)
    return True


def calibration(d, path):
    cal = d.get("calibration")
    if not cal:
        return False
    W, H = 860, 400
    s = Svg(W, H, "Calibrated tiers: how much drift changes an answer", f'Each dot is one answer by one quant ({cal["n"]} in total). The curve is a logistic fit; the tier boundaries sit where it crosses 2%, 5% and 15%.')
    L, R, T, B = 70, 30, 96, 62
    cand = d["candidates"]
    kls = [p["kl"] for p in cal["points"]] + list(cal["kl_at"]) + [max(c["summary"]["kl_mean"], 1e-6) for c in cand]
    lo = 10 ** math.floor(math.log10(max(min(kls), 1e-6)))
    hi = 10 ** math.ceil(math.log10(max(kls)))
    X = lambda v: L + (math.log10(max(v, lo)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (W - L - R)
    Y = lambda p: T + (H - T - B) * (1 - p)
    P = lambda k: 1 / (1 + math.exp(-(cal["a"] + cal["b"] * math.log(max(k, 1e-6)))))
    for p in (0, 0.25, 0.5, 0.75, 1):
        s.line(L, Y(p), W - R, Y(p), dash="2 4")
        s.text(L - 8, Y(p) + 4, f"{p:.0%}", anchor="end", mono=True)
    e = math.log10(lo)
    while e <= math.log10(hi) + 1e-9:
        s.text(X(10 ** e), H - B + 22, f"{10 ** e:g}", anchor="middle", mono=True)
        e += 1
    s.text(L + (W - L - R) / 2, H - 14, "KL on the original's answer (log)", anchor="middle", size=11)
    for k, lvl in zip(cal["kl_at"], cal["levels"]):
        s.line(X(k), T, X(k), H - B, stroke=MUTED, dash="3 4")
        s.text(X(k) + 4, H - B - 8, f"{lvl:.0%}", size=11)
    col = colors(d)
    for j, p in enumerate(cal["points"]):
        jit = ((j * 9301 + 49297) % 233280) / 233280 * 0.06 - 0.03
        y = 0.96 + jit if p["changed"] else 0.04 + jit
        s.add(f'<circle cx="{X(p["kl"]):.1f}" cy="{Y(y):.1f}" r="3" fill="{col.get(p["candidate"], INK2)}" opacity="0.7"/>')
    pts = []
    for i in range(201):
        k = 10 ** (math.log10(lo) + (math.log10(hi) - math.log10(lo)) * i / 200)
        pts.append(f"{X(k):.1f},{Y(P(k)):.1f}")
    s.add(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{INK}" stroke-width="2"/>')
    for c in cand:
        n = c["model"]["spec"]["name"]
        k = max(c["summary"]["kl_mean"], 1e-6)
        s.add(f'<circle cx="{X(k):.1f}" cy="{Y(P(k)):.1f}" r="7" fill="{col[n]}" stroke="{BG}" stroke-width="2"/>')
        s.text(X(k) + 10, Y(P(k)) - 8, f"{n} ~{P(k):.0%}", fill=INK, size=12, mono=True)
    s.text(W - R - 4, Y(0.96) - 10, "answer changed", size=11, anchor="end")
    s.text(W - R - 4, Y(0.04) + 20, "answer unchanged", size=11, anchor="end")
    s.save(path)
    return True


def luck(d, path):
    st = d.get("stats")
    if not st or not st["pairs"] or not st["pairs"][0].get("acc_diff"):
        return False
    pairs = st["pairs"]
    W = 860
    row = 44
    H = 120 + len(pairs) * row + 60
    s = Svg(W, H, "Is it just luck?", f'95% bootstrap intervals from {st["replicates"]} paired resamples. Difference in correct answers, smaller quant minus larger.')
    L, R, T = 220, 130, 104
    lo = min(min(p["acc_diff"]["lo"] for p in pairs), -0.05)
    hi = max(max(p["acc_diff"]["hi"] for p in pairs), 0.05)
    span = max(-lo, hi) * 1.15
    X = lambda v: L + (v + span) / (2 * span) * (W - L - R)
    s.line(X(0), T - 14, X(0), T + len(pairs) * row, stroke=MUTED, dash="4 4")
    s.text(X(0), T - 20, "no difference", anchor="middle", size=11)
    for i, p in enumerate(pairs):
        a = p["acc_diff"]
        y = T + i * row + 14
        sig = a["lo"] > 0 or a["hi"] < 0
        colr = BAD if sig and a["est"] < 0 else (ACCENT if sig else INK2)
        s.text(L - 16, y + 4, f'{p["smaller"]} vs {p["larger"]}', fill=INK, size=13, anchor="end", mono=True)
        s.line(X(a["lo"]), y, X(a["hi"]), y, stroke=colr, width=3)
        s.add(f'<circle cx="{X(a["est"]):.1f}" cy="{y:.1f}" r="6" fill="{colr}"/>')
        s.text(W - R + 12, y - 2, f'{a["est"] * 100:+.0f}% [{a["lo"] * 100:+.0f}, {a["hi"] * 100:+.0f}]', fill=INK2, size=11.5, mono=True)
        s.text(W - R + 12, y + 13, "real difference" if sig else "could be luck", fill=colr if sig else MUTED, size=11)
    v = -math.floor(span * 10) / 10
    while v <= span + 1e-9:
        s.text(X(v), T + len(pairs) * row + 14, f"{v * 100:+.0f}%", anchor="middle", mono=True)
        v += 0.1
    stab = st["stability"]
    y = H - 34
    s.text(28, y + 13, "recommendation stability", size=12, fill=INK2)
    x = 210
    total = W - x - 28
    col = colors(d)
    for n, f in stab:
        w = total * f
        s.rect(x, y, w - 2, 20, col.get(n, "#3a3a3a"), rx=3)
        if w > 70:
            s.text(x + 8, y + 14, f"{n} {f:.0%}", fill="#000", size=11.5, mono=True)
        x += w
    s.save(path)
    return True


def pipeline(path):
    W, H = 860, 300
    s = Svg(W, H, "How bitrot works", "The same tokens go through every model, so each position can be compared exactly.")
    boxes = [
        (28, 110, 150, "your prompts", "text + test questions", INK2),
        (218, 90, 170, "original (F16)", "teacher forcing: top-64 next-token probabilities", INK),
        (218, 190, 170, "each quant", "probabilities on the same 64 tokens", INK),
        (430, 90, 190, "token drift", "KL · top-1 agreement · flips", ACCENT),
        (430, 190, 190, "real answers", "greedy answers, graded · broken / fixed", ACCENT),
        (662, 140, 170, "verdict", "calibrated tiers · bootstrap · pick", INK),
    ]
    for x, y, w, title, sub, colr in boxes:
        s.add(f'<rect x="{x}" y="{y}" width="{w}" height="64" rx="10" fill="#151515" stroke="{LINE}"/>')
        s.text(x + 14, y + 26, title, fill=colr, size=14, weight=500)
        lines, cur = [], ""
        for word in sub.split():
            if len(cur) + len(word) > (w - 24) // 6.6:
                lines.append(cur)
                cur = word
            else:
                cur = (cur + " " + word).strip()
        lines.append(cur)
        for k, ln in enumerate(lines[:2]):
            s.text(x + 14, y + 44 + k * 14, ln, size=11)

    def arrow(x1, y1, x2, y2):
        s.add(f'<path d="M{x1},{y1} C{(x1 + x2) / 2},{y1} {(x1 + x2) / 2},{y2} {x2},{y2}" fill="none" stroke="{MUTED}" stroke-width="1.5"/>')
        s.add(f'<path d="M{x2 - 7},{y2 - 4} L{x2},{y2} L{x2 - 7},{y2 + 4}" fill="none" stroke="{MUTED}" stroke-width="1.5"/>')

    arrow(178, 142, 218, 122)
    arrow(178, 142, 218, 222)
    arrow(388, 122, 430, 122)
    arrow(388, 222, 430, 122)
    arrow(388, 222, 430, 222)
    arrow(620, 122, 662, 172)
    arrow(620, 222, 662, 172)
    s.text(28, 280, "Coarsened KL over the reference's top-64 tokens + a tail bucket is a lower bound on the full-vocabulary KL.", size=11.5)
    s.save(path)


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    d = json.load(open(src, encoding="utf-8"))
    pipeline(os.path.join(out, "pipeline.svg"))
    quality_vs_size(d, os.path.join(out, "quality-vs-size.svg"))
    by_category(d, os.path.join(out, "by-category.svg"))
    answers(d, os.path.join(out, "answers.svg"))
    calibration(d, os.path.join(out, "calibration.svg"))
    luck(d, os.path.join(out, "luck.svg"))
    print("charts written to", out)


if __name__ == "__main__":
    main()
