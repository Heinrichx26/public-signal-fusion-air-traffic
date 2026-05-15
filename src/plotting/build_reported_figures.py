from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject


ROOT = Path(__file__).resolve().parents[2]


def _first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


RESULTS = _first_existing([
    ROOT / "results" / "experiments",
    ROOT.parent.parent.parent / "results" / "experiments",
])
MAIN = RESULTS / "atcscc_full_year_windows"
SUPP = RESULTS / "supplemental_validation"
OUT = ROOT / "results" / "figures"


BLUE = colors.HexColor("#1f77b4")
ORANGE = colors.HexColor("#d95f02")
GREEN = colors.HexColor("#2ca25f")
GRAY = colors.HexColor("#5f6b7a")
LIGHT = colors.HexColor("#e7ebf0")
TEXT = colors.HexColor("#1b1f24")
FONT = "DejaVuSans"
FONT_BOLD = "DejaVuSans-Bold"
FONT_DIR = _first_existing([
    ROOT / "assets" / "fonts" / "dejavu",
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/Library/Fonts"),
    Path("C:/Windows/Fonts"),
])

pdfmetrics.registerFont(TTFont(FONT, str(_first_existing([
    FONT_DIR / "DejaVuSans.ttf",
    FONT_DIR / "Arial.ttf",
    FONT_DIR / "arial.ttf",
]))))
pdfmetrics.registerFont(TTFont(FONT_BOLD, str(_first_existing([
    FONT_DIR / "DejaVuSans-Bold.ttf",
    FONT_DIR / "Arial Bold.ttf",
    FONT_DIR / "arialbd.ttf",
]))))


def _canvas(path: Path, width: float = 7.1 * inch, height: float = 4.35 * inch) -> canvas.Canvas:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=(width, height))
    c.setTitle(path.stem)
    return c


def _strip_unused_reportlab_helvetica(path: Path) -> None:
    reader = PdfReader(str(path))
    writer = PdfWriter()
    changed = False
    seed = b"1 0 0 1 0 0 cm  BT /F1 12 Tf 14.4 TL ET\n"
    for page in reader.pages:
        data = page.get_contents().get_data()
        if seed in data:
            data = data.replace(seed, b"1 0 0 1 0 0 cm\n")
            stream = DecodedStreamObject()
            stream.set_data(data)
            page[NameObject("/Contents")] = stream
            changed = True
        resources = page.get("/Resources")
        fonts = resources.get_object().get("/Font") if resources else None
        if fonts:
            fonts = fonts.get_object()
            font_obj = fonts.get(NameObject("/F1"))
            if font_obj and font_obj.get_object().get("/BaseFont") == "/Helvetica" and b"/F1" not in data:
                del fonts[NameObject("/F1")]
                changed = True
        writer.add_page(page)
    if changed:
        tmp = path.with_suffix(".tmp.pdf")
        with tmp.open("wb") as f:
            writer.write(f)
        tmp.replace(path)


def _save(c: canvas.Canvas) -> None:
    path = Path(c._filename)
    c.save()
    _strip_unused_reportlab_helvetica(path)


def _title(c: canvas.Canvas, title: str, subtitle: str | None = None) -> None:
    w, h = c._pagesize
    c.setFillColor(TEXT)
    c.setFont(FONT_BOLD, 12)
    c.drawString(0.35 * inch, h - 0.38 * inch, title)
    if subtitle:
        c.setFillColor(GRAY)
        c.setFont(FONT, 8.5)
        c.drawString(0.35 * inch, h - 0.57 * inch, subtitle)


def _axis(c: canvas.Canvas, x0: float, y0: float, width: float, height: float, ymax: float, label_color=GRAY) -> None:
    c.setStrokeColor(LIGHT)
    c.setLineWidth(0.5)
    c.setFont(FONT, 7)
    c.setFillColor(label_color)
    for i in range(5):
        y = y0 + height * i / 4
        value = ymax * i / 4
        c.line(x0, y, x0 + width, y)
        c.drawRightString(x0 - 6, y - 2, f"{value:.2f}")
    c.setStrokeColor(colors.black)
    c.line(x0, y0, x0, y0 + height)
    c.line(x0, y0, x0 + width, y0)


def _line(c: canvas.Canvas, xs: list[float], ys: list[float], color, width: float = 1.7) -> None:
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    for i in range(len(xs) - 1):
        c.line(xs[i], ys[i], xs[i + 1], ys[i + 1])
    for x, y in zip(xs, ys):
        c.circle(x, y, 2.4, stroke=0, fill=1)


def _legend(c: canvas.Canvas, items: list[tuple[str, colors.Color]], x: float, y: float, text_color=GRAY) -> None:
    c.setFont(FONT, 7.5)
    for label, color in items:
        c.setFillColor(color)
        c.rect(x, y - 4, 10, 6, stroke=0, fill=1)
        c.setFillColor(text_color)
        c.drawString(x + 14, y - 4, label)
        x += 108


def _legend_centered(c: canvas.Canvas, items: list[tuple[str, colors.Color]], y: float, text_color=GRAY) -> None:
    total = sum(14 + pdfmetrics.stringWidth(label, FONT, 7.5) + 34 for label, _ in items) - 34
    x = (c._pagesize[0] - total) / 2
    c.setFont(FONT, 7.5)
    for label, color in items:
        c.setFillColor(color)
        c.rect(x, y - 4, 10, 6, stroke=0, fill=1)
        c.setFillColor(text_color)
        c.drawString(x + 14, y - 4, label)
        x += 14 + pdfmetrics.stringWidth(label, FONT, 7.5) + 34


def _point_label(
    c: canvas.Canvas,
    x: float,
    y: float,
    label: str,
    color,
    dx: float = 0,
    dy: float = 10,
) -> None:
    font_size = 6.8
    text_width = pdfmetrics.stringWidth(label, FONT_BOLD, font_size)
    cx = x + dx
    by = y + dy
    c.setFillColor(colors.white)
    c.roundRect(cx - text_width / 2 - 3, by - 2, text_width + 6, 10, 2, stroke=0, fill=1)
    c.setStrokeColor(color)
    c.setLineWidth(0.4)
    c.roundRect(cx - text_width / 2 - 3, by - 2, text_width + 6, 10, 2, stroke=1, fill=0)
    c.setFillColor(color)
    c.setFont(FONT_BOLD, font_size)
    c.drawCentredString(cx, by + 0.8, label)


def fig1_multi_source_fusion() -> None:
    c = _canvas(OUT / "fig1_multi_source_fusion.pdf", 7.1 * inch, 3.2 * inch)
    w, h = c._pagesize

    def section(x: float, y: float, width: float, height: float, label: str) -> None:
        c.setStrokeColor(colors.HexColor("#CBD5E1"))
        c.setLineWidth(0.7)
        c.setDash(4, 3)
        c.roundRect(x, y, width, height, 6, stroke=1, fill=0)
        c.setDash()
        c.setFillColor(colors.white)
        c.rect(x + 6, y + height - 6, 70, 11, stroke=0, fill=1)
        c.setFillColor(GRAY)
        c.setFont(FONT_BOLD, 6.8)
        c.drawString(x + 8, y + height - 3, label)

    def arrow(x1: float, y1: float, x2: float, y2: float, width: float = 1.25) -> None:
        c.setStrokeColor(GRAY)
        c.setFillColor(GRAY)
        c.setLineWidth(width)
        c.line(x1, y1, x2, y2)
        angle = math.atan2(y2 - y1, x2 - x1)
        head = 7.0
        spread = 0.42
        p1 = (x2, y2)
        p2 = (x2 - head * math.cos(angle - spread), y2 - head * math.sin(angle - spread))
        p3 = (x2 - head * math.cos(angle + spread), y2 - head * math.sin(angle + spread))
        path = c.beginPath()
        path.moveTo(*p1)
        path.lineTo(*p2)
        path.lineTo(*p3)
        path.close()
        c.drawPath(path, stroke=0, fill=1)

    section(0.22 * inch, 2.15 * inch, 6.66 * inch, 0.90 * inch, "Input signals")
    nodes = [
        ("Weather", "Local capacity", 0.40 * inch, 2.34 * inch, BLUE),
        ("Advisory", "GDP/GS action", 2.02 * inch, 2.34 * inch, ORANGE),
        ("Demand", "Schedule pressure", 3.64 * inch, 2.34 * inch, colors.HexColor("#7B5EA7")),
        ("Outcome", "Delay closure", 5.26 * inch, 2.34 * inch, GREEN),
    ]
    for title, sub, x, y, color in nodes:
        c.setFillColor(colors.white)
        c.setStrokeColor(color)
        c.setLineWidth(1.4)
        c.roundRect(x, y, 1.24 * inch, 0.56 * inch, 5, stroke=1, fill=1)
        c.setFillColor(color)
        c.setFont(FONT_BOLD, 13.2)
        c.drawCentredString(x + 0.62 * inch, y + 0.33 * inch, title)
        c.setFillColor(GRAY)
        c.setFont(FONT, 8.2)
        c.drawCentredString(x + 0.62 * inch, y + 0.16 * inch, sub)
    for x in [1.71 * inch, 3.33 * inch, 4.95 * inch]:
        arrow(x, 2.62 * inch, x + 0.24 * inch, 2.62 * inch, 1.25)
    arrow(3.55 * inch, 2.15 * inch, 3.55 * inch, 1.96 * inch, 1.1)
    c.setFillColor(colors.HexColor("#F8FAFC"))
    c.setStrokeColor(TEXT)
    c.setLineWidth(1.2)
    c.roundRect(0.56 * inch, 1.28 * inch, 5.98 * inch, 0.66 * inch, 7, stroke=1, fill=1)
    c.setFillColor(TEXT)
    c.setFont(FONT_BOLD, 16.5)
    c.drawCentredString(w / 2, 1.60 * inch, "Residual operational state")
    c.setFont(FONT, 9)
    c.setFillColor(GRAY)
    c.drawCentredString(w / 2, 1.41 * inch, "mild weather | advisory | demand-adjusted residual | realized outcome")
    arrow(3.55 * inch, 1.27 * inch, 3.55 * inch, 1.06 * inch, 1.1)
    section(0.72 * inch, 0.40 * inch, 5.66 * inch, 0.62 * inch, "Validation evidence")
    evidence = [
        ("active", BLUE),
        ("post 3 h", ORANGE),
        ("matched", colors.HexColor("#7B5EA7")),
        ("external", GREEN),
    ]
    x = 1.00 * inch
    for label, color in evidence:
        c.setFillColor(colors.white)
        c.setStrokeColor(color)
        c.setLineWidth(1.05)
        c.roundRect(x, 0.58 * inch, 1.10 * inch, 0.26 * inch, 5, stroke=1, fill=1)
        c.setFillColor(color)
        c.setFont(FONT_BOLD, 8.9)
        c.drawCentredString(x + 0.55 * inch, 0.67 * inch, label)
        x += 1.34 * inch
    c.showPage()
    _save(c)


def fig2_temporal_fusion() -> None:
    c = _canvas(OUT / "fig2_temporal_fusion.pdf", 7.1 * inch, 3.9 * inch)
    x0, y = 0.55 * inch, 2.42 * inch
    timeline_w = 5.9 * inch
    c.setStrokeColor(colors.black)
    c.setLineWidth(1)
    c.line(x0, y, x0 + timeline_w, y)
    for i, label in enumerate(["t-1", "t", "t+1", "t+2", "t+3"]):
        x = x0 + timeline_w * i / 4
        c.line(x, y - 5, x, y + 5)
        c.setFillColor(GRAY)
        c.setFont(FONT, 7.5)
        c.drawCentredString(x, y - 18, label)
    c.setFillColor(ORANGE)
    c.rect(x0 + 1.45 * inch, y + 0.16 * inch, 1.5 * inch, 0.18 * inch, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#f7c17b"))
    c.rect(x0 + 2.95 * inch, y + 0.16 * inch, 2.95 * inch, 0.18 * inch, stroke=0, fill=1)
    c.setFillColor(TEXT)
    c.setFont(FONT_BOLD, 8.7)
    c.drawString(x0 + 1.45 * inch, y + 0.46 * inch, "Active GDP/GS overlap")
    c.drawString(x0 + 3.05 * inch, y + 0.46 * inch, "Three-hour post-advisory window")
    c.setFillColor(colors.HexColor("#F3F4F6"))
    c.rect(x0 + 0.55 * inch, y + 0.16 * inch, 0.90 * inch, 0.18 * inch, stroke=0, fill=1)
    c.setFillColor(TEXT)
    c.setFont(FONT_BOLD, 8.2)
    c.drawString(x0 + 0.55 * inch, y + 0.46 * inch, "Lead horizon")
    rows = [
        ("BTS outcomes", "scheduled arrivals and realized delay/cancellation", GREEN),
        ("IEM ASOS weather", "visibility, wind, ceiling, temperature", BLUE),
        ("FAA ATCSCC", "advisory interval overlap by airport-hour", ORANGE),
    ]
    for idx, (label, sub, color) in enumerate(rows):
        yy = 1.56 * inch - idx * 0.47 * inch
        c.setFillColor(color)
        c.circle(0.72 * inch, yy + 0.06 * inch, 4, stroke=0, fill=1)
        c.setFillColor(TEXT)
        c.setFont(FONT_BOLD, 9)
        c.drawString(0.85 * inch, yy, label)
        c.setFillColor(GRAY)
        c.setFont(FONT, 7.8)
        c.drawString(2.15 * inch, yy, sub)
    c.setFillColor(colors.white)
    c.setStrokeColor(ORANGE)
    c.roundRect(0.85 * inch, 0.36 * inch, 5.35 * inch, 0.45 * inch, 6, stroke=1, fill=1)
    c.setFillColor(TEXT)
    c.setFont(FONT_BOLD, 9.2)
    c.drawCentredString(3.525 * inch, 0.52 * inch, "Strong advisory hour = at least 45 min of GDP or GS overlap")
    c.showPage()
    _save(c)


def fig3_monthly_effects() -> None:
    df = pd.read_csv(MAIN / "window_month_mild_conflict_delta.csv")
    df = df[df["window"].isin(["active", "post_3h"])].copy()
    pivot = df.pivot(index="month", columns="window", values="delta_arr_delay60_rate")
    c = _canvas(OUT / "fig3_monthly_effects.pdf")
    x0, y0, ww, hh = 0.62 * inch, 1.02 * inch, 5.95 * inch, 2.85 * inch
    ymax = 0.32
    _axis(c, x0, y0, ww, hh, ymax)
    months = list(range(1, 13))
    xs = [x0 + ww * (m - 1) / 11 for m in months]
    point_xy = {}
    for name, color in [("active", BLUE), ("post_3h", ORANGE)]:
        ys = [y0 + hh * float(pivot.loc[m, name]) / ymax for m in months]
        _line(c, xs, ys, color)
        point_xy[name] = dict(zip(months, ys))
    active = pivot["active"]
    post = pivot["post_3h"]
    active_max = int(active.idxmax())
    active_min = int(active.idxmin())
    post_max = int(post.idxmax())
    post_min = int(post.idxmin())
    _point_label(c, xs[active_max - 1], point_xy["active"][active_max], f"+{active.loc[active_max]:.3f}", BLUE, dy=12)
    _point_label(c, xs[post_max - 1], point_xy["post_3h"][post_max], f"+{post.loc[post_max]:.3f}", ORANGE, dy=-16)
    _point_label(c, xs[active_min - 1], point_xy["active"][active_min], f"+{active.loc[active_min]:.3f}", BLUE, dy=10)
    _point_label(c, xs[post_min - 1], point_xy["post_3h"][post_min], f"+{post.loc[post_min]:.3f}", ORANGE, dy=-16)
    c.setFont(FONT, 7)
    c.setFillColor(GRAY)
    for m, x in zip(months, xs):
        c.drawCentredString(x, y0 - 15, str(m))
    c.drawCentredString(x0 + ww / 2, y0 - 32, "Month in 2025")
    _legend_centered(c, [("Active", BLUE), ("Post 3 h", ORANGE)], 0.26 * inch)
    c.showPage()
    _save(c)


def fig4_counterfactual() -> None:
    df = pd.read_csv(MAIN / "post3h_counterfactual_state_pairs.csv")
    row = df[df["label"] == "mild_strong_vs_nonmild_no_strong"].iloc[0]
    values = [
        ("Mild + strong advisory", row["conflict_arr_delay60_rate"], ORANGE),
        ("Nonmild + no advisory", row["baseline_arr_delay60_rate"], BLUE),
    ]
    c = _canvas(OUT / "fig4_counterfactual_states.pdf", 6.2 * inch, 3.8 * inch)
    x0, y0, ww, hh = 0.82 * inch, 0.92 * inch, 4.9 * inch, 2.45 * inch
    ymax = 0.32
    _axis(c, x0, y0, ww, hh, ymax)
    bar_w = 0.75 * inch
    for i, (label, value, color) in enumerate(values):
        x = x0 + 1.2 * inch + i * 1.8 * inch
        bh = hh * value / ymax
        c.setFillColor(color)
        c.rect(x, y0, bar_w, bh, stroke=0, fill=1)
        c.setFillColor(TEXT)
        c.setFont(FONT_BOLD, 10)
        c.drawCentredString(x + bar_w / 2, y0 + bh + 10, f"{value:.3f}")
        c.setFillColor(GRAY)
        c.setFont(FONT, 7.2)
        c.drawCentredString(x + bar_w / 2, y0 - 17, label)
    c.setFillColor(TEXT)
    c.setFont(FONT_BOLD, 9)
    c.drawString(3.7 * inch, 2.75 * inch, "Difference: +0.203")
    c.showPage()
    _save(c)


def fig5_temporal_windows() -> None:
    df = pd.read_csv(MAIN / "lag_window_scorecard.csv")
    keys = [
        ("Lead 3 h", "lead_3h", "all"),
        ("Lead 1 h", "lead_1h", "all"),
        ("Active", "active", "all"),
        ("Lag 1 h", "lag_1h", "clean_no_active"),
        ("Lag 2 h", "lag_2h", "clean_no_active"),
        ("Lag 3 h", "lag_3h", "clean_no_active"),
        ("Lag 6 h", "lag_6h", "clean_no_active"),
    ]
    vals = []
    base = []
    for _, window, scope in keys:
        row = df[(df["window"] == window) & (df["scope"] == scope)].iloc[0]
        vals.append(float(row["conflict_arr_delay60_rate"]))
        base.append(float(row["baseline_arr_delay60_rate"]))
    c = _canvas(OUT / "fig5_temporal_windows.pdf")
    x0, y0, ww, hh = 0.72 * inch, 1.04 * inch, 5.8 * inch, 2.82 * inch
    ymax = 0.34
    _axis(c, x0, y0, ww, hh, ymax)
    xs = [x0 + ww * i / (len(keys) - 1) for i in range(len(keys))]
    _line(c, xs, [y0 + hh * v / ymax for v in vals], ORANGE)
    _line(c, xs, [y0 + hh * v / ymax for v in base], GRAY, 1.0)
    c.setFont(FONT, 6.9)
    c.setFillColor(GRAY)
    for (label, _, _), x in zip(keys, xs):
        c.drawCentredString(x, y0 - 15, label)
    _legend_centered(c, [("Conflict state", ORANGE), ("Mild non-advisory baseline", GRAY)], 0.26 * inch)
    c.showPage()
    _save(c)


def fig4_counterfactual_temporal() -> None:
    counter = pd.read_csv(MAIN / "post3h_counterfactual_state_pairs.csv")
    counter_row = counter[counter["label"] == "mild_strong_vs_nonmild_no_strong"].iloc[0]
    bars = [
        ("Mild + advisory", float(counter_row["conflict_arr_delay60_rate"]), ORANGE),
        ("Nonmild + no advisory", float(counter_row["baseline_arr_delay60_rate"]), BLUE),
    ]

    lag = pd.read_csv(MAIN / "lag_window_scorecard.csv")
    keys = [
        ("Lead 3 h", "lead_3h", "all"),
        ("Lead 1 h", "lead_1h", "all"),
        ("Active", "active", "all"),
        ("Lag 1 h", "lag_1h", "clean_no_active"),
        ("Lag 2 h", "lag_2h", "clean_no_active"),
        ("Lag 3 h", "lag_3h", "clean_no_active"),
        ("Lag 6 h", "lag_6h", "clean_no_active"),
    ]
    conflict = []
    baseline = []
    for _, window, scope in keys:
        row = lag[(lag["window"] == window) & (lag["scope"] == scope)].iloc[0]
        conflict.append(float(row["conflict_arr_delay60_rate"]))
        baseline.append(float(row["baseline_arr_delay60_rate"]))

    c = _canvas(OUT / "fig4_counterfactual_temporal.pdf", 7.1 * inch, 3.55 * inch)
    ymax = 0.34

    c.setFillColor(TEXT)
    c.setFont(FONT_BOLD, 8.8)
    c.drawString(0.42 * inch, 3.22 * inch, "(a) Counterfactual state comparison")
    c.drawString(3.35 * inch, 3.22 * inch, "(b) Advisory-window risk profile")

    x0, y0, ww, hh = 0.52 * inch, 0.70 * inch, 2.74 * inch, 2.25 * inch
    _axis(c, x0, y0, ww, hh, ymax, label_color=TEXT)
    bar_w = 0.48 * inch
    for i, (label, value, color) in enumerate(bars):
        x = x0 + 0.48 * inch + i * 1.02 * inch
        bh = hh * value / ymax
        c.setFillColor(color)
        c.rect(x, y0, bar_w, bh, stroke=0, fill=1)
        c.setFillColor(TEXT)
        c.setFont(FONT_BOLD, 8.4)
        c.drawCentredString(x + bar_w / 2, y0 + bh + 7, f"{value:.3f}")
        c.setFillColor(TEXT)
        c.setFont(FONT, 6.5)
        c.drawCentredString(x + bar_w / 2, y0 - 15, label)
    c.setFillColor(TEXT)
    c.setFont(FONT_BOLD, 8.0)
    c.drawCentredString(x0 + ww * 0.70, y0 + hh * 0.72, "Difference +0.203")

    x1, y1, w1, h1 = 3.66 * inch, 0.70 * inch, 2.98 * inch, 2.25 * inch
    _axis(c, x1, y1, w1, h1, ymax, label_color=TEXT)
    xs = [x1 + w1 * i / (len(keys) - 1) for i in range(len(keys))]
    conflict_y = [y1 + h1 * v / ymax for v in conflict]
    baseline_y = [y1 + h1 * v / ymax for v in baseline]
    _line(c, xs, conflict_y, ORANGE)
    _line(c, xs, baseline_y, GRAY, 1.0)
    c.setFont(FONT, 6.3)
    c.setFillColor(TEXT)
    for (label, _, _), x in zip(keys, xs):
        c.drawCentredString(x, y1 - 15, label)
    active_idx = 2
    lag3_idx = 5
    lag6_idx = 6
    _point_label(c, xs[active_idx], conflict_y[active_idx], f"{conflict[active_idx]:.3f}", ORANGE, dy=10)
    _point_label(c, xs[lag3_idx], conflict_y[lag3_idx], f"{conflict[lag3_idx]:.3f}", ORANGE, dy=10)
    _point_label(c, xs[lag6_idx], conflict_y[lag6_idx], f"{conflict[lag6_idx]:.3f}", ORANGE, dx=-7, dy=10)
    c.setFillColor(TEXT)
    c.setFont(FONT, 7.2)
    c.drawRightString(x1 + w1, y1 + h1 * baseline[-1] / ymax - 16, "baseline near 0.06")
    _legend_centered(c, [("Conflict state", ORANGE), ("Mild non-advisory baseline", GRAY)], 0.27 * inch, text_color=TEXT)

    c.showPage()
    _save(c)


def fig6_prediction_gain() -> None:
    df = pd.read_csv(MAIN / "fusion_prediction_monthly_increment.csv")
    df = df[df["model"] == "active_post3h_fusion"].copy()
    long_df = df[df["target"] == "long_arrival_delay"].set_index("fold_month")
    cancel_df = df[df["target"] == "cancellation"].set_index("fold_month")
    c = _canvas(OUT / "fig6_prediction_gain.pdf")
    x0, y0, ww, hh = 0.62 * inch, 1.02 * inch, 5.95 * inch, 2.85 * inch
    ymax = 0.10
    _axis(c, x0, y0, ww, hh, ymax, label_color=TEXT)
    months = list(range(1, 13))
    xs = [x0 + ww * (m - 1) / 11 for m in months]
    long_values = [float(long_df.loc[m, "auc_gain"]) for m in months]
    cancel_values = [float(cancel_df.loc[m, "auc_gain"]) for m in months]
    long_ys = [y0 + hh * v / ymax for v in long_values]
    cancel_ys = [y0 + hh * v / ymax for v in cancel_values]
    _line(c, xs, long_ys, BLUE)
    _line(c, xs, cancel_ys, GREEN)
    long_s = pd.Series(long_values, index=months)
    cancel_s = pd.Series(cancel_values, index=months)
    long_max = int(long_s.idxmax())
    long_min = int(long_s.idxmin())
    cancel_max = int(cancel_s.idxmax())
    cancel_min = int(cancel_s.idxmin())
    _point_label(c, xs[long_max - 1], long_ys[long_max - 1], f"+{long_s.loc[long_max]:.3f}", BLUE, dy=12)
    _point_label(c, xs[long_min - 1], long_ys[long_min - 1], f"+{long_s.loc[long_min]:.3f}", BLUE, dy=-16)
    _point_label(c, xs[cancel_max - 1], cancel_ys[cancel_max - 1], f"+{cancel_s.loc[cancel_max]:.3f}", GREEN, dy=12)
    _point_label(c, xs[cancel_min - 1], cancel_ys[cancel_min - 1], f"+{cancel_s.loc[cancel_min]:.3f}", GREEN, dy=-16)
    c.setFont(FONT, 7)
    c.setFillColor(TEXT)
    for m, x in zip(months, xs):
        c.drawCentredString(x, y0 - 15, str(m))
    c.drawCentredString(x0 + ww / 2, y0 - 32, "Held-out month")
    _legend_centered(c, [("Long delay", BLUE), ("Cancellation", GREEN)], 0.26 * inch, text_color=TEXT)
    c.showPage()
    _save(c)


def main() -> None:
    fig1_multi_source_fusion()
    fig2_temporal_fusion()
    fig3_monthly_effects()
    fig4_counterfactual_temporal()
    fig6_prediction_gain()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
