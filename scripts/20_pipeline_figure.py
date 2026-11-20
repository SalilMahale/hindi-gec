# -*- coding: utf-8 -*-
"""Draw the pipeline figure for the About page and the report: app/assets/pipeline.png.

One path from raw text to the demo: what runs, in what order, and what each
step writes. Every number in the figure is read from a saved file
(reports/app_stats.json, reports/results_final.json), so redrawing after a
change keeps it true. Colours: blue = script you run, yellow = file it
writes, violet = model, grey = outside source, green = the demo.

Run:  .venv\\Scripts\\python.exe scripts\\20_pipeline_figure.py
"""

from __future__ import annotations  # modern type-hint syntax

import json  # read the stats
import math  # arrow heads
import sys  # stdout, import path
from pathlib import Path  # file paths
from typing import List, Sequence, Tuple  # type hints

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # print Hindi
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # make hindi_gec importable

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from hindi_gec.config import load_config  # noqa: E402

W, H = 1980, 1300
BW, BH, GAP, X0 = 250, 130, 26, 40  # box width / height, gap, left margin
ROWS = {"A": 230, "B": 500, "C": 800, "D": 1040}  # top of each row of boxes
KIND = {"script": ("#DCE9F8", "#2A78D6"), "file": ("#FFF3D6", "#C98500"), "model": ("#E6E1FA", "#6250D6"),
        "source": ("#ECEAE6", "#555555"), "demo": ("#D4F1E4", "#1BAF7A")}
INK, MUTED, ARROW = "#1F2433", "#4B5163", "#4D4D4D"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    try:  # Segoe UI ships with Windows; PIL finds it by name in the system font folder
        return ImageFont.truetype("segoeuib.ttf" if bold else "segoeui.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def xy(col: int, row: str) -> Tuple[int, int]:  # top-left corner of a grid box
    return X0 + col * (BW + GAP), ROWS[row]


def box(d: ImageDraw.ImageDraw, col: int, row: str, kind: str, title: str, lines: Sequence[str],
        step: str = "") -> None:
    x, y = xy(col, row)
    fill, edge = KIND[kind]
    d.rounded_rectangle([x, y, x + BW, y + BH], radius=16, fill=fill, outline=edge, width=4)
    d.text((x + BW / 2, y + 38), title, fill=INK, font=font(19, True), anchor="mm")
    for k, line in enumerate(lines):
        d.text((x + BW / 2, y + 74 + k * 24), line, fill=MUTED, font=font(16), anchor="mm")
    if step:  # numbered circle = run order
        d.ellipse([x - 16, y - 22, x + 30, y + 24], fill="#2A78D6", outline="white", width=3)
        d.text((x + 7, y + 1), step, fill="white", font=font(17, True), anchor="mm")


def head(d: ImageDraw.ImageDraw, p: Tuple[float, float], q: Tuple[float, float]) -> None:
    a = math.atan2(q[1] - p[1], q[0] - p[0])
    pts = [q, (q[0] - 16 * math.cos(a - 0.45), q[1] - 16 * math.sin(a - 0.45)),
           (q[0] - 16 * math.cos(a + 0.45), q[1] - 16 * math.sin(a + 0.45))]
    d.polygon(pts, fill=ARROW)


def path(d: ImageDraw.ImageDraw, pts: List[Tuple[float, float]], label: str = "", at: int = 0,
         dashed: bool = False) -> None:
    """Poly-line with an arrow head at the end; optional label next to segment ``at``."""
    for p, q in zip(pts, pts[1:]):
        if dashed:
            n = int(math.dist(p, q) // 14)
            for k in range(0, n, 2):
                a = (p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n)
                b = (p[0] + (q[0] - p[0]) * (k + 1) / n, p[1] + (q[1] - p[1]) * (k + 1) / n)
                d.line([a, b], fill=ARROW, width=3)
        else:
            d.line([p, q], fill=ARROW, width=3)
    head(d, pts[-2], pts[-1])
    if label:
        p, q = pts[at], pts[at + 1]
        d.text(((p[0] + q[0]) / 2 + 8, (p[1] + q[1]) / 2 - 14), label, fill=MUTED, font=font(15), anchor="lm")


def right(col: int, row: str) -> Tuple[int, int]:
    x, y = xy(col, row)
    return x + BW, y + BH // 2


def left(col: int, row: str) -> Tuple[int, int]:
    x, y = xy(col, row)
    return x, y + BH // 2


def top(col: int, row: str) -> Tuple[int, int]:
    x, y = xy(col, row)
    return x + BW // 2, y


def bottom(col: int, row: str) -> Tuple[int, int]:
    x, y = xy(col, row)
    return x + BW // 2, y + BH


def chain(d: ImageDraw.ImageDraw, row: str, cols: Sequence[int]) -> None:
    for a, b in zip(cols, cols[1:]):
        p, q = right(a, row), left(b, row)
        path(d, [(p[0] + 2, p[1]), (q[0] - 4, q[1])])


def main() -> None:
    cfg = load_config()
    s = json.loads(cfg.path("app_stats").read_text(encoding="utf-8"))
    rf = json.loads(cfg.path("results_final").read_text(encoding="utf-8"))
    c, y, r, t = s["corpus"], s["synthetic"], s["real"], s["training"]
    test = rf["sets"]["hiwiked_test"]

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((X0, 50), "End-to-end pipeline: what runs, in what order, and what it produces", fill=INK, font=font(40, True))
    d.text((X0, 112), "Numbered circles = run order.  Blue = script you run,  yellow = file it writes,  violet = model,  "
           "grey = outside source,  green = demo.", fill=MUTED, font=font(21))
    for row, name in (("A", "A.  BUILD TRAINING DATA"), ("B", "B.  BUILD REAL DATA, TRAIN THE MODEL"),
                      ("C", "C.  EVALUATE ONCE, BUILD THE DEMO")):
        d.text((X0, ROWS[row] - 62), name, fill="#2A78D6", font=font(21, True))

    # A: corpus -> synthetic errors
    box(d, 0, "A", "source", "Hindi Wikipedia", ["10 biography categories", "(MediaWiki API)"])
    box(d, 1, "A", "script", "01_build_corpus.py", ["scrape, clean, filter,", "remove near-duplicates"], "1")
    box(d, 2, "A", "file", "bio_clean.jsonl", [f"{c['sentences']:,} clean sentences", f"{c['pages']:,} people"])
    box(d, 3, "A", "script", "03_generate_synthetic", ["break sentences on purpose", "8 error types"], "2")
    box(d, 4, "A", "file", "bio_synthetic.jsonl", [f"{y['pairs']:,} (wrong, right) pairs",
                                                   f"train {y['train']:,} / val {y['val']:,}"])
    chain(d, "A", [0, 1, 2, 3, 4])

    # B: real data -> split -> training
    box(d, 0, "B", "source", "HiWikEd .src / .trg", ["real human corrections", "from Wikipedia history"])
    box(d, 1, "B", "script", "02_filter_hiwiked.py", ["keep biography pairs,", "remove leaks, type edits"], "3")
    box(d, 2, "B", "file", "hiwiked_bio.jsonl", [f"{r['pairs']:,} real pairs", f"dev {r['dev']:,} / test {r['test']:,}"])
    box(d, 3, "B", "script", "14_prepare_dev.py", ["split dev; drop", f"{r['dev_train_dropped']} near-duplicates"], "4")
    box(d, 4, "B", "file", "real splits", [f"dev_train {r['dev_train']:,} · dev_select {r['dev_select']:,}",
                                           f"test {r['test']:,}"])
    box(d, 5, "B", "script", "04_train_corrector.py", [f"{t['pairs']:,} pairs, {t['epochs']} epochs",
                                                       f"{t['steps']:,} steps, {t['minutes']:.0f} min"], "5")
    box(d, 6, "B", "model", "models/corrector", ["IndicBART corrector", f"val loss {t['val_loss']:.4f}"])
    box(d, 5, "A", "model", "IndicBART", ["pre-trained", "ai4bharat/IndicBART"])
    chain(d, "B", [0, 1, 2, 3, 4, 5, 6])
    p, q = bottom(5, "A"), top(5, "B")
    path(d, [(p[0], p[1] + 2), (q[0], q[1] - 4)], "starting weights")
    p, q = bottom(4, "A"), top(5, "B")
    mid = (ROWS["A"] + BH + ROWS["B"]) // 2 - 20  # run between the two rows, clear of the IndicBART box
    path(d, [(p[0], p[1] + 2), (p[0], mid), (q[0] - 60, mid), (q[0] - 60, q[1] - 4)], "synthetic train", 1)
    p = bottom(2, "A")
    path(d, [(p[0], p[1] + 2), (p[0], p[1] + 40), (xy(1, "B")[0] + BW - 40, p[1] + 40),
             (xy(1, "B")[0] + BW - 40, ROWS["B"] - 4)], "leakage check", 1, dashed=True)

    # C: evaluation and demo
    box(d, 1, "C", "script", "18_final_test.py", ["corrector on real", "test, exactly once"], "6")
    box(d, 1, "D", "script", "16_baseline_A.py", ["System A: rules +", "4-gram LM (CPU)"], "7")
    box(d, 2, "C", "script", "19_phase_e.py", ["P / R / F0.5, GLEU,", "confusion, speed"], "8")
    box(d, 3, "C", "file", "results_final.json", [f"test F0.5 {test['IndicBART corrector']['all edits']['f05']:.3f}",
                                                  f"System A {test['System A']['all edits']['f05']:.3f}"])
    box(d, 4, "C", "script", "09_app_stats.py", ["data + training", "numbers for the app"], "9")
    box(d, 4, "D", "script", "10_app_examples.py", ["one example per type,", "corrected exactly"], "10")
    box(d, 5, "C", "file", "app_stats.json", ["every number", "the app shows"])
    box(d, 5, "D", "file", "app_examples.json", [f"{len(s['error_types'])} example chips", "+ 5 correct sentences"])
    box(d, 6, "C", "demo", "streamlit_app.py", ["web demo, offline", "(backup: demo_cli.py)"])
    chain(d, "C", [1, 2, 3, 4, 5, 6])
    chain(d, "D", [4, 5])
    p, q = right(1, "D"), bottom(2, "C")
    path(d, [(p[0] + 2, p[1]), (q[0], p[1]), (q[0], q[1] + 4)])
    p, q = right(5, "D"), bottom(6, "C")
    path(d, [(p[0] + 2, p[1]), (q[0], p[1]), (q[0], q[1] + 4)])
    p, q = bottom(6, "B"), top(6, "C")
    path(d, [(p[0], p[1] + 2), (q[0], q[1] - 4)], "loads the model")
    p, q = bottom(4, "B"), top(1, "C")
    path(d, [(p[0], p[1] + 2), (p[0], ROWS["C"] - 30), (q[0], ROWS["C"] - 30), (q[0], q[1] - 4)],
         "real test + dev_select", 1)

    d.text((X0, 1200), "Helper scripts, not in the chain: 00_preprocess_demo.py (shows each preprocessing step), "
           "check_tokenizer_roundtrip.py (IndicBART does not damage Hindi text), 11_build_gazetteer.py (person names), "
           "13_other_patterns.py (dev error analysis), 20_pipeline_figure.py (this figure).", fill=MUTED, font=font(16))
    out = cfg.path("pipeline_figure")
    img.save(out)
    print(f"  written        : {out}  ({W}x{H})")


if __name__ == "__main__":
    main()
