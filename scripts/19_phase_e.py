# -*- coding: utf-8 -*-
"""Final results: the IndicBART corrector vs System A, and the evaluation extras.

Every number is computed from saved system outputs (reports/predictions/):

  final_test_corrector.jsonl     corrector on HiWikEd test, decoded once (18_final_test.py)
  final_test_baseline_A.jsonl    System A on HiWikEd test, decoded once (16_baseline_A.py --final-test)
  dev_select_corrector.jsonl     corrector on dev_select       } decoded here (not test data),
  synthetic_val_corrector.jsonl  corrector on synthetic val    } saved before scoring
  <set>_baseline_A.jsonl         System A on dev_select / synthetic val (16_baseline_A.py)

1. Edit-level P / R / F0.5 in three views (hindi_gec.scoring), GLEU
   (hindi_gec.gleu), exact match, for both systems on test, dev_select and
   synthetic val, next to the GLEU of the unchanged source ("do nothing").
2. Per-type scores on test (8-type classifier) for the corrector.
3. Type confusion matrix on synthetic val (true type = the generator's label,
   right by construction; predicted = the type of the corrector's edit at
   the same place, metrics.type_confusion). JSON + PNG.
4. Latency on the GPU: the full app pipeline (beams, per-edit confidence,
   explanations, alternatives) on 100 dev_select sentences, batch 1 and 16.
5. Error analysis: 20 random test failures with automatic category notes.

The test set is NOT decoded here: if final_test_corrector.jsonl is missing,
run 18_final_test.py (once).

Run:  .venv\\Scripts\\python.exe scripts\\19_phase_e.py
Writes reports/results_final.json, reports/results_final.md, reports/confusion.png,
reports/predictions/dev_select_corrector.jsonl, reports/predictions/synthetic_val_corrector.jsonl.
"""

from __future__ import annotations  # modern type-hint syntax

import json  # read/write JSON
import random  # seeded sample
import statistics  # mean
import sys  # stdout, import path
import time  # timing
from collections import Counter  # counts
from pathlib import Path  # file paths

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # print Hindi
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # make hindi_gec importable

from hindi_gec import EXTENDED_TYPES, OTHER  # noqa: E402
from hindi_gec.config import load_config  # noqa: E402
from hindi_gec.edit_extractor import extract_edits  # noqa: E402
from hindi_gec.gleu import corpus_gleu  # noqa: E402
from hindi_gec.inference import Corrector, guard_rejects  # noqa: E402
from hindi_gec.metrics import MISSED, SPURIOUS, type_confusion  # noqa: E402
from hindi_gec.schema import Edit, read_jsonl  # noqa: E402
from hindi_gec.scoring import decode_all, exact_match, guard_outputs, views  # noqa: E402

COMMAND = r".venv\Scripts\python.exe scripts\19_phase_e.py"
CORR, BASE = "IndicBART corrector", "System A"  # the two systems
SETS = (("hiwiked_test", "HiWikEd-bio test"), ("dev_select", "HiWikEd-bio dev_select"), ("synthetic_val", "synthetic val"))
RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]  # blue ramp


def save_preds(path: Path, rows: list, raws: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r, o in zip(rows, raws):
            f.write(json.dumps({"id": r["id"], "source": r["source"], "target": r["target"],
                                "raw_output": " ".join(o)}, ensure_ascii=False) + "\n")


def load_raw(path: Path) -> dict:  # id -> raw output tokens (corrector files) or output tokens (System A files)
    return {p["id"]: (p.get("raw_output", p.get("output")) or "").split() for p in read_jsonl(path)}


def summary(rows: list, finals: list, beta: float, n: int, n_guard: int = 0) -> dict:
    out = views(rows, finals, beta)
    out["gleu"] = round(corpus_gleu(((r["source_tokens"], o, r["target_tokens"]) for r, o in zip(rows, finals)), n), 6)
    out["exact_match"] = round(exact_match(rows, finals), 6)
    out["guard_rejected"] = n_guard
    return out


# ----------------------------------------------------------------------
# Confusion matrix
# ----------------------------------------------------------------------
def confusion(rows: list, finals: list) -> dict:
    cells: Counter = Counter()
    for r, o in zip(rows, finals):
        hyp = extract_edits(r["source_tokens"], o, extended=True)
        ref = [Edit(**e) for e in r["edits"]]  # generator labels: true by construction
        cells.update(type_confusion(hyp, ref))
    types = list(EXTENDED_TYPES)
    rows_l, cols_l = types + [SPURIOUS], types + [OTHER, MISSED]
    matrix = [[cells[(t, p)] for p in cols_l] for t in rows_l]
    diag = sum(cells[(t, t)] for t in types)
    found = sum(v for (t, p), v in cells.items() if t != SPURIOUS and p != MISSED)
    return {"pairs": len(rows), "true": rows_l, "predicted": cols_l, "matrix": matrix,
            "type_accuracy_when_found": round(diag / found, 6) if found else 0.0, "found": found,
            "reference_edits": sum(v for (t, _), v in cells.items() if t != SPURIOUS)}


def confusion_png(c: dict, path: Path) -> None:
    """Row-normalised heatmap (share of each true type), counts printed in the cells."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=15)
    bold = ImageFont.load_default(size=17)
    cw, ch, left, top = 86, 34, 120, 92
    rows, cols, mat = c["true"], c["predicted"], c["matrix"]
    img = Image.new("RGB", (left + cw * len(cols) + 20, top + ch * len(rows) + 50), "white")
    d = ImageDraw.Draw(img)
    d.text((left, 12), f"Type confusion, IndicBART corrector on synthetic val ({c['pairs']} pairs)",
           fill="#1f2433", font=bold)
    d.text((left, 36), "row = true type (generator), column = type of the system edit at the same place; "
           "colour = row share", fill="#5b6170", font=font)
    for j, p in enumerate(cols):
        d.text((left + j * cw + cw / 2, top - 10), p, fill="#1f2433", font=font, anchor="ms")
    for i, t in enumerate(rows):
        total = sum(mat[i]) or 1
        d.text((left - 8, top + i * ch + ch / 2), t, fill="#1f2433", font=font, anchor="rm")
        for j, v in enumerate(mat[i]):
            share = v / total
            col = RAMP[min(len(RAMP) - 1, round(share * (len(RAMP) - 1)))]
            x, y = left + j * cw, top + i * ch
            d.rectangle([x + 1, y + 1, x + cw - 1, y + ch - 1], fill=col)
            if v:
                d.text((x + cw / 2, y + ch / 2), str(v), fill="white" if share > 0.5 else "#1f2433",
                       font=font, anchor="mm")
    d.text((left, top + ch * len(rows) + 18), f"type right when the edit was found: "
           f"{c['type_accuracy_when_found']:.1%} of {c['found']}", fill="#5b6170", font=font)
    img.save(path)


# ----------------------------------------------------------------------
# Latency
# ----------------------------------------------------------------------
def latency(cfg, corr: Corrector, sents: list) -> dict:
    ev = cfg["evaluation"]
    sync = corr.torch.cuda.synchronize if corr.device == "cuda" else (lambda: None)
    for s in sents[:5]:  # warm-up (CUDA kernels, caches)
        corr.correct(s)
    one, batch = [], []
    b = ev["latency_batch"]
    for _ in range(ev["latency_repeats"]):
        for s in sents:
            sync()
            t = time.perf_counter()
            corr.correct(s)
            sync()
            one.append((time.perf_counter() - t) * 1000)
        for k in range(0, len(sents), b):
            chunk = sents[k:k + b]
            sync()
            t = time.perf_counter()
            corr.correct_batch(chunk, batch_size=b)
            sync()
            batch.append((time.perf_counter() - t) * 1000 / len(chunk))

    def p95(x: list) -> float:
        x = sorted(x)
        return x[min(len(x) - 1, round(0.95 * (len(x) - 1)))]
    return {"device": corr.device, "sentences": len(sents), "repeats": ev["latency_repeats"], "batch": b,
            "batch_1": {"mean_ms": round(statistics.mean(one), 1), "p95_ms": round(p95(one), 1), "samples": len(one)},
            "batched": {"mean_ms_per_sentence": round(statistics.mean(batch), 1),
                        "p95_ms_per_sentence": round(p95(batch), 1), "batches": len(batch)}}


# ----------------------------------------------------------------------
# Error analysis
# ----------------------------------------------------------------------
def error_analysis(cfg, rows: list, raws: list, finals: list) -> list:
    g = cfg["corrector"]["hallucination_guard"]
    fails = [k for k, (r, o) in enumerate(zip(rows, finals)) if o != list(r["target_tokens"])]
    pick = sorted(random.Random(cfg.seed).sample(fails, min(cfg["evaluation"]["error_analysis_n"], len(fails))))
    out = []
    for k in pick:
        r, o = rows[k], finals[k]
        hyp = extract_edits(r["source_tokens"], o, extended=True)
        ref = extract_edits(r["source_tokens"], r["target_tokens"], extended=True)
        hk = {(tuple(e.src_span), e.tgt) for e in hyp}
        notes = []
        if guard_rejects(r["source_tokens"], raws[k], g["max_token_edits"], g["max_edit_ratio"]):
            notes.append("guard rejected the model output")
        for t, p in type_confusion(hyp, ref):
            if t == SPURIOUS:
                notes.append(f"spurious {p}")
            elif p == MISSED:
                notes.append(f"missed {t}")
        for e in ref:  # found at the right place but fixed differently
            if (tuple(e.src_span), e.tgt) not in hk and any(
                    h.src_span[0] < e.src_span[1] and e.src_span[0] < h.src_span[1] for h in hyp):
                notes.append(f"wrong fix {e.type}")
        out.append({"id": r["id"], "source": r["source"], "target": r["target"], "output": " ".join(o),
                    "reference_edits": [f"{e.type}: {e.src or '∅'} → {e.tgt or '∅'}" for e in ref],
                    "system_edits": [f"{e.type}: {e.src or '∅'} → {e.tgt or '∅'}" for e in hyp],
                    "notes": notes, "out_of_taxonomy_only": all(e.type == OTHER for e in ref)})
    return out


def main() -> None:
    cfg = load_config()
    beta, n = cfg["evaluation"]["beta"], cfg["evaluation"]["gleu_n"]
    g = cfg["corrector"]["hallucination_guard"]
    rep, pred = cfg.path("reports"), cfg.path("predictions")
    t0 = time.monotonic()
    print("=" * 72)
    print("FINAL RESULTS : IndicBART corrector vs System A, from saved outputs")
    print("=" * 72)
    test_file = pred / "final_test_corrector.jsonl"
    if not test_file.exists():
        sys.exit(f"{test_file} is missing: run scripts/18_final_test.py once first.")
    hw = list(read_jsonl(cfg.path("hiwiked_bio")))
    split = json.loads(cfg.path("dev_split").read_text(encoding="utf-8"))["split"]
    data = {"hiwiked_test": [r for r in hw if r["split"] == "test"],
            "dev_select": [r for r in hw if split.get(r["id"]) == "dev_select"],
            "synthetic_val": [r for r in read_jsonl(cfg.path("bio_synthetic")) if r["split"] == "val"]}

    corr = Corrector(cfg)
    for name in ("dev_select", "synthetic_val"):  # not test data: decoded here, saved, then scored from the file
        t = time.monotonic()
        save_preds(pred / f"{name}_corrector.jsonl", data[name], decode_all(corr, data[name]))
        print(f"  decoded        : {name} ({len(data[name])} pairs) in {time.monotonic() - t:.0f} s "
              f"-> predictions/{name}_corrector.jsonl")

    results, finals_c, raws_c = {}, {}, {}
    for name, label in SETS:
        rows = data[name]
        c_raw = load_raw(pred / ("final_test_corrector.jsonl" if name == "hiwiked_test" else f"{name}_corrector.jsonl"))
        a_out = load_raw(pred / f"{'final_test' if name == 'hiwiked_test' else name}_baseline_A.jsonl")
        assert set(c_raw) == set(a_out) == {r["id"] for r in rows}, f"{name}: prediction ids do not match the data"
        raws = [c_raw[r["id"]] for r in rows]
        finals, n_guard = guard_outputs(rows, raws, g["max_token_edits"], g["max_edit_ratio"])
        base = [a_out[r["id"]] for r in rows]
        results[name] = {"label": label, "pairs": len(rows),
                         "gleu_source": round(corpus_gleu(((r["source_tokens"], r["source_tokens"], r["target_tokens"])
                                                           for r in rows), n), 6),
                         CORR: summary(rows, finals, beta, n, n_guard), BASE: summary(rows, base, beta, n)}
        finals_c[name], raws_c[name] = finals, raws
        x, y = results[name][CORR], results[name][BASE]
        print(f"  {name:14s}: corrector F0.5 {x['all edits']['f05']:.3f} GLEU {x['gleu']:.4f} | System A F0.5 "
              f"{y['all edits']['f05']:.3f} GLEU {y['gleu']:.4f} | source GLEU {results[name]['gleu_source']:.4f}")

    b = json.loads((rep / "baseline_A.json").read_text(encoding="utf-8"))["sets"]  # consistency with 16's own scoring
    for name in ("hiwiked_test", "dev_select", "synthetic_val"):
        for v in ("all edits", "original 7 types", "extended (+PRON)"):
            mine, theirs = results[name][BASE][v], b[name][v]
            assert (mine["tp"], mine["fp"], mine["fn"]) == (theirs["tp"], theirs["fp"], theirs["fn"]), (name, v)
    print("  check          : System A counts equal reports/baseline_A.json in every set and view")

    conf = confusion(data["synthetic_val"], finals_c["synthetic_val"])
    confusion_png(conf, rep / "confusion.png")
    print(f"  confusion      : type right when found {conf['type_accuracy_when_found']:.1%} of {conf['found']} "
          f"({conf['reference_edits']} reference edits) -> reports/confusion.png")

    sel = [r["source"] for r in data["dev_select"]][: cfg["evaluation"]["latency_sentences"]]
    lat = latency(cfg, corr, sel)
    print(f"  latency        : {lat['device']}, batch 1 mean {lat['batch_1']['mean_ms']} ms p95 {lat['batch_1']['p95_ms']} ms; "
          f"batch {lat['batch']} {lat['batched']['mean_ms_per_sentence']} ms/sentence "
          f"p95 {lat['batched']['p95_ms_per_sentence']}")
    del corr

    errors = error_analysis(cfg, data["hiwiked_test"], raws_c["hiwiked_test"], finals_c["hiwiked_test"])
    per_type = results["hiwiked_test"][CORR]["extended (+PRON)"]["by_type"]
    out = {"generated_by": COMMAND, "systems": [CORR, BASE], "sets": results,
           "per_type_test": {t: per_type[t] for t in list(EXTENDED_TYPES) + [OTHER] if t in per_type},
           "confusion": conf, "latency": lat, "error_analysis": errors,
           "sources": {"test": ["reports/predictions/final_test_corrector.jsonl",
                                "reports/predictions/final_test_baseline_A.jsonl"],
                       "dev_select": ["reports/predictions/dev_select_corrector.jsonl",
                                      "reports/predictions/dev_select_baseline_A.jsonl"],
                       "synthetic_val": ["reports/predictions/synthetic_val_corrector.jsonl",
                                         "reports/predictions/synthetic_val_baseline_A.jsonl"]}}
    cfg.path("results_final").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    md = ["# Final results — IndicBART corrector vs System A", "",
          f"Produced by `{COMMAND}` from saved system outputs (reports/predictions/). HiWikEd-bio test was decoded "
          "once per system (18_final_test.py, 16_baseline_A.py --final-test) after every choice was made on "
          "dev_select and synthetic val. Edit-level scores with the hallucination guard on, β = 0.5.", ""]
    for name, label in SETS:
        r = results[name]
        md += [f"## {label} ({r['pairs']:,} pairs)", "",
               "| system | P | R | F0.5 all edits | F0.5 7 types | F0.5 + PRON | GLEU | exact match |",
               "|---|---|---|---|---|---|---|---|"]
        for s in (CORR, BASE):
            x = r[s]
            a = x["all edits"]
            md.append(f"| {s} | {a['p']:.3f} | {a['r']:.3f} | **{a['f05']:.3f}** | {x['original 7 types']['f05']:.3f} | "
                      f"{x['extended (+PRON)']['f05']:.3f} | {x['gleu']:.4f} | {x['exact_match']:.1%} |")
        md.append(f"| do nothing (unchanged source) | | | | | | {r['gleu_source']:.4f} | |")
        md.append("")
    md += ["## Per error type, HiWikEd-bio test (corrector, 8-type classifier)", "",
           "| type | TP | FP | FN | F0.5 |", "|---|---|---|---|---|"]
    for t, x in out["per_type_test"].items():
        md.append(f"| {t} | {x['tp']} | {x['fp']} | {x['fn']} | {x['f05']:.3f} |")
    md += ["", "## Type confusion (synthetic val)", "", "![confusion](confusion.png)", "",
           f"When the corrector found an error it gave it the right type {conf['type_accuracy_when_found']:.1%} of the "
           f"time ({conf['found']:,} found of {conf['reference_edits']:,} reference edits).", "",
           "## Latency (full app pipeline, 100 dev_select sentences, each setting timed 3x)", "",
           "| device | 1 sentence: mean | p95 | batch of 16: mean per sentence | p95 |", "|---|---|---|---|---|",
           f"| {lat['device']} | {lat['batch_1']['mean_ms']} ms | {lat['batch_1']['p95_ms']} ms | "
           f"{lat['batched']['mean_ms_per_sentence']} ms | {lat['batched']['p95_ms_per_sentence']} ms |", "",
           f"## Error analysis: {len(errors)} random test failures", ""]
    cnt = Counter(x.rsplit(" ", 1)[0] if x.startswith(("missed", "spurious", "wrong")) else x
                  for e in errors for x in e["notes"])
    md += ["Note counts: " + ", ".join(f"{k} {v}" for k, v in cnt.most_common()) + ".", ""]
    for k, e in enumerate(errors, 1):
        md += [f"{k}. `{e['id']}` — {'; '.join(e['notes']) or 'no note'}",
               f"   - source: {e['source']}", f"   - output: {e['output']}", f"   - target: {e['target']}"]
    (rep / "results_final.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n  written        : reports/results_final.json, reports/results_final.md, reports/confusion.png"
          f"\n  wall clock     : {(time.monotonic() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
