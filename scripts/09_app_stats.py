# -*- coding: utf-8 -*-
"""Collect the data and training numbers the web app shows into one file: reports/app_stats.json.

The app never hard-codes a number. This script runs no
model; it reads only files that earlier steps saved:

    data/processed/bio_clean.jsonl          corpus size (sentences, people)
    data/synthetic/bio_synthetic.jsonl      synthetic pairs and their split
    reports/synthetic_quality_gate.json     generated error-type counts
    data/processed/hiwiked_bio.jsonl        real pairs; edit types re-typed with the 8-type classifier
    data/processed/hiwiked_dev_split.json   dev_train / dev_select / dropped near-duplicates
    models/corrector/train_summary.json     training-loss curve, steps, val loss, run time
    reports/results_final.json              headline scores (19_phase_e.py, from saved outputs)

Run after 19_phase_e.py:  .venv\\Scripts\\python.exe scripts\\09_app_stats.py
"""

from __future__ import annotations  # modern type-hint syntax

import json  # read/write JSON
import sys  # stdout, import path
from collections import Counter  # counting dictionary
from pathlib import Path  # file paths

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # print Hindi
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # make hindi_gec importable

from hindi_gec import EXTENDED_TYPES, OTHER  # noqa: E402
from hindi_gec.config import load_config  # noqa: E402
from hindi_gec.edit_extractor import extract_edits  # noqa: E402
from hindi_gec.schema import read_jsonl  # noqa: E402

COMMAND = r".venv\Scripts\python.exe scripts\09_app_stats.py"


def main() -> None:
    cfg = load_config()
    rep = cfg.path("reports")
    print("=" * 72)
    print("APP STATS : numbers for the web app, from saved files only")
    print("=" * 72)

    clean = list(read_jsonl(cfg.path("bio_clean")))
    corpus = {"sentences": len(clean), "pages": len({r["page_title"] for r in clean})}
    print(f"  bio_clean      : {corpus['sentences']} sentences, {corpus['pages']} pages")

    syn_split = Counter(r["split"] for r in read_jsonl(cfg.path("bio_synthetic")))
    gate = json.loads((rep / "synthetic_quality_gate.json").read_text(encoding="utf-8"))
    synthetic = {"pairs": gate["pairs"], "train": syn_split["train"], "val": syn_split["val"],
                 "clean_share": gate["none_share"], "type_counts": gate["type_counts"]}
    print(f"  synthetic      : {synthetic['pairs']} pairs (train {synthetic['train']}, val {synthetic['val']})")

    hw = list(read_jsonl(cfg.path("hiwiked_bio")))
    split = json.loads(cfg.path("dev_split").read_text(encoding="utf-8"))["split"]
    sides = Counter(split.values())
    types = Counter(e.type for r in hw for e in extract_edits(r["source_tokens"], r["target_tokens"], extended=True))
    real = {"pairs": len(hw), "dev": sum(r["split"] == "dev" for r in hw), "test": sum(r["split"] == "test" for r in hw),
            "dev_train": sides["dev_train"], "dev_train_dropped": sides["dev_train_dropped"],
            "dev_select": sides["dev_select"],
            "edit_types": {t: types[t] for t in list(EXTENDED_TYPES) + [OTHER]}, "edits": sum(types.values())}
    print(f"  hiwiked_bio    : {real['pairs']} pairs: dev {real['dev']} (dev_train {real['dev_train']}, dev_select "
          f"{real['dev_select']}, near-duplicates dropped {real['dev_train_dropped']}), test {real['test']}; "
          f"{real['edits']} edits, OTHER {types[OTHER]} (8-type classifier)")

    s = json.loads((cfg.path("model") / "train_summary.json").read_text(encoding="utf-8"))
    k = cfg["splits"]["dev_train_upsample"]
    training = {"steps": s["steps_done"], "minutes": round(s["seconds"] / 60, 1), "val_loss": round(s["val_loss"], 4),
                "losses": s["losses"], "epochs": cfg["corrector"]["epochs"],
                "pairs": synthetic["train"] + real["dev_train"] * k,
                "synthetic_pairs": synthetic["train"], "real_pairs": real["dev_train"], "real_upsample": k}
    print(f"  training       : {training['pairs']} pairs ({synthetic['train']} synthetic + {real['dev_train']} real x {k}), "
          f"{training['steps']} steps, {training['minutes']} min, val loss {training['val_loss']}")

    rf = json.loads(cfg.path("results_final").read_text(encoding="utf-8"))
    t = rf["sets"]["hiwiked_test"]["IndicBART corrector"]
    headline = {"test_f05": t["all edits"]["f05"], "test_precision": t["all edits"]["p"],
                "test_f05_7types": t["original 7 types"]["f05"], "test_f05_extended": t["extended (+PRON)"]["f05"]}
    stats = {"generated_by": COMMAND, "model_label": cfg["app"]["model_label"], "error_types": list(EXTENDED_TYPES),
             "corpus": corpus, "synthetic": synthetic, "real": real, "training": training, "headline": headline}
    out = cfg.path("app_stats")
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  headline       : real-test F0.5 {headline['test_f05']:.3f} (P {headline['test_precision']:.3f}); "
          f"{len(EXTENDED_TYPES)} error types\n  written        : {out}")


if __name__ == "__main__":
    main()
