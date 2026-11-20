# -*- coding: utf-8 -*-
"""Phase B2 — person-name gazetteer, and how much it fixes AGR-HON vs AGR-NUM labels.

1. Build the gazetteer from all bio_clean page titles (names.py rules) and save
   it to data/processed/name_gazetteer.txt (the live app loads it).
2. Measure on synthetic val. The generator knows each edit's true type by
   construction (it used the page title to decide honorific). The live app
   never has a page title, so the question is: without the title, how often
   does the edit classifier give an AGR-HON / AGR-NUM edit its true type?
     before : no title, no gazetteer           (the classifier until now)
     after  : no title, gazetteer from TRAIN pages only (val people unseen: honest)
     deploy : no title, gazetteer from all pages (what the app uses)
     title  : with the page title              (upper reference)
   An edit counts when the classifier produces an edit with the same source
   span and replacement; agreement = same type.
3. The test sentence: लाल बहादुर शास्त्री एक अद्भुत इंसान था । -> था → थे.

Run:  .venv\\Scripts\\python.exe scripts\\11_build_gazetteer.py
Writes data/processed/name_gazetteer.txt and reports/honorific_relabel.json.
"""

from __future__ import annotations  # modern type-hint syntax

import json  # write the report
import sys  # stdout, import path
from collections import Counter  # counts
from pathlib import Path  # file paths

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # print Hindi
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # make hindi_gec importable

from hindi_gec.config import load_config  # noqa: E402
from hindi_gec.edit_extractor import extract_edits  # noqa: E402
from hindi_gec.names import build_gazetteer, save_gazetteer  # noqa: E402
from hindi_gec.schema import read_jsonl  # noqa: E402

COMMAND = r".venv\Scripts\python.exe scripts\11_build_gazetteer.py"
TYPES = ("AGR-HON", "AGR-NUM")


def main() -> None:
    cfg = load_config()
    nc = cfg["names"]
    print("=" * 72)
    print("PHASE B2 : PERSON-NAME GAZETTEER")
    print("=" * 72)
    clean = list(read_jsonl(cfg.path("bio_clean")))
    counts = Counter(t for r in clean for t in r["tokens"])  # corpus word counts
    syn = list(read_jsonl(cfg.path("bio_synthetic")))
    train_pages = {r["page_title"] for r in syn if r["split"] == "train"}
    val = [r for r in syn if r["split"] == "val"]
    val_pages = {r["page_title"] for r in val}
    print(f"  page titles    : {len({r['page_title'] for r in clean})} in bio_clean; "
          f"{len(train_pages)} train / {len(val_pages)} val pages in synthetic data (overlap "
          f"{len(train_pages & val_pages)})")

    deploy = build_gazetteer((r["page_title"] for r in clean), counts, nc["max_common_rank"], nc["keep_if_titles"])
    train_only = build_gazetteer(train_pages, counts, nc["max_common_rank"], nc["keep_if_titles"])
    save_gazetteer(deploy, cfg.path("name_gazetteer"))
    print(f"  gazetteer      : {len(deploy)} name tokens (train-pages-only version: {len(train_only)})")

    settings = {"before": ("", frozenset()), "after": (None, train_only), "deploy": (None, deploy), "title": (True, frozenset())}
    stats = {k: Counter() for k in settings}
    for r in val:
        gold = {(tuple(e["src_span"]), e["tgt"]): e["type"] for e in r["edits"] if e["type"] in TYPES}
        if not gold:
            continue
        for name, (title, names) in settings.items():
            pt = r["page_title"] if title else ""
            pred = {(tuple(e.src_span), e.tgt): e.type
                    for e in extract_edits(r["source_tokens"], r["target_tokens"], pt, names)}
            for k, g in gold.items():
                p = pred.get(k, "unmatched")
                stats[name][(g, p)] += 1
    report = {"generated_by": COMMAND, "gazetteer_size": len(deploy), "gazetteer_size_train_only": len(train_only)}
    print(f"\n  {'setting':8s} {'edits':>6s} {'agree':>6s} {'rate':>7s}   HON->NUM  NUM->HON  unmatched")
    for name, c in stats.items():
        n = sum(c.values())
        agree = sum(v for (g, p), v in c.items() if g == p)
        report[name] = {"edits": n, "agree": agree, "rate": round(agree / n, 4),
                        "hon_as_num": c[("AGR-HON", "AGR-NUM")], "num_as_hon": c[("AGR-NUM", "AGR-HON")],
                        "unmatched": sum(v for (g, p), v in c.items() if p == "unmatched"),
                        "by_gold": {t: {"edits": sum(v for (g, _), v in c.items() if g == t),
                                        "agree": c[(t, t)]} for t in TYPES}}
        x = report[name]
        print(f"  {name:8s} {n:6d} {agree:6d} {agree / n:7.1%}   {x['hon_as_num']:8d}  {x['num_as_hon']:8d}  {x['unmatched']:9d}")
    for name in settings:
        b = report[name]["by_gold"]
        print(f"    {name:8s} AGR-HON {b['AGR-HON']['agree']}/{b['AGR-HON']['edits']}   "
              f"AGR-NUM {b['AGR-NUM']['agree']}/{b['AGR-NUM']['edits']}")

    sent = "लाल बहादुर शास्त्री एक अद्भुत इंसान था ।".split()
    fixed = sent[:-2] + ["थे", "।"]
    t_before = extract_edits(sent, fixed)[0].type
    t_after = extract_edits(sent, fixed, names=deploy)[0].type
    print(f"\n  test sentence  : {' '.join(sent)}  था → थे : before {t_before}, after {t_after}"
          f"  (शास्त्री in gazetteer: {'शास्त्री' in deploy})")
    report["test_sentence"] = {"sentence": " ".join(sent), "before": t_before, "after": t_after}
    out = cfg.path("reports") / "honorific_relabel.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  written        : {cfg.path('name_gazetteer')}\n  written        : {out}")


if __name__ == "__main__":
    main()
