# -*- coding: utf-8 -*-
"""Prepare training: dev_train / dev_select split and data sanity checks.

1. Split HiWikEd-bio **dev** 75 / 25 into dev_train (real pairs added to
   training, 5x) and dev_select (model selection only). HiWikEd has no page
   titles, so pairs are grouped by their target sentence: several edits of
   one sentence always land on the same side, so nothing leaks.
1b. Near-duplicate check (hindi_gec.leakage): a dev_train pair whose source or
   target token set has Jaccard >= splits.dev_train_near_dup_jaccard with any
   dev_select or test pair is relabelled "dev_train_dropped" and is not
   trained on. Test is read here only for its token sets; nothing is scored on it.
   Each partner is saved in reports/dev_train_near_dups.json for inspection.
2. Classifier agreement on synthetic val: does extract_edits(extended=True)
   give each generated edit the type it was built with (PRON included)?
3. A few generated PRON pairs and forced-SPELL / natural pairs, for a skim.

Run:  .venv\\Scripts\\python.exe scripts\\14_prepare_dev.py
Writes data/processed/hiwiked_dev_split.json.
"""

from __future__ import annotations  # modern type-hint syntax

import json  # write the split
import random  # seeded shuffle
import sys  # stdout, import path
from collections import Counter  # counts
from pathlib import Path  # file paths

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # print Hindi
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # make hindi_gec importable

from hindi_gec import OTHER  # noqa: E402
from hindi_gec.config import load_config  # noqa: E402
from hindi_gec.edit_extractor import extract_edits  # noqa: E402
from hindi_gec.leakage import near_duplicates  # noqa: E402
from hindi_gec.schema import read_jsonl  # noqa: E402


def main() -> None:
    cfg = load_config()
    print("=" * 72)
    print("PREPARE TRAINING : dev split, near-duplicates, data checks")
    print("=" * 72)
    hw = list(read_jsonl(cfg.path("hiwiked_bio")))
    dev = [r for r in hw if r["split"] == "dev"]
    test = [r for r in hw if r["split"] == "test"]
    groups = sorted({r["target"] for r in dev})  # one group per target sentence
    random.Random(cfg.seed).shuffle(groups)
    n_train = round(cfg["splits"]["hiwiked_dev"]["dev_train"] * len(groups))
    side = {g: ("dev_train" if k < n_train else "dev_select") for k, g in enumerate(groups)}
    split = {r["id"]: side[r["target"]] for r in dev}
    thr = cfg["splits"]["dev_train_near_dup_jaccard"]
    train_rows = [r for r in dev if split[r["id"]] == "dev_train"]
    sel_rows = [dict(r, split="dev_select") for r in dev if split[r["id"]] == "dev_select"]
    hits = near_duplicates(train_rows, sel_rows + test, thr)
    for i in hits:
        split[i] = "dev_train_dropped"
    by_side = Counter(s for h in hits.values() for s in {x[1] for x in h})
    both = sum(any(js >= thr and jt >= thr for _, _, js, jt in h) for h in hits.values())
    print(f"  near-duplicates: {len(hits)} of {len(train_rows)} dev_train pairs have Jaccard >= {thr} "
          f"(source or target) with an evaluation pair -> dropped")
    print(f"                   partner in dev_select: {by_side['dev_select']}, in test: {by_side['test']} "
          f"(a pair can have both); source AND target >= {thr}: {both}")
    (cfg.path("reports") / "dev_train_near_dups.json").write_text(json.dumps(
        {"threshold": thr, "dropped": len(hits), "dev_train_before": len(train_rows),
         "partners": {i: [{"id": o, "split": sp, "j_source": js, "j_target": jt} for o, sp, js, jt in h]
                      for i, h in hits.items()}}, ensure_ascii=False, indent=1), encoding="utf-8")
    c = Counter(split.values())
    edits = Counter((split[r["id"]], e["type"] != OTHER) for r in dev for e in r["edits"])
    cfg.path("dev_split").write_text(json.dumps({"generated_by": r".venv\Scripts\python.exe scripts\14_prepare_dev.py",
                                                 "split": split}, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"  dev            : {len(dev)} pairs, {len(groups)} distinct targets")
    for s in ("dev_train", "dev_train_dropped", "dev_select"):
        print(f"  {s:14s} : {c[s]} pairs, {edits[(s, True)]} in-taxonomy + {edits[(s, False)]} OTHER edits")
    overlap = {r['target'] for r in dev if split[r['id']] == 'dev_train'} & \
              {r['target'] for r in dev if split[r['id']] == 'dev_select'}
    print(f"  target overlap : {len(overlap)}")

    val = [r for r in read_jsonl(cfg.path("bio_synthetic")) if r["split"] == "val"]
    agree = total = 0
    conf: Counter = Counter()
    by_type: Counter = Counter()
    for r in val:
        gold = {(tuple(e["src_span"]), e["tgt"]): e["type"] for e in r["edits"]}
        pred = {(tuple(e.src_span), e.tgt): e.type
                for e in extract_edits(r["source_tokens"], r["target_tokens"], extended=True)}
        for k, g in gold.items():
            total += 1
            by_type[(g, "n")] += 1
            if pred.get(k) == g:
                agree += 1
                by_type[(g, "ok")] += 1
            else:
                conf[(g, pred.get(k, "unmatched"))] += 1
    print(f"\n  classifier vs generator types, synthetic val (extended): {agree} / {total} = {agree / total:.1%}")
    print("    " + "  ".join(f"{t} {by_type[(t, 'ok')]}/{by_type[(t, 'n')]}"
                             for t in sorted({t for t, _ in by_type})))
    for (g, p), n in conf.most_common(5):
        print(f"    {n:4d}  generator {g:8s} -> classifier {p}")

    train = list(read_jsonl(cfg.path("bio_synthetic")))
    pron = [r for r in train if r["split"] == "train" and any(e["type"] == "PRON" for e in r["edits"])]
    forced = [r for r in train if any(e["type"] == "SPELL" and e["src"] == "उन्होने" for e in r["edits"])]
    left = sum("उन्होने" in r["target_tokens"] for r in train)
    print(f"\n  PRON pairs {len(pron)}; pairs with the उन्होंने -> उन्होने SPELL edit {len(forced)}; "
          f"targets still containing उन्होने: {left}")
    rng = random.Random(cfg.seed)
    for r in rng.sample(pron, 5) + rng.sample(forced, 2):
        e = ", ".join(f"{x['type']}: {x['src']} → {x['tgt']}" for x in r["edits"])
        print(f"    [{e}]\n      SRC: {r['source']}\n      TGT: {r['target']}")
    print(f"\n  written        : {cfg.path('dev_split')}")


if __name__ == "__main__":
    main()
