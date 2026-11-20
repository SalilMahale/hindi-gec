# -*- coding: utf-8 -*-
"""Example chips for the web app (one per error type, PRON included) + the correct-sentence bank.

1. Chips. Candidates come from synthetic val (never trained on): exactly one
   generator edit, of that type, that the 8-type edit classifier also gives
   that type, at most app.example_max_tokens tokens. The corrector is run on
   them *live*; a chip must be corrected exactly (output == target). A chip
   already in reports/app_examples.json is kept if its target is a synthetic
   val sentence (val pages are never trained on), the classifier gives its one
   edit that type, and it is still corrected exactly, so the demo stays
   familiar; otherwise one exact candidate is chosen with the fixed seed.
2. Bank. reports/demo_sentence_bank.txt holds 5 correct sentences (the
   generator's clean 20 %). They are re-checked: the corrector must leave
   every one unchanged. If the file is missing, 5 are drawn from the short
   error-free val sentences the corrector leaves unchanged.

Run:  .venv\\Scripts\\python.exe scripts\\10_app_examples.py
Writes reports/app_examples.json (and reports/demo_sentence_bank.txt if it was missing).
"""

from __future__ import annotations  # modern type-hint syntax

import json  # read/write JSON
import random  # seeded choice
import sys  # stdout, import path
from pathlib import Path  # file paths

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # print Hindi
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # make hindi_gec importable

from hindi_gec import EXTENDED_TYPES  # noqa: E402
from hindi_gec.config import load_config  # noqa: E402
from hindi_gec.edit_extractor import extract_edits  # noqa: E402
from hindi_gec.inference import Corrector  # noqa: E402
from hindi_gec.schema import read_jsonl  # noqa: E402

LABELS = {"AGR-GEN": "लिंग", "AGR-NUM": "वचन", "AGR-HON": "आदर", "POSTP": "परसर्ग", "ERG": "ने", "AUX": "काल",
          "SPELL": "वर्तनी", "PRON": "सर्वनाम"}  # short Hindi chip labels
N_BANK = 5  # correct sentences in the bank


def run(corr: Corrector, sources: list) -> list:  # guarded corrector output text for each source sentence
    return [r.corrected for r in corr.correct_batch(sources, batch_size=32)]


def main() -> None:
    cfg = load_config()
    max_tok = cfg["app"]["example_max_tokens"]
    rng = random.Random(cfg.seed)
    print("=" * 72)
    print(f"APP EXAMPLES : {cfg['app']['model_label']} ({cfg['paths']['model']})")
    print("=" * 72)
    corr = Corrector(cfg)
    names = corr.names  # same gazetteer as the live app

    val = [r for r in read_jsonl(cfg.path("bio_synthetic")) if r["split"] == "val"
           and len(r["source_tokens"]) <= max_tok]  # short val pairs
    val_targets = {r["target"] for r in read_jsonl(cfg.path("bio_synthetic")) if r["split"] == "val"}
    old = {c["type"]: c for c in (json.loads(cfg.path("app_examples").read_text(encoding="utf-8"))
                                  if cfg.path("app_examples").exists() else [])}
    chips = []
    for t in EXTENDED_TYPES:
        cands = []
        for r in val:
            if len(r["edits"]) != 1 or r["edits"][0]["type"] != t:
                continue
            ref = extract_edits(r["source_tokens"], r["target_tokens"], names=names, extended=True)  # as the app types it
            if len(ref) == 1 and ref[0].type == t:
                cands.append((r, ref[0]))
        if t in old and old[t]["target"] in val_targets:  # the current chip: a val sentence (never trained on)
            o = old[t]
            ref = extract_edits(o["source"].split(), o["target"].split(), names=names, extended=True)
            if len(ref) == 1 and ref[0].type == t:
                cands.insert(0, ({"source": o["source"], "target": o["target"], "id": o["id"]}, ref[0]))
        outs = run(corr, [r["source"] for r, _ in cands])
        exact = [(r, e) for (r, e), o in zip(cands, outs) if o == r["target"]]
        keep = [(r, e) for r, e in exact if t in old and r["source"] == old[t]["source"]]
        r, e = keep[0] if keep else rng.choice(exact)
        how = "kept" if keep else ("re-picked" if t in old else "new")
        print(f"  {t:8s}: {len(cands):4d} short candidates, {len(exact):4d} corrected exactly -> {how}")
        chips.append({"label": LABELS[t], "type": t, "source": r["source"], "target": r["target"],
                      "edit": f"{e.src or '∅'} → {e.tgt or '∅'}", "id": r["id"]})
    cfg.path("app_examples").write_text(json.dumps(chips, ensure_ascii=False, indent=2), encoding="utf-8")

    bank_path = cfg.path("demo_sentence_bank")
    if bank_path.exists():
        bank = [l.strip() for l in bank_path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    else:
        clean = [r for r in val if not r["edits"]]
        kept = [r for r, o in zip(clean, run(corr, [r["source"] for r in clean])) if o == r["source"]]
        bank = [r["source"] for r in rng.sample(kept, N_BANK)]
        bank_path.write_text("# Correct sentences (synthetic val, no error) the corrector leaves unchanged.\n"
                             "# Written by scripts/10_app_examples.py.\n" + "".join(s + "\n" for s in bank),
                             encoding="utf-8")
    outs = run(corr, bank)
    for s, o in zip(bank, outs):
        print(f"  bank {'unchanged' if o == s else 'CHANGED -> ' + o}: {s}")
    if any(o != s for s, o in zip(bank, outs)):
        raise SystemExit("a bank sentence was changed; stop and investigate")

    for c in chips:
        print(f"\n  [{c['type']}] {c['edit']}\n    SRC: {c['source']}\n    TGT: {c['target']}")
    print(f"\n  written        : {cfg.path('app_examples')}  ({len(chips)} chips); bank {len(bank)}/{len(bank)} unchanged")


if __name__ == "__main__":
    main()
