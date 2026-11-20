# -*- coding: utf-8 -*-
"""Step 1 — generate bio_synthetic.jsonl from bio_clean (PLAN §3).

Order of operations, and why:
1. Drop sentences IndicBART cannot reproduce (tokenizer round-trip fails, or
   longer than corrector.max_source_len sub-words). A target the model cannot
   even copy would teach it nothing but noise.
2. Split by page_title (train 95 % / val 5 %) BEFORE generating, so no
   biography contributes to both sides.
3. Generate each split with its own type quota; a single set of seen sources
   keeps sources unique across both splits.
4. Run the automated quality gate and print it as a table.
5. Write reports/synthetic_sample_30.txt for the 5-minute manual skim.

The 8 types include PRON and the forced SPELL pair उन्होंने -> उन्होने; corpus
targets that contain उन्होने are normalised first, and each such original
sentence is emitted as a real error pair (generator docstring). Forced pairs
are added on top of the type quota (generator.forced_spell_pairs).

Run:  .venv\\Scripts\\python.exe scripts\\03_generate_synthetic.py [--out FILE]
Writes data/synthetic/bio_synthetic.jsonl, reports/synthetic_quality_gate.json, reports/synthetic_sample_30.txt.
"""

from __future__ import annotations  # modern type-hint syntax

import argparse  # command-line options
import json  # write the gate report
import random  # seeded random generator
import sys  # stdout, import path
import time  # timing
from collections import Counter  # counting dictionary
from pathlib import Path  # file paths

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # print Hindi on Windows
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # make hindi_gec importable

from hindi_gec import EXTENDED_TYPES  # noqa: E402
from hindi_gec.config import load_config, set_seed  # noqa: E402
from hindi_gec.error_generator import generate_split, normalise_forced, quality_gate, split_by_page  # noqa: E402
from hindi_gec.schema import read_jsonl, write_jsonl  # noqa: E402
from hindi_gec.seq2seq_format import encode, load_tokenizer, round_trip, source_text  # noqa: E402  # tokenizer checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)  # optional output path
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg.seed)
    cc = cfg["corrector"]
    g = dict(cfg["generator"])  # generator settings
    types = EXTENDED_TYPES  # the 7 types + PRON
    t0 = time.monotonic()

    print("=" * 72)
    print("STEP 1 : SYNTHETIC ERROR GENERATION")
    print("=" * 72)
    rows = list(read_jsonl(cfg.path("bio_clean")))  # all clean sentences
    print(f"  read           : {cfg.path('bio_clean')}  ({len(rows)} sentences)")
    vocab = Counter(t for r in rows for t in r["tokens"])  # word frequencies (used by the SPELL veto)

    print("\n--- eligibility (IndicBART round-trip, length) ---")
    tok = load_tokenizer(cc["model_name"], local_files_only=True)  # IndicBART tokenizer
    keep, drop = [], Counter()
    for r in rows:
        s = " ".join(r["tokens"])
        if round_trip(tok, s, cc["lang_tag"]) != (s, s):  # tokenizer can't reproduce it exactly
            drop["round-trip fails"] += 1
        elif len(encode(tok, source_text(s, cc["lang_tag"]))) > cc["max_source_len"]:  # too many sub-words (> 64)
            drop[f"> {cc['max_source_len']} sub-words"] += 1
        else:
            keep.append(r)  # eligible
    for k, v in drop.items():
        print(f"  dropped        : {v:6d}  {k}")
    print(f"  eligible       : {len(keep)}")

    natural = {}  # id(normalised sentence) -> corruptions that restore the corpus original
    if g.get("normalise_forced_targets"):
        fixed = []
        for r in keep:
            norm, corrs = normalise_forced(r["tokens"], g["forced_spell"])
            if corrs:
                r = dict(r, tokens=norm)  # the target is the corrected sentence
                natural[id(r)] = corrs
            fixed.append(r)
        keep = fixed
        print(f"  normalised     : {len(natural)} sentences with a forced-pair misspelling "
              f"({sum(len(c) for c in natural.values())} tokens) -> targets fixed, originals kept as real pairs")
        vocab = Counter(t for r in keep for t in r["tokens"])  # frequencies after normalisation
    splits = split_by_page(keep, cfg["splits"]["synthetic"]["val"], cfg.seed)  # train/val split by biography page
    total = len(keep)
    rng = random.Random(cfg.seed)  # one random generator for the whole run (reproducible)
    seen: set = set()  # sources already generated (shared by both splits -> no duplicates across them)
    pairs, rejects = [], Counter()
    print("\n--- generating ---")
    for name in ("train", "val"):
        sents = splits[name]
        n_target = round(g["target_pairs"] * len(sents) / total)  # pairs for this split, proportional to its size
        nat = [(s, natural[id(s)]) for s in sents if id(s) in natural]  # the corpus's own misspellings in this split
        n_forced = round(g.get("forced_spell_pairs", 0) * len(sents) / total)  # forced pairs on top of the quota
        ps, rj = generate_split(sents, n_target, name, vocab, g, rng, seen, g["max_passes"], nat, n_forced)
        pairs += ps
        rejects += rj
        print(f"  {name:5s}: {len({s['page_title'] for s in sents}):5d} pages, "
              f"{len(sents):6d} sentences -> {len(ps):6d} pairs (target {n_target})")
    for i, p in enumerate(pairs, 1):
        p.id = f"syn_{i:06d}"  # give every pair an id
    out = Path(args.out) if args.out else cfg.path("bio_synthetic")  # data/synthetic/bio_synthetic.jsonl
    out.parent.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(pairs, out)
    print(f"  written        : {out}  ({n} pairs)")
    for k, v in rejects.most_common():
        print(f"  rejected       : {v:6d}  {k}")

    print("\n--- quality gate ---")
    targets = dict(g["type_distribution"])
    if g.get("forced_spell_on_top"):  # forced edits are typed SPELL but sit outside the quota: expected shares shift
        fe = sum(1 for p in pairs for e in p.edits if e.type == "SPELL" and e.src in g["forced_spell"].values())
        ne = sum(len(p.edits) for p in pairs)
        f = fe / ne
        targets = {t: v * (1 - f) + (f if t == "SPELL" else 0) for t, v in targets.items()}
        ordinary = (sum(1 for p in pairs for e in p.edits if e.type == "SPELL") - fe) / (ne - fe)
        print(f"  forced edits   : {fe} of {ne} ({f:.2%}) on top; ordinary SPELL share of the rest {ordinary:.2%} "
              f"(target {g['type_distribution']['SPELL']:.2%}); gate targets shifted accordingly")
    rep = quality_gate(pairs, targets, g["p_clean"], g["type_share_tolerance"], types)  # automated checks
    tol = g["type_share_tolerance"]
    print(f"  {'type':8s} {'count':>7s} {'share':>7s} {'target':>7s} {'diff':>7s}  ok?")
    for t in types:  # one row per type: actual share vs target
        sh, tg = rep["type_shares"][t], targets[t]
        print(f"  {t:8s} {rep['type_counts'].get(t, 0):7d} {sh:7.1%} {tg:7.1%} "
              f"{sh - tg:+7.1%}  {'yes' if abs(sh - tg) <= tol else 'NO'}")
    print(f"  {'edits':8s} {rep['edits']:7d}")
    print(f"  NONE pairs share : {rep['none_share']:.1%}  (target {g['p_clean']:.0%})  "
          f"{'yes' if rep['none_ok'] else 'NO'}")
    print(f"  errors per pair  : " + ", ".join(f"{k}: {v}" for k, v in sorted(rep['errors_per_pair'].items())))
    print(f"  no-op pairs      : {rep['noop']}")
    print(f"  duplicate sources: {rep['duplicates']}")
    print(f"  invalid spans    : {rep['bad_spans']}  (span text, undo-edits == target)")
    print(f"  page leakage     : {rep['page_leak']}  (pages in both train and val)")
    print(f"  GATE             : {'PASSED' if rep['passed'] else 'FAILED'}")
    gate_path = cfg.path("reports") / "synthetic_quality_gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gate_path, "w", encoding="utf-8") as f:  # save the gate report as JSON
        json.dump({k: (dict(v) if isinstance(v, Counter) else v) for k, v in rep.items()},
                  f, ensure_ascii=False, indent=2)

    # Manual sample: 4 per error type + 2 NONE, from train, fixed seed.
    srng = random.Random(cfg.seed + 1)  # separate seeded generator for the sample
    picks = []
    for t in types:
        pool = [p for p in pairs if p.split == "train" and p.edits and p.edits[0].type == t]  # training pairs whose first error is type t
        picks += srng.sample(pool, min(4, len(pool)))  # pick 4
    picks += srng.sample([p for p in pairs if p.split == "train" and not p.edits], 2)  # plus 2 error-free pairs
    sample_path = cfg.path("reports") / "synthetic_sample_30.txt"
    with open(sample_path, "w", encoding="utf-8") as f:  # write them for a human to read
        f.write("Step 1 manual skim — is each SOURCE a plausible human mistake, and is\n")
        f.write("TARGET (the original clean sentence) its correct fix? Mark bad lines.\n")
        f.write("=" * 72 + "\n\n")
        for i, p in enumerate(picks, 1):
            types = ", ".join(f"{e.type}: {e.src or '∅'} → {e.tgt or '∅'}" for e in p.edits) or "NONE"
            f.write(f"{i:2d}. [{types}]\n    SRC: {p.source}\n    TGT: {p.target}\n\n")
    print(f"\n  gate report    : {gate_path}")
    print(f"  manual sample  : {sample_path}  ({len(picks)} pairs)")
    print(f"  wall clock     : {(time.monotonic() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
