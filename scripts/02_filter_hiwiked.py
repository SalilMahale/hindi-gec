# -*- coding: utf-8 -*-
"""Step 2 — real biography test set from HiWikEd, plus two sanity checks.

1. Clean HiWikEd exactly like bio_clean, keep pairs whose target scores
   >= threshold on the biography score (PLAN §2.3); fall back to threshold 1
   if fewer than min_pairs_required survive.
2. Leakage check: HiWikEd and bio_clean both come from Hindi Wikipedia. A
   HiWikEd target that also appears in bio_clean was (via its synthetic
   corruptions) a training target, which would inflate real-data scores.
   Those pairs are removed from the test set and counted.
3. Split dev 40 % / test 60 % (PLAN §2.4) with the fixed seed. HiWikEd has no
   page titles, so the split is by pair.
4. Extract and type the real edits; print the real-error type distribution.
5. Classifier agreement: on synthetic val, does the edit extractor recover
   the type the generator assigned by construction?

Run:  .venv\\Scripts\\python.exe scripts\\02_filter_hiwiked.py
"""

from __future__ import annotations  # modern type-hint syntax

import random  # seeded shuffle for the dev/test split
import sys  # stdout settings, import path
from collections import Counter  # counting dictionary
from pathlib import Path  # file paths

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # print Hindi on Windows console
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # make hindi_gec importable

from hindi_gec import ERROR_TYPES, OTHER  # noqa: E402  # the 7 types and the OTHER label
from hindi_gec.config import load_config  # noqa: E402
from hindi_gec.domain_filter import filter_biography, prepare_pairs  # noqa: E402  # cleaning + biography scoring
from hindi_gec.edit_extractor import extract_edits  # noqa: E402  # finds and types edits
from hindi_gec.preprocessing import load_corpus  # noqa: E402  # reads the .src/.trg files
from hindi_gec.schema import Pair, read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    cfg = load_config()
    fcfg, pcfg = cfg["domain_filter"], cfg["preprocessing"]  # domain-filter and preprocessing settings
    print("=" * 72)
    print("STEP 2 : REAL BIOGRAPHY TEST SET (HiWikEd)")
    print("=" * 72)
    raw = load_corpus(cfg.path("hiwiked_src"), cfg.path("hiwiked_trg"))  # all raw HiWikEd pairs
    print(f"  read           : {len(raw)} raw pairs")
    pairs, stats = prepare_pairs(raw, pcfg["min_tokens"], pcfg["max_tokens"])  # clean them; get rejection counts
    for k, v in stats.items():
        print(f"  dropped        : {v:6d}  {k}")
    print(f"  usable pairs   : {len(pairs)}")

    print("\n--- biography score of target (PLAN §2.3) ---")
    dist = Counter(min(sc, 5) for _, _, sc in filter_biography(pairs, fcfg, threshold=0))  # score histogram (5 = "5 or more")
    for sc in sorted(dist):
        print(f"  score {sc}{'+' if sc == 5 else ' '}       : {dist[sc]:6d}")
    th = fcfg["threshold"]  # configured threshold (2)
    bio = filter_biography(pairs, fcfg, th)  # pairs that look like biography sentences
    print(f"  threshold {th}    : {len(bio)} pairs")
    if len(bio) < fcfg["min_pairs_required"]:  # too few pairs (< 1000)?
        th = 1  # relax the threshold
        bio = filter_biography(pairs, fcfg, th)
        print(f"  < {fcfg['min_pairs_required']} -> threshold lowered to 1 : {len(bio)} pairs")

    print("\n--- leakage check against bio_clean ---")
    clean = {" ".join(r["tokens"]) for r in read_jsonl(cfg.path("bio_clean"))}  # every training sentence as a string
    leaked_tgt = [b for b in bio if " ".join(b[1]) in clean]  # test targets also in training data
    leaked_src = [b for b in bio if " ".join(b[0]) in clean]  # test sources also in training data
    print(f"  target in bio_clean : {len(leaked_tgt)}")
    print(f"  source in bio_clean : {len(leaked_src)}  (the uncorrected version survived on Wikipedia)")
    drop = {id(b) for b in leaked_tgt} | {id(b) for b in leaked_src}  # identities of pairs to remove
    bio = [b for b in bio if id(b) not in drop]  # remove leaked pairs
    print(f"  kept after removal  : {len(bio)}")

    rng = random.Random(cfg.seed)  # seeded random generator
    order = list(range(len(bio)))
    rng.shuffle(order)  # random order of pair indices
    n_dev = round(cfg["splits"]["hiwiked_bio"]["dev"] * len(bio))  # 40 % for dev
    split_of = {i: ("dev" if k < n_dev else "test") for k, i in enumerate(order)}  # first 40 % of the shuffle -> dev, rest -> test
    records = []
    for i, (src, tgt, _sc) in enumerate(bio):  # build Pair records
        records.append(Pair(id=f"hwe_{i + 1:05d}", origin="hiwiked", page_title="",
                            source=" ".join(src), target=" ".join(tgt),
                            source_tokens=src, target_tokens=tgt,
                            edits=extract_edits(src, tgt), split=split_of[i]))  # real edits found by the extractor
    out = cfg.path("hiwiked_bio")
    n = write_jsonl(records, out)  # save
    sc = Counter(r.split for r in records)  # count dev/test
    print(f"\n  written        : {out}  ({n} pairs: dev {sc['dev']}, test {sc['test']}, threshold {th})")

    print("\n--- real-error type distribution (edit extractor on hiwiked_bio) ---")
    types = Counter(e.type for r in records for e in r.edits)  # count of each type in real data
    n_e = sum(types.values())  # total real edits
    target = cfg["generator"]["type_distribution"]  # planned shares
    print(f"  {'type':8s} {'count':>6s} {'share':>7s} {'in-taxonomy':>12s} {'PLAN §3.3':>10s}")
    n_tax = n_e - types[OTHER]  # edits inside the 7 types
    for t in list(ERROR_TYPES) + [OTHER]:
        tax = f"{types[t] / n_tax:12.1%}" if t != OTHER and n_tax else f"{'':12s}"  # share among in-taxonomy edits
        plan = f"{target[t]:10.0%}" if t in target else ""  # planned share
        print(f"  {t:8s} {types[t]:6d} {types[t] / n_e:7.1%} {tax} {plan}")
    print(f"  {'edits':8s} {n_e:6d}   per pair {n_e / max(n, 1):.2f}")
    print(f"  pairs with >= 1 in-taxonomy edit : "
          f"{sum(1 for r in records if any(e.type != OTHER for e in r.edits))}")
    biggest = max(ERROR_TYPES, key=lambda t: abs(types[t] / max(n_tax, 1) - target[t]))  # type whose real share differs most from plan
    gap = types[biggest] / max(n_tax, 1) - target[biggest]
    print(f"  largest gap vs PLAN §3.3 : {biggest} {gap:+.1%} "
          f"({'> 10 points: §3.3 says adopt measured' if abs(gap) > 0.10 else 'within 10 points'})")
    print("\n  sample real edits per type:")
    for t in list(ERROR_TYPES) + [OTHER]:
        ex = [f"{e.src or '∅'} → {e.tgt or '∅'}" for r in records for e in r.edits if e.type == t][:4]  # first 4 examples
        print(f"    {t:8s}: " + " | ".join(ex))

    print("\n--- classifier agreement on synthetic val (types by construction) ---")
    agree = total = 0
    confusion: Counter = Counter()  # (generator types, extractor types) -> count, for disagreements
    for r in read_jsonl(cfg.path("bio_synthetic")):
        if r["split"] != "val" or not r["edits"]:  # only validation pairs with errors
            continue
        gold = sorted(e["type"] for e in r["edits"])  # types the generator created
        pred = sorted(e.type for e in extract_edits(r["source_tokens"], r["target_tokens"], r["page_title"]))  # types our classifier assigns
        total += 1
        agree += gold == pred  # True adds 1
        if gold != pred:
            confusion[(" ".join(gold), " ".join(pred))] += 1
    print(f"  pairs whose extracted types == generator types : {agree} / {total} = {agree / total:.1%}")
    for (g, p), c in confusion.most_common(6):  # 6 most common disagreements
        print(f"    {c:4d}  generator [{g}]  ->  extractor [{p}]")


if __name__ == "__main__":
    main()
