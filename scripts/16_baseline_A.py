# -*- coding: utf-8 -*-
"""System A baseline (rules + 4-gram LM), CPU only.

1. LM training data: bio_clean with the same target normalisation as the
   generator (उन्होने -> उन्होंने), MINUS every page in synthetic val: the val
   sentences are bio_clean sentences, so leaving them in would let the LM
   "remember" the answers.
2. Tune the margin on HiWikEd dev_select (all-edits F0.5), the same held-out
   real set the corrector is checked on.
3. Score synthetic val and dev_select at the chosen margin in three views
   (all edits, original 7 types, extended +PRON).
4. ``--final-test``: score HiWikEd test once with the chosen margin (run only
   in the single final evaluation, never for tuning).

Run:  .venv\\Scripts\\python.exe scripts\\16_baseline_A.py [--final-test]
Writes models/baseline_A_4gram.pkl, reports/baseline_A.json / .md and reports/predictions/<set>_baseline_A.jsonl.
"""

from __future__ import annotations  # modern type-hint syntax

import argparse  # command-line options
import json  # read/write JSON
import os  # environment
import pickle  # save the LM
import sys  # stdout, import path
import time  # timing
from collections import Counter  # counts
from pathlib import Path  # file paths

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only (can run while the GPU trains)
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # print Hindi
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # make hindi_gec importable

from hindi_gec import EXTENDED_TYPES, OTHER  # noqa: E402
from hindi_gec.baseline import NGramLM, at_margin, trajectory  # noqa: E402
from hindi_gec.config import load_config  # noqa: E402
from hindi_gec.edit_extractor import extract_edits  # noqa: E402
from hindi_gec.error_generator import normalise_forced  # noqa: E402
from hindi_gec.metrics import restrict_to_taxonomy, score_corpus  # noqa: E402
from hindi_gec.schema import read_jsonl  # noqa: E402

COMMAND = r".venv\Scripts\python.exe scripts\16_baseline_A.py"
VIEWS = (("all edits", False, False), ("original 7 types", False, True), ("extended (+PRON)", True, True))


def score(rows: list, outs: list, beta: float) -> dict:
    res = {}
    for name, ext, restrict in VIEWS:
        hr = [(extract_edits(r["source_tokens"], o, extended=ext),
               extract_edits(r["source_tokens"], r["target_tokens"], extended=ext)) for r, o in zip(rows, outs)]
        sc = score_corpus(restrict_to_taxonomy(h, rf) for h, rf in hr) if restrict else score_corpus(hr)
        c = sc.overall
        res[name] = {"tp": c.tp, "fp": c.fp, "fn": c.fn, "p": round(c.precision(), 6), "r": round(c.recall(), 6),
                     "f05": round(c.f_beta(beta), 6),
                     "by_type": {t: {"tp": x.tp, "fp": x.fp, "fn": x.fn, "f05": round(x.f_beta(beta), 6)}
                                 for t, x in sc.by_type.items() if t in EXTENDED_TYPES + (OTHER,)}}
    res["exact_match"] = round(sum(o == list(r["target_tokens"]) for r, o in zip(rows, outs)) / len(rows), 6)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-test", action="store_true", help="score HiWikEd test ONCE (final evaluation only)")
    args = ap.parse_args()
    cfg = load_config()
    b, beta = cfg["baseline"], cfg["evaluation"]["beta"]
    forced = cfg["generator"]["forced_spell"]
    right_of = {w: r for r, w in forced.items()}  # उन्होने -> उन्होंने
    t0 = time.monotonic()
    print("=" * 72)
    print("PHASE D : SYSTEM A BASELINE (rules + 4-gram stupid-backoff LM, CPU)")
    print("=" * 72)
    syn = [r for r in read_jsonl(cfg.path("bio_synthetic")) if r["split"] == "val"]
    val_pages = {r["page_title"] for r in syn}
    clean = [r for r in read_jsonl(cfg.path("bio_clean")) if r["page_title"] not in val_pages]
    sents = [normalise_forced(r["tokens"], forced)[0] for r in clean]
    lm = NGramLM(b["a_lm_order"], b["a_backoff_alpha"]).fit(sents)
    vocab = Counter(t for s in sents for t in s)
    with open(cfg.path("baseline_lm"), "wb") as f:
        pickle.dump(lm, f)
    print(f"  LM             : {len(sents)} sentences ({len(val_pages)} val pages left out), {lm.n_tokens} tokens, "
          f"{len(lm.counts)} n-grams, order {lm.order}, alpha {lm.alpha}")

    def run(rows: list, name: str) -> list:  # margin-free trajectories
        t = time.monotonic()
        tr = [trajectory(r["source_tokens"], lm, vocab, b["a_max_changes"], b["a_spell_min_count"], right_of)
              for r in rows]
        print(f"  decoded {name:16s}: {len(rows)} sentences in {time.monotonic() - t:.0f} s")
        return tr

    split = json.loads(cfg.path("dev_split").read_text(encoding="utf-8"))["split"]
    dev_sel = [r for r in read_jsonl(cfg.path("hiwiked_bio")) if split.get(r["id"]) == "dev_select"]
    tr_dev = run(dev_sel, "dev_select")
    print("\n  margin tuning on dev_select (all edits):")
    tuning = []
    for m in b["a_margins"]:
        outs = [at_margin(r["source_tokens"], st, m) for r, st in zip(dev_sel, tr_dev)]
        v = score(dev_sel, outs, beta)["all edits"]
        tuning.append({"margin": m, **{k: v[k] for k in ("tp", "fp", "fn", "p", "r", "f05")}})
        print(f"    margin {m:5.1f}: TP {v['tp']:3d} FP {v['fp']:4d} FN {v['fn']:3d}  P {v['p']:.3f}  R {v['r']:.3f}  "
              f"F0.5 {v['f05']:.3f}")
    best = max(tuning, key=lambda x: (x["f05"], x["margin"]))  # ties -> larger margin (more conservative)
    m = best["margin"]
    print(f"  chosen margin  : {m}")

    report = {"generated_by": COMMAND, "lm": {"order": lm.order, "alpha": lm.alpha, "sentences": len(sents),
                                             "tokens": lm.n_tokens, "ngrams": len(lm.counts)},
              "tuning_dev_select": tuning, "margin": m, "sets": {}}
    for name, rows, tr in (("dev_select", dev_sel, tr_dev), ("synthetic_val", syn, None)):
        tr = tr if tr is not None else run(rows, name)
        outs = [at_margin(r["source_tokens"], st, m) for r, st in zip(rows, tr)]
        report["sets"][name] = {"pairs": len(rows), **score(rows, outs, beta)}
        with open(cfg.path("predictions") / f"{name}_baseline_A.jsonl", "w", encoding="utf-8") as f:  # for 19_phase_e.py
            for r, o in zip(rows, outs):
                f.write(json.dumps({"id": r["id"], "output": " ".join(o)}, ensure_ascii=False) + "\n")
    if args.final_test:
        test = [r for r in read_jsonl(cfg.path("hiwiked_bio")) if r["split"] == "test"]
        tr = run(test, "hiwiked_test")
        outs = [at_margin(r["source_tokens"], st, m) for r, st in zip(test, tr)]
        report["sets"]["hiwiked_test"] = {"pairs": len(test), **score(test, outs, beta)}
        with open(cfg.path("predictions") / "final_test_baseline_A.jsonl", "w", encoding="utf-8") as f:
            for r, o in zip(test, outs):
                f.write(json.dumps({"id": r["id"], "source": r["source"], "target": r["target"],
                                    "output": " ".join(o)}, ensure_ascii=False) + "\n")
    out = cfg.path("reports") / "baseline_A.json"
    old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    if "hiwiked_test" in old.get("sets", {}) and not args.final_test:  # keep a final test result already made
        report["sets"]["hiwiked_test"] = old["sets"]["hiwiked_test"]
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    md = ["# System A baseline — rules + 4-gram LM", "",
          f"Produced by `{COMMAND}`. Margin {m} (tuned on dev_select, all-edits F0.5). Guard not applicable.", "",
          "| set | pairs | view | TP | FP | FN | P | R | F0.5 |", "|---|---|---|---|---|---|---|---|---|"]
    print(f"\n  {'set':18s} {'view':18s}   P      R      F0.5")
    for sname, s in report["sets"].items():
        for vname, _, _ in VIEWS:
            v = s[vname]
            md.append(f"| {sname} | {s['pairs']} | {vname} | {v['tp']} | {v['fp']} | {v['fn']} | {v['p']:.3f} | "
                      f"{v['r']:.3f} | {v['f05']:.3f} |")
            print(f"  {sname:18s} {vname:18s} {v['p']:.3f}  {v['r']:.3f}  {v['f05']:.3f}")
    (cfg.path("reports") / "baseline_A.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n  written        : {out}, reports/baseline_A.md\n  wall clock     : {(time.monotonic() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
