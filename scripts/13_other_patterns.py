# -*- coding: utf-8 -*-
"""Phase C step 1 — which OTHER patterns do real editors fix? HiWikEd **dev only**.

The test split is never read. For every dev edit typed OTHER by the edit
classifier, other_patterns.pattern() assigns a linguistic pattern; counts and
examples are printed, and each pattern is mapped to the candidate new error
type it would support (table in reports/other_patterns_dev.md). A candidate is implemented only with
>= min_support dev examples.

Also mined from dev: SPELL confusion pairs (wrong -> right) seen >= 3 times and
never the other way round, excluding verb spelling-family pairs (गये/गए), as the
data-driven SPELL candidate requires.

Run:  .venv\\Scripts\\python.exe scripts\\13_other_patterns.py
Writes reports/other_patterns_dev.json and reports/other_patterns_dev.md.
"""

from __future__ import annotations  # modern type-hint syntax

import json  # write JSON
import sys  # stdout, import path
from collections import Counter, defaultdict  # counts
from pathlib import Path  # file paths

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # print Hindi
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # make hindi_gec importable

from hindi_gec import OTHER  # noqa: E402
from hindi_gec.config import load_config  # noqa: E402
from hindi_gec.morphology import FORM_INDEX  # noqa: E402
from hindi_gec.other_patterns import pattern  # noqa: E402
from hindi_gec.schema import read_jsonl  # noqa: E402

COMMAND = r".venv\Scripts\python.exe scripts\13_other_patterns.py"
MIN_SUPPORT = 20  # drop candidates with < 20 dev examples
SPELL_MIN = 3  # data-driven SPELL pairs need >= 3 occurrences
CANDIDATE = {  # pattern -> candidate new type (table in reports/other_patterns_dev.md); None = not a grammar type we can generate
    "PRON-HON": "PRON", "POSS-AGR": "PRON", "PRON-PROX": None, "PRON-OTHER": None,
    "AGR-OBJ": "AGR-OBJ", "VERB-AGR": "AGR-ADJ/VERB (unlisted verb)", "AGR-ADJ": "AGR-ADJ", "NOUN-OBL": "NOUN-OBL",
    "NOUN-INFL": "NOUN-OBL", "VERB-FORM": None, "ORTH": "SPELL (data-driven)", "FACT": None,
    "WORD-CHOICE": None, "INSERT": None, "DELETE": None, "MULTI": None, "PUNCT": None,
}


def main() -> None:
    cfg = load_config()
    print("=" * 72)
    print("PHASE C STEP 1 : OTHER PATTERNS ON HIWIKED-BIO DEV (test not read)")
    print("=" * 72)
    dev = [r for r in read_jsonl(cfg.path("hiwiked_bio")) if r["split"] == "dev"]
    edits = [(r, e) for r in dev for e in r["edits"]]
    other = [(r, e) for r, e in edits if e["type"] == OTHER]
    print(f"  dev pairs {len(dev)}, edits {len(edits)}, OTHER {len(other)} ({len(other) / len(edits):.1%})")

    groups = defaultdict(list)
    for r, e in other:
        p = pattern(e["src"].split(), e["tgt"].split(), r["target_tokens"], e["tgt_span"][0])
        groups[p].append((r, e))
    rows = []
    print(f"\n  {'pattern':12s} {'count':>5s} {'share':>6s}  {'candidate':26s} examples")
    for p, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        pairs = Counter(f"{e['src'] or '∅'} → {e['tgt'] or '∅'}" for _, e in items)
        cand = CANDIDATE.get(p)
        rows.append({"pattern": p, "count": len(items), "share": round(len(items) / len(other), 4),
                     "candidate": cand, "supported": cand is not None and len(items) >= MIN_SUPPORT,
                     "top_pairs": pairs.most_common(8),
                     "in_context": [" ".join(r["target_tokens"]) for r, _ in items[:3]]})
        print(f"  {p:12s} {len(items):5d} {len(items) / len(other):6.1%}  {str(cand):26s} "
              + " | ".join(f"{k} ×{v}" for k, v in pairs.most_common(4)))

    by_cand = Counter()
    for row in rows:
        if row["candidate"]:
            by_cand[row["candidate"]] += row["count"]
    print("\n  candidate new types (sum of their patterns):")
    for c, n in by_cand.most_common():
        print(f"    {c:28s} {n:4d}  {'keep (>= ' + str(MIN_SUPPORT) + ')' if n >= MIN_SUPPORT else 'drop (< ' + str(MIN_SUPPORT) + ')'}")

    spell = Counter((e["src"], e["tgt"]) for _, e in edits if e["type"] == "SPELL")
    fam = lambda a, b: a in FORM_INDEX and b in FORM_INDEX  # noqa: E731  verb spelling family (गये/गए)
    mined = [(a, b, n) for (a, b), n in spell.most_common()
             if n >= SPELL_MIN and spell[(b, a)] == 0 and not fam(a, b)]
    print(f"\n  SPELL edits on dev: {sum(spell.values())} ({len(spell)} distinct pairs); "
          f"pairs >= {SPELL_MIN}x, one-directional, not verb family: {len(mined)}")
    for a, b, n in mined[:15]:
        print(f"    {a} → {b}  ×{n}")
    fam_n = sum(n for (a, b), n in spell.items() if fam(a, b))
    print(f"  (verb spelling-family SPELL edits, excluded: {fam_n})")

    rep = {"generated_by": COMMAND, "split": "dev", "dev_pairs": len(dev), "edits": len(edits),
           "other_edits": len(other), "min_support": MIN_SUPPORT, "patterns": rows,
           "candidates": dict(by_cand.most_common()),
           "spell": {"edits": sum(spell.values()), "distinct": len(spell), "family_excluded": fam_n,
                     "mined_pairs": [{"wrong": a, "right": b, "count": n} for a, b, n in mined]}}
    (cfg.path("reports") / "other_patterns_dev.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                                                 encoding="utf-8")
    md = ["# OTHER edits on HiWikEd-bio dev, by pattern", "",
          f"Produced by `{COMMAND}`. Dev only ({len(dev)} pairs; test not read). {len(other)} of {len(edits)} "
          f"dev edits ({len(other) / len(edits):.1%}) are OTHER.", "",
          "| pattern | count | share of OTHER | candidate type | top pairs |", "|---|---|---|---|---|"]
    for row in rows:
        md.append(f"| {row['pattern']} | {row['count']} | {row['share']:.1%} | {row['candidate'] or '—'} | "
                  + ", ".join(f"{k} ×{v}" for k, v in row["top_pairs"][:5]) + " |")
    md += ["", "Candidate types (sum of patterns; keep if >= 20):", ""]
    md += [f"- {c}: {n}" for c, n in by_cand.most_common()]
    md += ["", f"Data-driven SPELL pairs (>= {SPELL_MIN}x on dev, one-directional, not verb family): "
           + (", ".join(f"{a} → {b} ×{n}" for a, b, n in mined) or "none")]
    (cfg.path("reports") / "other_patterns_dev.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n  written        : reports/other_patterns_dev.json, reports/other_patterns_dev.md")


if __name__ == "__main__":
    main()
