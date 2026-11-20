# -*- coding: utf-8 -*-
"""Score a system's outputs the same way everywhere: guard, then edit-level P / R / F0.5 in three views.

The three views answer three different questions about the same outputs:

* **all edits**: every reference edit counts, including the ones outside the
  taxonomy (word choice, noun plurals …). This is the honest headline: how much
  of what human editors fixed did the system fix?
* **original 7 types**: reference edits typed OTHER by the 7-type classifier
  are left out, and so are system edits identical to them (they are neither
  rewarded nor punished). Every other system edit still counts, so false
  positives are not hidden (metrics.restrict_to_taxonomy).
* **extended (+PRON)**: as above, but with the 8-type classifier, so pronoun
  edits count as an error type instead of OTHER. The model is trained on these 8.

The hallucination guard is applied to raw outputs before scoring, exactly as
in the app: an output that changes more than max(3, 25 % of the words) is
replaced by the unchanged input.
"""

from __future__ import annotations  # modern type-hint syntax

from typing import Any, Dict, List, Sequence, Tuple  # type hints

from . import EXTENDED_TYPES, OTHER  # the 8 types + OTHER
from .edit_extractor import extract_edits  # (source, output) -> typed edits
from .inference import guard_rejects  # hallucination guard
from .metrics import restrict_to_taxonomy, score_corpus  # edit-level scoring

VIEWS: Tuple[Tuple[str, bool, bool], ...] = (  # (name, 8-type classifier?, leave out OTHER references?)
    ("all edits", False, False),
    ("original 7 types", False, True),
    ("extended (+PRON)", True, True),
)


def decode_all(corr: Any, rows: Sequence[Dict], batch: int = 32) -> List[List[str]]:
    """Raw best outputs (tokens) for every row, decoded in length-sorted batches (less padding)."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["source_tokens"]))
    outs: List[List[str]] = [[] for _ in rows]
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        for i, (toks, _c) in zip(idx, corr.generate([rows[i]["source_tokens"] for i in idx])):
            outs[i] = toks
    return outs


def guard_outputs(rows: Sequence[Dict], raws: Sequence[Sequence[str]], max_edits: int, max_ratio: float
                  ) -> Tuple[List[List[str]], int]:
    """(final outputs, number rejected by the guard): a rejected output becomes the unchanged source."""
    finals, n = [], 0
    for r, o in zip(rows, raws):
        if guard_rejects(r["source_tokens"], o, max_edits, max_ratio):
            finals.append(list(r["source_tokens"]))
            n += 1
        else:
            finals.append(list(o))
    return finals, n


def views(rows: Sequence[Dict], finals: Sequence[Sequence[str]], beta: float) -> Dict[str, Dict]:
    """TP / FP / FN, P / R / F_beta and per-type counts in the three views."""
    out = {}
    for name, ext, restrict in VIEWS:
        hr = [(extract_edits(r["source_tokens"], o, extended=ext),
               extract_edits(r["source_tokens"], r["target_tokens"], extended=ext)) for r, o in zip(rows, finals)]
        sc = score_corpus(restrict_to_taxonomy(h, rf) for h, rf in hr) if restrict else score_corpus(hr)
        c = sc.overall
        out[name] = {"tp": c.tp, "fp": c.fp, "fn": c.fn, "p": round(c.precision(), 6), "r": round(c.recall(), 6),
                     "f05": round(c.f_beta(beta), 6),
                     "by_type": {t: {"tp": x.tp, "fp": x.fp, "fn": x.fn, "f05": round(x.f_beta(beta), 6)}
                                 for t, x in sc.by_type.items() if t in EXTENDED_TYPES + (OTHER,)}}
    return out


def exact_match(rows: Sequence[Dict], finals: Sequence[Sequence[str]]) -> float:  # share of outputs equal to the target
    return sum(list(o) == list(r["target_tokens"]) for r, o in zip(rows, finals)) / len(rows)
