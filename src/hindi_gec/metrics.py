# -*- coding: utf-8 -*-
"""Edit-level precision, recall and F0.5 (PLAN §5.2).

An edit is identified by (source span, replacement tokens). For every
sentence, the reference edits R come from extract_edits(source, target) and
the hypothesis edits H from extract_edits(source, system_output):

    TP = |H ∩ R|        edits the system made that the reference also made
    FP = |H \\ R|        changes the system should not have made
    FN = |R \\ H|        errors the system missed

Counts are summed over the corpus before dividing (micro-average), then

    P   = TP / (TP + FP)          of what it changed, how much was right
    R   = TP / (TP + FN)          of what was wrong, how much it fixed
    F_β = (1 + β²) · P · R / (β² · P + R)

β < 1 weights precision more heavily; β = 0.5 makes precision count about
twice as much as recall, the GEC standard: wrongly "correcting" good text
costs a user more than leaving an error in.

Conventions for empty sets (as in ERRANT/M2): with no hypothesis edits
P = 1.0; with no reference edits R = 1.0; F is 0.0 if P + R weighting is 0.

Per type: a matched edit counts as TP for its reference type; an unmatched
hypothesis edit is FP for the type the classifier gives it; an unmatched
reference edit is FN for its reference type.
"""

from __future__ import annotations  # modern type-hint syntax

from collections import defaultdict  # dict that creates a default value for missing keys
from dataclasses import dataclass, field  # simple record classes
from typing import Dict, Iterable, List, Sequence, Tuple  # type hints

from .schema import Edit  # Edit record

EditKey = Tuple[int, int, str]          # (src start, src end, replacement text)


def edit_key(e: Edit) -> EditKey:  # two edits are "the same" if they change the same words to the same text
    return (e.src_span[0], e.src_span[1], e.tgt)


@dataclass
class Counts:  # true positives, false positives, false negatives
    tp: int = 0  # correct corrections
    fp: int = 0  # wrong/unnecessary corrections
    fn: int = 0  # missed errors

    def precision(self) -> float:  # P = TP / (TP + FP)
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0  # 1.0 if the system made no edits

    def recall(self) -> float:  # R = TP / (TP + FN)
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0  # 1.0 if there were no errors to find

    def f_beta(self, beta: float) -> float:  # F_beta: weighted harmonic mean of P and R
        p, r = self.precision(), self.recall()
        b2 = beta * beta  # beta squared (0.25 for F0.5)
        return (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) else 0.0  # formula; 0 if denominator is 0


@dataclass
class Score:  # totals for the whole corpus
    overall: Counts = field(default_factory=Counts)  # counts over all edits
    by_type: Dict[str, Counts] = field(default_factory=lambda: defaultdict(Counts))  # counts per error type (auto-created)


def score_sentence(hyp: Sequence[Edit], ref: Sequence[Edit], score: Score) -> None:
    """Add one sentence's TP/FP/FN to ``score`` (overall and per type)."""
    ref_by_key = {edit_key(e): e for e in ref}  # reference (gold) edits by key
    hyp_by_key = {edit_key(e): e for e in hyp}  # system (hypothesis) edits by key
    for k, e in hyp_by_key.items():  # each system edit
        if k in ref_by_key:  # gold has the same edit
            score.overall.tp += 1  # true positive
            score.by_type[ref_by_key[k].type].tp += 1  # credited to the gold edit's type
        else:  # gold doesn't have it
            score.overall.fp += 1  # false positive
            score.by_type[e.type].fp += 1  # charged to the type our classifier gives it
    for k, e in ref_by_key.items():  # each gold edit
        if k not in hyp_by_key:  # system missed it
            score.overall.fn += 1  # false negative
            score.by_type[e.type].fn += 1


def restrict_to_taxonomy(hyp: Sequence[Edit], ref: Sequence[Edit], other: str = "OTHER"
                         ) -> Tuple[List[Edit], List[Edit]]:
    """Drop reference edits outside the 7 types, and system edits identical to them.

    Out-of-taxonomy real edits (word choice, noun plurals …) are never trained
    on, so counting them as misses measures scope, not quality. A system edit
    that happens to reproduce one is neither rewarded nor punished. Every
    other system edit is kept, so false positives still count.
    """
    other_keys = {edit_key(e) for e in ref if e.type == other}  # keys of gold edits typed OTHER
    return ([e for e in hyp if edit_key(e) not in other_keys],  # system edits minus those matching OTHER gold edits
            [e for e in ref if e.type != other])  # gold edits minus OTHER


def score_corpus(pairs: Iterable[Tuple[Sequence[Edit], Sequence[Edit]]]) -> Score:
    """Micro-averaged score over (hypothesis edits, reference edits) pairs."""
    s = Score()  # start from zero
    for hyp, ref in pairs:  # each sentence
        score_sentence(hyp, ref, s)  # add its counts
    return s  # totals (P/R/F computed from totals = micro-average)


def table_rows(score: Score, beta: float, types: Sequence[str]) -> List[List[str]]:
    """Rows: name, TP, FP, FN, P, R, F_beta — overall first, then each type."""
    def row(name: str, c: Counts) -> List[str]:  # format one table row as strings
        return [name, str(c.tp), str(c.fp), str(c.fn),
                f"{c.precision():.3f}", f"{c.recall():.3f}", f"{c.f_beta(beta):.3f}"]  # 3 decimal places
    rows = [row("overall", score.overall)]  # first row = overall
    for t in types:  # then one row per error type
        if t in score.by_type:  # only types that occurred
            rows.append(row(t, score.by_type[t]))
    return rows


MISSED, SPURIOUS = "(missed)", "(none)"  # confusion-matrix labels for an unmatched edit


def _touch(a: Sequence[int], b: Sequence[int]) -> bool:
    """Do two source spans [start, end) refer to the same place? An insertion has start == end."""
    if a[0] < b[1] and b[0] < a[1]:  # proper overlap
        return True
    if a[0] == a[1]:  # a is an insertion inside or at the edge of b
        return b[0] <= a[0] <= b[1]
    return b[0] == b[1] and a[0] <= b[0] <= a[1]


def type_confusion(hyp: Sequence[Edit], ref: Sequence[Edit]) -> List[Tuple[str, str]]:
    """(true type, predicted type) cells for one sentence, for a type confusion matrix.

    Unlike F0.5, which needs an exact match, an edit is paired here with any edit
    at the same place (touching source spans). So a system that fixed the
    right word but gave it the wrong replacement still shows which type it
    "thought" the error was. A reference edit with no partner is (type, MISSED);
    a system edit with no partner is (SPURIOUS, type).
    """
    cells, used = [], set()
    for r in ref:
        partner = next((k for k, h in enumerate(hyp) if k not in used and _touch(r.src_span, h.src_span)), None)
        if partner is None:
            cells.append((r.type, MISSED))
        else:
            used.add(partner)
            cells.append((r.type, hyp[partner].type))
    cells += [(SPURIOUS, h.type) for k, h in enumerate(hyp) if k not in used]
    return cells
