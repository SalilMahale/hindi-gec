# -*- coding: utf-8 -*-
"""Pick the biography sentences out of HiWikEd (PLAN §2.3).

HiWikEd is a corpus of real corrections mined from Hindi Wikipedia's edit
history, across every topic. Our system is trained only on biography prose,
so it is evaluated on the biography part: real human errors in the same
register. There are no page titles in HiWikEd, so the domain is recognised
from the *corrected* sentence itself with a transparent score:

    score = (# distinct life-event keywords present)   जन्म, निधन, पुत्र, विवाह …
          + 1 if a 4-digit year in range appears       1931
          + 1 if an honorific pronoun appears          उन्होंने, उनका …

A sentence scoring >= threshold is "biography". Scoring the target, not the
source, means the error itself can't change whether a pair counts as a
biography.

Normalisation mirrors bio_clean exactly (``remove_noise`` -> ``normalize`` ->
``tokenize``) so the corrector sees test sentences in the same shape it was
trained on. One difference is repaired: HiWikEd ends sentences with a Latin
"." where bio_clean has the danda "।"; a final "." token becomes "।" on both
sides, so sentence-final punctuation never shows up as a spurious edit.
"""

from __future__ import annotations  # modern type-hint syntax

import re  # regular expressions
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple  # type hints

from .preprocessing import normalize, remove_noise, tokenize  # the shared cleaning steps

_YEAR = re.compile(r"^\d{4}$")  # a token that is exactly 4 digits


def clean_tokens(line: str) -> List[str]:
    """bio_clean's preprocessing, plus final "." -> "।"."""
    toks = tokenize(normalize(remove_noise(line)))  # clean -> normalise -> split into tokens
    if toks and toks[-1] == ".":  # sentence ends with an English full stop
        toks[-1] = "।"  # replace it with the Hindi danda
    return toks


def biography_score(tokens: Sequence[str], fcfg: Mapping) -> int:
    """PLAN §2.3 score of one (target) sentence."""
    present = set(tokens)  # set of words for fast lookup
    score = sum(1 for kw in set(fcfg["life_event_keywords"]) if kw in present)  # +1 for each life-event keyword found
    lo, hi = fcfg["year_range"]  # e.g. 1800..2026
    if any(_YEAR.match(t) and lo <= int(t) <= hi for t in tokens):  # contains a plausible year
        score += 1
    if present & set(fcfg["honorific_pronouns"]):  # contains उन्होंने/उनका/... (set intersection non-empty)
        score += 1
    return score


def prepare_pairs(
    raw_pairs: Iterable[Tuple[str, str]], min_tokens: int, max_tokens: int
) -> Tuple[List[Tuple[List[str], List[str]]], dict]:
    """Clean HiWikEd pairs; drop empty, unchanged, out-of-range and duplicate ones."""
    out: List[Tuple[List[str], List[str]]] = []  # kept pairs
    seen = set()  # pairs already kept
    stats: dict = {}  # rejection reason -> count

    def bump(k: str) -> None:  # helper: increase a counter
        stats[k] = stats.get(k, 0) + 1

    for src_raw, tgt_raw in raw_pairs:
        src, tgt = clean_tokens(src_raw), clean_tokens(tgt_raw)  # clean both sides
        if not src or not tgt:
            bump("empty after cleaning")
        elif src == tgt:  # cleaning removed the only difference
            bump("no edit left after cleaning")
        elif not (min_tokens <= len(tgt) <= max_tokens and min_tokens <= len(src) <= max_tokens):  # too short/long
            bump("length out of range")
        elif (key := (" ".join(src), " ".join(tgt))) in seen:  # ":=" assigns key and tests it in one step
            bump("duplicate")
        else:
            seen.add(key)
            out.append((src, tgt))  # keep the pair
    return out, stats


def filter_biography(
    pairs: Sequence[Tuple[List[str], List[str]]], fcfg: Mapping, threshold: Optional[int] = None
) -> List[Tuple[List[str], List[str], int]]:
    """Keep pairs whose target scores >= threshold; returns (src, tgt, score)."""
    th = fcfg["threshold"] if threshold is None else threshold  # use config threshold unless one is passed in
    return [(s, t, sc) for s, t in pairs if (sc := biography_score(t, fcfg)) >= th]  # score the corrected side, keep if high enough
