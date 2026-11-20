# -*- coding: utf-8 -*-
"""GLEU for grammatical error correction (Napoles et al., 2015; revised 2016).

Why a second metric
-------------------
Edit-level F0.5 only credits an edit whose span and replacement exactly match
the reference edit. GLEU compares n-grams of whole sentences, so it also gives
partial credit (the right verb with a slightly different span) and it
punishes leaving a wrong n-gram in place. It is the BLEU idea adapted to GEC:
BLEU only compares the output H with the reference R. GLEU also looks at the
source S and subtracts the n-grams that were wrong in S and are still wrong in H.

Formula (one reference; n = 1..4; counts summed over the corpus)
---------------------------------------------------------------
For every sentence and every n:

    match_n   = |ngrams(H) ∩ ngrams(R)|                  n-grams the output shares with the reference
    stale_n   = |ngrams(H) ∩ (ngrams(S) − ngrams(R))|     source n-grams the reference changed, still in H
    num_n     = max(0, match_n − stale_n)
    den_n     = max(1, len(H) − n + 1)                   number of n-grams in H

∩ and − are multiset (Counter) operations. Summed over sentences:

    p_n  = Σ num_n / Σ den_n
    BP   = 1 if c ≥ r else exp(1 − r / c)                c = Σ len(H), r = Σ len(R)
    GLEU = BP · exp( (1/4) · Σ_n log p_n )

BP (brevity penalty) stops a system from winning by deleting words. If any
p_n is 0, GLEU is 0, as in the official gleu.py (cnap/gec-ranking).

GLEU of the unchanged source (H = S) is the "do nothing" baseline. A system
is useful only if it scores above that.
"""

from __future__ import annotations  # modern type-hint syntax

import math  # log / exp
from collections import Counter  # multiset of n-grams
from typing import Iterable, List, Sequence, Tuple  # type hints

ORDER = 4


def ngrams(tokens: Sequence[str], n: int) -> Counter:  # multiset of the n-grams of one sentence
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def sentence_stats(src: Sequence[str], hyp: Sequence[str], ref: Sequence[str], order: int = ORDER) -> List[int]:
    """[len(H), len(R), num_1, den_1, …, num_order, den_order] for one sentence."""
    stats = [len(hyp), len(ref)]
    for n in range(1, order + 1):
        h, r, s = ngrams(hyp, n), ngrams(ref, n), ngrams(src, n)
        match = sum((h & r).values())
        stale = sum((h & (s - r)).values())
        stats += [max(0, match - stale), max(1, len(hyp) - n + 1)]
    return stats


def corpus_gleu(triples: Iterable[Tuple[Sequence[str], Sequence[str], Sequence[str]]], order: int = ORDER) -> float:
    """GLEU over (source, hypothesis, reference) token triples, statistics summed before dividing."""
    tot = [0] * (2 + 2 * order)
    for s, h, r in triples:
        tot = [a + b for a, b in zip(tot, sentence_stats(s, h, r, order))]
    c, r = tot[0], tot[1]
    nums, dens = tot[2::2], tot[3::2]
    if c == 0 or any(x == 0 for x in nums):
        return 0.0
    bp = 1.0 if c >= r else math.exp(1 - r / c)
    return bp * math.exp(sum(math.log(a / b) for a, b in zip(nums, dens)) / order)
