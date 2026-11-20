# -*- coding: utf-8 -*-
"""Near-duplicate check between real training pairs and the evaluation sets.

Why
---
The corrector trains on HiWikEd dev_train. HiWikEd comes from Wikipedia revision
history, so the same sentence often appears several times with small
differences: one edit, then another edit a few revisions later. The
dev_train / dev_select split already groups pairs by identical target
sentence. A sentence that differs by one or two words still lands on both
sides, though, and so does a sentence that also occurs in the test split.
Training on it would let the model "remember" an evaluation answer.

Measure
-------
Jaccard similarity of the two token SETS:

    J(A, B) = |A ∩ B| / |A ∪ B|        1.0 = same words, 0.0 = no word shared

A training pair is a near-duplicate of an evaluation pair if J ≥ threshold
on the source sentences OR on the target sentences. "Or" is the cautious
choice: a pair whose target is almost an evaluation target leaks the answer
even if the wrong versions differ.

Example: "उनका जन्म 1931 में हुआ था ।" vs "उनका जन्म 1932 में हुआ था ।" share 6 of 8
distinct words, so J = 0.75. With one more word in common (sentences of 9 words
differing in 1), J = 8 / 10 = 0.8, which counts as a near-duplicate.
"""

from __future__ import annotations  # modern type-hint syntax

from typing import Dict, FrozenSet, Iterable, List, Mapping, Sequence, Tuple  # type hints


def jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def near_duplicates(train: Sequence[Mapping], others: Iterable[Mapping], threshold: float
                    ) -> Dict[str, List[Tuple[str, str, float, float]]]:
    """train id -> [(other id, its split, J source, J target)] for every near-duplicate partner.

    Rows need id, split, source_tokens and target_tokens.
    """
    others = [(o["id"], o["split"], frozenset(o["source_tokens"]), frozenset(o["target_tokens"])) for o in others]
    hits: Dict[str, List[Tuple[str, str, float, float]]] = {}
    for r in train:
        s, t = frozenset(r["source_tokens"]), frozenset(r["target_tokens"])
        for oid, osplit, os_, ot in others:
            js, jt = jaccard(s, os_), jaccard(t, ot)
            if js >= threshold or jt >= threshold:
                hits.setdefault(r["id"], []).append((oid, osplit, round(js, 4), round(jt, 4)))
    return hits
