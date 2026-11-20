# -*- coding: utf-8 -*-
"""Near-duplicate check: Jaccard values counted by hand on Hindi sentences."""

from __future__ import annotations  # modern type-hint syntax

from hindi_gec.leakage import jaccard, near_duplicates  # functions under test


def row(i: str, split: str, src: str, tgt: str) -> dict:
    return {"id": i, "split": split, "source_tokens": src.split(), "target_tokens": tgt.split()}


def test_jaccard_by_hand() -> None:
    a = frozenset("उनका जन्म 1931 में हुआ था ।".split())
    b = frozenset("उनका जन्म 1932 में हुआ था ।".split())
    assert jaccard(a, b) == 6 / 8  # 6 shared of 8 distinct words
    assert jaccard(a, a) == 1.0


def test_near_duplicates_source_or_target() -> None:
    tr = [row("t1", "dev_train", "उन्होने 1950 में दिल्ली विश्वविद्यालय से स्नातक की उपाधि ली ।",
              "उन्होंने 1950 में दिल्ली विश्वविद्यालय से स्नातक की उपाधि ली ।"),
          row("t2", "dev_train", "वह एक प्रसिद्ध गायक था ।", "वह एक प्रसिद्ध गायक थे ।")]
    ev = [row("e1", "test", "उन्होने 1951 में दिल्ली विश्वविद्यालय से स्नातक की उपाधि ली ।",
              "उन्होंने 1951 में दिल्ली विश्वविद्यालय से स्नातक की उपाधि ली ।")]
    # t1 vs e1: 11 distinct words each, 10 shared (only the year differs) -> 10 / 12 = 0.833 on both sides
    hits = near_duplicates(tr, ev, 0.8)
    assert list(hits) == ["t1"]
    assert hits["t1"] == [("e1", "test", round(10 / 12, 4), round(10 / 12, 4))]
    assert near_duplicates(tr, ev, 0.85) == {}  # 0.833 < 0.85
