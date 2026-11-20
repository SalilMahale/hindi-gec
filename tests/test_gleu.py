# -*- coding: utf-8 -*-
"""GLEU on a two-sentence Hindi corpus; every n-gram count below was worked out by hand.

Sentence 1: S "वह घर गया था ।" (wrong gender), R "वह घर गई थी ।".
Sentence 2: an already-correct sentence, S = H = R = "उनका जन्म दिल्ली में हुआ ।" (6 tokens),
so its stats are num = den = 6, 5, 4, 3 for n = 1..4.
"""

from __future__ import annotations  # modern type-hint syntax

import math  # exp

import pytest  # approx

from hindi_gec.gleu import corpus_gleu, sentence_stats  # functions under test

S1, R1 = "वह घर गया था ।".split(), "वह घर गई थी ।".split()
OK = "उनका जन्म दिल्ली में हुआ ।".split()


def test_sentence_stats_half_corrected() -> None:
    # H "वह घर गई था ।": gender fixed, auxiliary still wrong.
    # n=1: shared with R {वह, घर, गई, ।} = 4, still-wrong source unigram था = 1 -> 3 / 5
    # n=2: shared {वह घर, घर गई} = 2, still-wrong "था ।" = 1                  -> 1 / 4
    # n=3: shared {वह घर गई} = 1, no source trigram left                    -> 1 / 3
    # n=4: nothing shared                                                    -> 0 / 2
    h = "वह घर गई था ।".split()
    assert sentence_stats(S1, h, R1) == [5, 5, 3, 5, 1, 4, 1, 3, 0, 2]


def test_do_nothing_stats_subtract_the_uncorrected_ngrams() -> None:
    # n=1: shared {वह, घर, ।} = 3 minus still-wrong {गया, था} = 2 -> 1; n=2: 1 - 3 -> clipped to 0
    assert sentence_stats(S1, S1, R1)[2:6] == [1, 5, 0, 4]


def test_corpus_gleu_values_and_order() -> None:
    perfect = corpus_gleu([(S1, R1, R1), (OK, OK, OK)])
    half = corpus_gleu([(S1, "वह घर गई था ।".split(), R1), (OK, OK, OK)])
    nothing = corpus_gleu([(S1, S1, R1), (OK, OK, OK)])
    assert perfect == pytest.approx(1.0)
    assert half == pytest.approx(((9 / 11) * (6 / 9) * (5 / 7) * (3 / 5)) ** 0.25)  # sums: 9/11, 6/9, 5/7, 3/5
    assert nothing == pytest.approx(((7 / 11) * (5 / 9) * (4 / 7) * (3 / 5)) ** 0.25)  # 7/11, 5/9, 4/7, 3/5
    assert perfect > half > nothing


def test_brevity_penalty_and_zero_precision() -> None:
    short = ["उनका", "जन्म", "दिल्ली", "में"]  # 4 of the 6 words: precision 1 at every n, but too short
    assert corpus_gleu([(OK, short, OK)]) == pytest.approx(math.exp(1 - 6 / 4))
    assert corpus_gleu([(S1, S1, R1)]) == 0.0  # no bigram survives the subtraction -> GLEU 0
