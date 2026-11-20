# -*- coding: utf-8 -*-
"""System A baseline: stupid-backoff scores hand-computed on a two-sentence corpus."""

from __future__ import annotations  # modern type-hint syntax

import math  # log

from hindi_gec.baseline import NGramLM, at_margin, candidates, correct, trajectory  # functions under test

CORPUS = [["वे", "कवि", "थे"], ["वे", "लेखक", "थे"]]
# bigram LM: unigrams वे 2, कवि 1, लेखक 1, थे 2, </s> 2, <s> 2 (padding context); N = 8 tokens, V = 6
LM = NGramLM(order=2, alpha=0.4).fit(CORPUS)


def test_seen_bigram_is_relative_frequency() -> None:
    assert math.isclose(LM.logscore("कवि", ("वे",)), math.log(1 / 2))  # c(वे कवि) / c(वे)
    assert math.isclose(LM.logscore("वे", ("<s>",)), math.log(2 / 2))


def test_backoff_and_unknown_word() -> None:
    assert math.isclose(LM.logscore("थे", ("वे",)), math.log(0.4) + math.log(2 / 8))  # α · c(थे) / N
    assert math.isclose(LM.logscore("क़लम", ("वे",)), math.log(0.4) + math.log(1 / (8 + 6)))  # α · 1 / (N + V)


def test_sentence_score_sums_logs() -> None:
    # log S(वे|<s>) + log S(कवि|वे) + log S(थे|कवि) + log S(</s>|थे) = log 1 + log ½ + log 1 + log 1
    assert math.isclose(LM.sentence_score(["वे", "कवि", "थे"]), math.log(0.5))


def test_candidates_cover_the_error_types() -> None:
    cands = candidates("वे कवि था ।".split(), {}, 5, {})
    assert (2, 3, ("थे",)) in cands  # AGR: the verb group re-rendered to masculine plural
    assert (2, 3, ("है",)) in cands  # AUX: tense swap
    assert (0, 1, ("उन्होंने",)) in cands  # ERG: वे -> उन्होंने
    p = candidates("उनका जन्म दिल्ली पर हुआ ।".split(), {}, 5, {"उन्होने": "उन्होंने"})
    assert (3, 4, ("में",)) in p and (0, 1, ("उनकी",)) in p and (0, 1, ("उसका",)) in p


def test_correct_accepts_only_above_margin() -> None:
    lm = NGramLM(order=3, alpha=0.4).fit([["वे", "कवि", "थे", "।"]] * 20 + [["वह", "कवि", "था", "।"]] * 5)
    wrong = "वे कवि था ।".split()
    assert correct(wrong, lm, {}, 0.0, 2, 5, {}) == "वे कवि थे ।".split()
    steps = trajectory(wrong, lm, {}, 2, 5, {})
    assert at_margin(wrong, steps, steps[0][0] + 1) == wrong  # a margin above the best gain changes nothing
