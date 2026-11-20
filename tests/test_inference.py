# -*- coding: utf-8 -*-
"""Hallucination guard and input preparation — hand-computed values."""

from __future__ import annotations  # modern type-hint syntax

from hindi_gec.inference import (edit_confidence, group_pieces, guard_rejects, prepare_input,  # functions under test
                                  split_sentences, token_edit_distance)

SRC = "उनका जन्म 1931 में रामेश्वरम में हुई थी ।".split()     # 9 tokens


def test_token_edit_distance_hand_computed() -> None:
    assert token_edit_distance(SRC, SRC) == 0  # identical -> 0 edits
    # two substitutions: हुई->हुआ, थी->था
    assert token_edit_distance(SRC, "उनका जन्म 1931 में रामेश्वरम में हुआ था ।".split()) == 2
    # one deletion
    assert token_edit_distance("गांधी ने आंदोलन किया".split(), "गांधी आंदोलन किया".split()) == 1
    assert token_edit_distance([], ["क", "ख"]) == 2  # empty -> two insertions


def test_guard_limit_is_max_of_count_and_ratio() -> None:
    # 9 tokens: limit = max(3, 0.25 * 9 = 2.25) = 3
    four = "हम कल 1931 से दिल्ली में हुई थी ।".split()           # 4 substitutions
    assert token_edit_distance(SRC, four) == 4
    assert guard_rejects(SRC, four, 3, 0.25)  # 4 > 3 -> rejected
    two ="उनका जन्म 1931 में रामेश्वरम में हुआ था ।".split()
    assert not guard_rejects(SRC, two, 3, 0.25)  # 2 <= 3 -> accepted
    # 20 tokens: limit = max(3, 5.0) = 5 -> 5 edits pass, 6 are rejected
    long = [f"w{i}" for i in range(20)]  # 20 dummy words
    five = ["x"] * 5 + long[5:]  # first 5 words replaced
    six = ["x"] * 6 + long[6:]  # first 6 words replaced
    assert not guard_rejects(long, five, 3, 0.25)
    assert guard_rejects(long, six, 3, 0.25)


def test_prepare_input_adds_missing_danda() -> None:
    assert prepare_input("उनका जन्म 1931 में हुआ था")[-1] == "।"  # danda added when missing
    assert prepare_input("उनका जन्म 1931 में हुआ था।").count("।") == 1  # not doubled when present


def test_split_sentences_on_danda_question_exclamation() -> None:
    para = "उनका जन्म 1931 में हुआ था। वे वैज्ञानिक थे ? क्या अद्भुत जीवन था !  उन्होंने विवाह नहीं किया"
    assert split_sentences(para) == [
        "उनका जन्म 1931 में हुआ था।",  # danda glued to the word
        "वे वैज्ञानिक थे ?",  # spaced question mark
        "क्या अद्भुत जीवन था !",  # exclamation; the double space after it is dropped
        "उन्होंने विवाह नहीं किया",  # last piece without an end mark is kept
    ]
    assert split_sentences("   ") == []  # nothing but spaces -> no sentences
    assert split_sentences("वे आए।।") == ["वे आए।।"]  # repeated end marks stay with their sentence

def test_group_pieces_word_starts_at_underscore() -> None:
    # SentencePiece pieces for "उनका जन्म हुआ ।": "▁" marks the start of a word
    pieces = ["▁उन", "का", "▁जन्म", "▁हु", "आ", "▁", "।"]
    assert group_pieces(pieces) == [[0, 1], [2], [3, 4], [5, 6]]  # "▁" alone + "।" make the word "।"


def test_edit_confidence_is_mean_subword_probability() -> None:
    word_probs = [[0.9, 0.7], [0.5], [0.2, 0.4], [1.0]]  # 4 output words
    assert edit_confidence([1, 2], word_probs, 0.95) == 0.5  # one word, one subword
    assert abs(edit_confidence([1, 3], word_probs, 0.95) - (0.5 + 0.2 + 0.4) / 3) < 1e-12  # mean over 3 subwords
    assert abs(edit_confidence([2, 2], word_probs, 0.95) - 0.3) < 1e-12  # deletion: the next word's subwords
    assert edit_confidence([4, 4], word_probs, 0.95) == 0.95  # deletion at the end: p(</s>)