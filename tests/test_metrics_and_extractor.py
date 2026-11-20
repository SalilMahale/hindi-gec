# -*- coding: utf-8 -*-
"""Edit extractor, metrics and domain filter — hand-computed expected values."""

from __future__ import annotations  # modern type-hint syntax

import pytest  # testing framework

from hindi_gec.config import load_config
from hindi_gec.domain_filter import biography_score, clean_tokens
from hindi_gec.edit_extractor import classify, extract_edits, spell_key
from hindi_gec.metrics import Counts, score_corpus
from hindi_gec.schema import Edit

FCFG = load_config()["domain_filter"]  # domain-filter settings (keywords etc.)


# ------------------------------------------------------------- extractor
def S(s: str):  # helper: sentence string -> list of words
    return s.split()


def test_extract_single_replacement_span_and_type() -> None:
    #       0    1    2    3   4         5   6   7  8
    src = S("उनका जन्म 1931 में रामेश्वरम में हुई थी ।")  # wrong (feminine verb for masculine जन्म)
    tgt = S("उनका जन्म 1931 में रामेश्वरम में हुआ था ।")  # correct
    (e,) = extract_edits(src, tgt)  # exactly one edit expected
    assert (e.src_span, e.tgt_span, e.src, e.tgt, e.type) == ([6, 8], [6, 8], "हुई थी", "हुआ था", "AGR-GEN")


def test_ergative_insertion_and_deletion() -> None:
    (e,) = extract_edits(S("गांधी आंदोलन शुरू किया ।"), S("गांधी ने आंदोलन शुरू किया ।"))  # missing ने
    assert (e.src_span, e.tgt_span, e.src, e.tgt, e.type) == ([1, 1], [1, 2], "", "ने", "ERG")  # empty source span = insertion
    (e,) = extract_edits(S("वे यह पुस्तक लिखी ।"), S("उन्होंने यह पुस्तक लिखी ।"))  # वे -> उन्होंने
    assert e.type == "ERG"


def test_light_verb_gap_merged_into_one_agreement_edit() -> None:
    # की जा सकती ← किया जा सकता : two difflib opcodes, one verb-group error
    (e,) = extract_edits(S("यह कार्य की जा सकती है ।"), S("यह कार्य किया जा सकता है ।"))
    assert (e.src_span, e.src, e.tgt, e.type) == ([2, 5], "की जा सकती", "किया जा सकता", "AGR-GEN")


@pytest.mark.parametrize("src, tgt, ctx, expected", [  # (wrong, right, sentence context, expected type)
    ("था", "थे", "वे वैज्ञानिक था ।", "AGR-HON"),     # sg -> pl, honorific वे
    ("थी", "थीं", "कई फिल्में बनी थी ।", "AGR-NUM"),   # sg -> pl, no honorific
    ("थे", "था", "जन्म हुआ थे ।", "AGR-NUM"),         # pl -> sg
    ("है", "था", "1931 में हुआ है ।", "AUX"),          # tense, same number
    ("से", "में", "दिल्ली से रहे ।", "POSTP"),
    ("प्रसीद्ध", "प्रसिद्ध", "", "SPELL"),             # ी -> ि
    ("उन्होने", "उन्होंने", "", "SPELL"),              # missing anusvara
    ("गये", "गए", "", "SPELL"),                        # spelling family, same cell
    ("अच्छा", "बहुत", "", "OTHER"),
])
def test_classify(src: str, tgt: str, ctx: str, expected: str) -> None:
    assert classify(S(src), S(tgt), S(ctx)) == expected


def test_spell_key_folds_but_keeps_consonants() -> None:
    assert spell_key("ज़िला") == spell_key("जीला") == spell_key("जिला")  # nukta / vowel length ignored
    assert spell_key("दिन") != spell_key("दान")  # different vowel quality is NOT folded


# ------------------------------------------------------------- metrics
def E(a: int, b: int, tgt: str, typ: str) -> Edit:  # helper: quickly build an Edit
    return Edit(src_span=[a, b], tgt_span=[a, b], src="x", tgt=tgt, type=typ)


def test_counts_hand_computed() -> None:
    # Sentence 1: ref {A,B}, hyp {A,C}   -> TP 1, FP 1, FN 1
    # Sentence 2: ref {D},   hyp {}      -> FN 1
    # Sentence 3: ref {},    hyp {}      -> nothing
    A, B, C, D = E(0, 1, "a", "POSTP"), E(3, 4, "b", "AGR-GEN"), E(5, 6, "c", "SPELL"), E(1, 2, "d", "AGR-GEN")
    s = score_corpus([([A, C], [A, B]), ([], [D]), ([], [])])  # list of (hypothesis, reference) per sentence
    assert (s.overall.tp, s.overall.fp, s.overall.fn) == (1, 1, 2)
    # P = 1/2 = 0.5; R = 1/3; F0.5 = 1.25 * 0.5 * (1/3) / (0.25 * 0.5 + 1/3)
    #                            = 0.208333 / 0.458333 = 0.454545...
    assert s.overall.precision() == pytest.approx(0.5)  # approx = allow tiny floating-point error
    assert s.overall.recall() == pytest.approx(1 / 3)
    assert s.overall.f_beta(0.5) == pytest.approx(5 / 11)
    # Per type: POSTP TP1; SPELL FP1; AGR-GEN FN2 -> R 0
    assert (s.by_type["POSTP"].tp, s.by_type["SPELL"].fp, s.by_type["AGR-GEN"].fn) == (1, 1, 2)
    assert s.by_type["AGR-GEN"].recall() == 0.0


def test_wrong_replacement_is_one_fp_and_one_fn() -> None:
    # Same span, different correction: not a match.
    s = score_corpus([([E(2, 3, "से", "POSTP")], [E(2, 3, "में", "POSTP")])])
    assert (s.overall.tp, s.overall.fp, s.overall.fn) == (0, 1, 1)
    assert s.overall.f_beta(0.5) == 0.0


def test_restrict_to_taxonomy() -> None:
    from hindi_gec.metrics import restrict_to_taxonomy
    other = E(0, 1, "और", "OTHER")  # out-of-scope edit
    agr = E(4, 5, "था", "AGR-NUM")  # in-scope reference edit
    fp = E(7, 8, "को", "POSTP")  # system edit not in the reference
    hyp, ref = restrict_to_taxonomy([other, fp], [other, agr])
    assert ref == [agr]                       # OTHER ref edit removed
    assert hyp == [fp]                        # matching OTHER hyp edit removed, FP kept
    s = score_corpus([(hyp, ref)])
    assert (s.overall.tp, s.overall.fp, s.overall.fn) == (0, 1, 1)


def test_empty_conventions() -> None:
    c = Counts()
    assert (c.precision(), c.recall()) == (1.0, 1.0)  # nothing to do and nothing done -> perfect
    assert Counts(tp=0, fp=0, fn=3).precision() == 1.0       # made no edits
    assert Counts(tp=0, fp=0, fn=3).f_beta(0.5) == 0.0       # R = 0


# ------------------------------------------------------------- domain filter
def test_biography_score_hand_computed() -> None:
    # जन्म (+1), पिता (+1), 1931 (+1), उनका (+1) = 4
    t = S("उनका जन्म 1931 में हुआ और पिता किसान थे ।")
    assert biography_score(t, FCFG) == 4
    assert biography_score(S("चाय की दुकान पर विज्ञापन दिखाई देते हैं ।"), FCFG) == 0  # not a biography sentence


def test_clean_tokens_maps_final_full_stop_to_danda() -> None:
    assert clean_tokens("आज हम विज्ञापन युग के सीमान्त पर आ खड़े हुए हैं .")[-1] == "।"  # "." -> "।"


# ------------------------------------------------------------- type confusion
def test_type_confusion_pairs_edits_at_the_same_place() -> None:
    from hindi_gec.metrics import MISSED, SPURIOUS, type_confusion
    src = S("उनका जन्म 1931 में रामेश्वरम में हुई थी ।")
    ref = extract_edits(src, S("उनका जन्म 1931 में रामेश्वरम में हुआ था ।"))  # AGR-GEN on हुई थी
    hyp = extract_edits(src, S("उनका जन्म 1931 में रामेश्वरम में हुई थीं ।"))  # same place, other change
    assert [e.type for e in ref] == ["AGR-GEN"] and len(hyp) == 1
    assert type_confusion(hyp, ref) == [("AGR-GEN", hyp[0].type)]  # paired despite the different replacement
    assert type_confusion([], ref) == [("AGR-GEN", MISSED)]
    far = extract_edits(src, S("उनका जन्म 1931 में रामेश्वरम पर हुई थी ।"))  # POSTP at another place
    assert type_confusion(far, ref) == [("AGR-GEN", MISSED), (SPURIOUS, "POSTP")]
