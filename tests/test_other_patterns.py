# -*- coding: utf-8 -*-
"""OTHER-edit patterns — one hand-checked real-style example per rule."""

from __future__ import annotations  # modern type-hint syntax

from hindi_gec.other_patterns import char_edits, pattern  # functions under test


def p(src: str, tgt: str, target: str, start: int) -> str:
    return pattern(src.split(), tgt.split(), target.split(), start)


def test_pronoun_patterns() -> None:
    assert p("उसने", "उन्होंने", "उन्होंने कई पुस्तकें लिखीं ।", 0) == "PRON-HON"  # number changed, same case
    assert p("उनके", "उनका", "उनका जन्म दिल्ली में हुआ ।", 0) == "POSS-AGR"  # genitive ending changed
    assert p("इन्होंने", "उन्होंने", "उन्होंने विवाह किया ।", 0) == "PRON-PROX"  # proximal -> distal
    assert p("जो", "जिसे", "जिसे सम्मानित किया गया ।", 0) == "PRON-OTHER"


def test_inflection_patterns_by_next_word() -> None:
    assert p("जिला", "जिले", "वे बस्ती जिले में पैदा हुए ।", 2) == "NOUN-OBL"  # followed by में
    assert p("दूसरा", "दूसरे", "वे दूसरे स्थान पर रहे ।", 1) == "AGR-ADJ"  # followed by a noun
    assert p("रचा", "रचीं", "उन्होंने कई कविताएँ रचीं ।", 3) == "AGR-OBJ"  # verb outside the paradigms, ने clause
    assert p("फिल्म", "फिल्मों", "उन्होंने कई फिल्मों में काम किया ।", 2) == "NOUN-INFL"


def test_other_patterns() -> None:
    assert p("है", "थे", "वे प्रसिद्ध कवि थे ।", 3) == "VERB-FORM"  # both verb forms, outside AUX/AGR rules
    assert p("सितंबर", "दिसंबर", "उनका जन्म 5 दिसंबर को हुआ ।", 3) == "FACT"
    assert p("गृह", "ग्रह", "यह एक ग्रह है ।", 2) == "ORTH"  # ृ -> ्र
    assert p("मौत", "मृत्यु", "उनकी मृत्यु 1950 में हुई ।", 1) == "WORD-CHOICE"
    assert p("", "भी", "वे कवि भी थे ।", 2) == "INSERT"


def test_char_edits_hand_computed() -> None:
    assert char_edits("गृह", "ग्रह")[0] == 2  # ृ replaced by ् and र inserted
    assert char_edits("कृष्णा", "कृष्ण")[0] == 1  # final ा deleted
