# -*- coding: utf-8 -*-
"""Error generator and morphology — one hand-checked case per error type.

Token indices are written out next to each sentence so every expected span
can be checked by eye.
"""

from __future__ import annotations  # modern type-hint syntax

import random  # seeded random generators for reproducible tests

from hindi_gec.error_generator import (  # functions under test
    Corruption,
    agreement_candidates,
    apply_corruptions,
    ergative_candidates,
    generate_split,
    overlaps,
    postposition_candidates,
    quality_gate,
    spelling_candidates,
    split_by_page,
    undo_edits,
)
from hindi_gec.morphology import find_verb_groups, render_group

# 0     1    2    3   4         5   6   7  8
JANM = "उनका जन्म 1931 में रामेश्वरम में हुआ था ।".split()  # "He was born in 1931 in Rameswaram."
# 0  1  2       3        4  5
VAI = "वे एक प्रसिद्ध वैज्ञानिक थे ।".split()  # "He was a famous scientist." (honorific वे ... थे)
# 0        1      2  3     4     5  6
SHIKSHA = "उन्होंने दिल्ली में शिक्षा प्राप्त की ।".split()  # "He got his education in Delhi."


# ------------------------------------------------------------- morphology
def test_feminine_plural_marked_only_on_last_element() -> None:
    toks = "फिल्में प्रदर्शित की गई थीं ।".split()
    (g,) = find_verb_groups(toks)  # exactly one verb group: की गई थीं
    assert (g.gender, g.number) == ("F", "pl")
    assert render_group(toks, g, "M", "pl") == ["किए", "गए", "थे"]  # re-inflected to masculine plural


def test_genitive_ki_is_not_a_verb() -> None:
    # की before a noun is the genitive; only हुई is a verb group.
    toks = "उनकी शिक्षा की शुरुआत दिल्ली में हुई ।".split()
    assert [(g.start, g.end) for g in find_verb_groups(toks)] == [(6, 7)]


def test_ke_lie_is_a_postposition_not_a_verb() -> None:
    toks = "उन्हें इस कार्य के लिए सम्मान दिया गया ।".split()
    assert [(g.start, g.end) for g in find_verb_groups(toks)] == [(6, 8)]  # only "दिया गया", not "लिए"


def test_y_spelling_family_is_preserved() -> None:
    toks = "उन्हें पुरस्कार दिये गये ।".split()  # writer uses y-spellings
    (g,) = find_verb_groups(toks)
    assert render_group(toks, g, "M", "sg") == ["दिया", "गया"]
    assert render_group(toks, g, "F", "sg") == ["दी", "गयी"]  # stays in the y-family (गयी, not गई)


# ------------------------------------------------------------- one per type
def test_agr_gen_flips_whole_verb_group() -> None:
    assert Corruption(6, 8, ("हुई", "थी"), "AGR-GEN") in agreement_candidates(JANM, "")  # हुआ था -> हुई थी


def test_agr_num_singular_to_plural() -> None:
    assert Corruption(6, 8, ("हुए", "थे"), "AGR-NUM") in agreement_candidates(JANM, "")  # हुआ था -> हुए थे


def test_agr_hon_plural_to_singular_after_honorific_subject() -> None:
    assert Corruption(4, 5, ("था",), "AGR-HON") in agreement_candidates(VAI, "")  # वे ... थे -> था


def test_plural_to_singular_without_honorific_is_agr_num() -> None:
    # 0  1   2   3      4   5  6   — no honorific trigger: a real plural subject
    toks = "उस समय कई फिल्में बनी थीं ।".split()
    assert Corruption(5, 6, ("थी",), "AGR-NUM") in agreement_candidates(toks, "")


def test_no_sg_to_pl_on_bare_copula_in_honorific_context() -> None:
    # 0    1   2      3    4    5  — "is" after an honorific subject: the clean
    # sentence may itself be wrong, so no AGR-NUM flip is generated from it.
    toks = "वे गौरी सेन के पुत्र है ।".split()
    assert not [c for c in agreement_candidates(toks, "") if c.type == "AGR-NUM"]


def test_aux_past_to_present_needs_past_anchor() -> None:
    assert Corruption(7, 8, ("है",), "AUX") in agreement_candidates(JANM, "")  # 1931 anchors the past -> था->है is an error
    assert not [c for c in agreement_candidates(VAI, "") if c.type == "AUX"]  # no past anchor -> no AUX error


def test_postp_substitution_and_number_guard() -> None:
    rng = random.Random(0)
    conf = {"में": {"से": 1.0}}  # always replace में with से
    assert postposition_candidates(SHIKSHA, conf, rng) == [Corruption(2, 3, ("से",), "POSTP")]
    # In JANM, में after 1931 is skipped; में after रामेश्वरम is used.
    assert postposition_candidates(JANM, conf, rng) == [Corruption(5, 6, ("से",), "POSTP")]


def test_erg_drop_from_pronoun() -> None:
    assert Corruption(0, 1, ("वे",), "ERG") in ergative_candidates(SHIKSHA)  # उन्होंने -> वे


def test_erg_delete_standalone_ne() -> None:
    # 0     1  2      3   4    5
    toks = "गांधी ने आंदोलन शुरू किया ।".split()
    (c,) = ergative_candidates(toks)
    assert c == Corruption(1, 2, (), "ERG")  # delete ने
    src, (e,) = apply_corruptions(toks, [c])
    assert src == "गांधी आंदोलन शुरू किया ।".split()
    assert (e.src_span, e.tgt_span, e.src, e.tgt) == ([1, 1], [1, 2], "", "ने")


def test_erg_insert_on_nominative_ve() -> None:
    assert ergative_candidates(VAI) == [Corruption(0, 1, ("उन्होंने",), "ERG")]  # वे -> उन्होंने (wrongly add ने)


def test_spell_matra_swap_in_content_words_only() -> None:
    # 0        1       2      3    4
    toks = "उन्होंने प्रसिद्ध पुस्तक लिखी ।".split()
    cands = spelling_candidates(toks, "", {}, 2, random.Random(0))
    assert cands == [Corruption(1, 2, ("प्रसीद्ध",), "SPELL"),
                     Corruption(2, 3, ("पूस्तक",), "SPELL")]   # लिखी is a verb: untouched


def test_spell_veto_on_common_variant_and_page_name() -> None:
    toks = "वे गाँव लौटे ।".split()
    assert spelling_candidates(toks, "", {"गांव": 177}, 2, random.Random(0)) == []  # गांव is a common real spelling -> vetoed
    toks = "पुरुषोत्तम दास टंडन ।".split()
    assert spelling_candidates(toks, "पुरुषोत्तम दास टंडन", {}, 2, random.Random(0)) == []  # never misspell the person's name


# ------------------------------------------------------------- mechanics
def test_two_edits_apply_and_undo() -> None:
    corrs = [Corruption(0, 1, ("वे",), "ERG"), Corruption(4, 6, ("प्राप्त", "किया"), "AGR-GEN")]
    src, edits = apply_corruptions(SHIKSHA, [Corruption(0, 1, ("वे",), "ERG"),
                                             Corruption(5, 6, ("किया",), "AGR-GEN")])
    assert src == "वे दिल्ली में शिक्षा प्राप्त किया ।".split()  # both errors applied
    assert undo_edits(src, edits) == SHIKSHA  # undoing gives back the original
    assert overlaps(corrs[0], Corruption(1, 2, ("x",), "SPELL"))   # adjacent counts
    assert not overlaps(corrs[0], Corruption(2, 3, ("x",), "SPELL"))  # one-word gap -> no overlap


def _toy_corpus():  # small fake corpus: 5 sentences × 4 copies, spread over 3 "pages"
    sents = [JANM, VAI, SHIKSHA, "उनकी शिक्षा दिल्ली में हुई ।".split(),
             "गांधी ने आंदोलन शुरू किया ।".split()]
    return [{"page_title": f"p{i % 3}", "tokens": t} for i, t in enumerate(sents * 4)]


def test_generated_pairs_pass_mechanical_gate() -> None:
    gcfg = {  # minimal generator config
        "p_clean": 0.2, "errors_per_sentence": {1: 0.75, 2: 0.25},
        "type_distribution": {"AGR-NUM": 0.25, "AGR-GEN": 0.20, "POSTP": 0.20, "SPELL": 0.12,
                              "AGR-HON": 0.10, "AUX": 0.08, "ERG": 0.05},
        "postp_confusion": {"में": {"पर": 0.5, "से": 0.5}}, "spell_veto_min_count": 2,
    }
    pairs, _ = generate_split(_toy_corpus(), 20, "train", {}, gcfg, random.Random(1), set())  # generate 20 pairs
    rep = quality_gate(pairs, gcfg["type_distribution"], 0.2, 1.0)  # tolerance 1.0 = ignore share checks on this tiny set
    assert rep["noop"] == 0 and rep["duplicates"] == 0 and rep["bad_spans"] == 0


def test_split_by_page_has_no_leak() -> None:
    sp = split_by_page(_toy_corpus(), 0.34, seed=42)
    assert not ({s["page_title"] for s in sp["train"]} & {s["page_title"] for s in sp["val"]})  # no page in both splits
    assert sp["train"] and sp["val"]  # both splits non-empty


# ------------------------------------------------------------- genitive swaps, spelling vetoes
from hindi_gec.error_generator import genitive_candidates  # noqa: E402
from hindi_gec.edit_extractor import extract_edits  # noqa: E402


def test_genitive_ka_to_ki() -> None:
    # 0    1  2    3    4   5
    toks = "राम का जन्म दिल्ली में हुआ ।".split()
    assert genitive_candidates(toks, {"का": {"की": 1.0}}, random.Random(0)) == \
        [Corruption(1, 2, ("की",), "POSTP")]  # का -> की


def test_genitive_ke_to_ka_only_before_compound_head() -> None:
    conf = {"के": {"का": 1.0}}
    # के बाद: compound head -> का बाद is certainly wrong
    toks = "विवाह के बाद वे दिल्ली गए ।".split()
    assert genitive_candidates(toks, conf, random.Random(0)) == [Corruption(1, 2, ("का",), "POSTP")]
    # राम के पुत्र: "Ram's sons" is correct Hindi, so no के -> का here
    assert genitive_candidates("राम के पुत्र थे ।".split(), conf, random.Random(0)) == []


def test_verbal_ki_is_not_a_genitive_candidate() -> None:
    # प्राप्त की । : की is the verb, followed by the danda
    toks = "उन्होंने शिक्षा प्राप्त की ।".split()
    assert genitive_candidates(toks, {"की": {"का": 1.0}}, random.Random(0)) == []


def test_genitive_swap_is_extracted_as_postp() -> None:
    (e,) = extract_edits("राम की जन्म दिल्ली में हुआ ।".split(), "राम का जन्म दिल्ली में हुआ ।".split())
    assert (e.src, e.tgt, e.type) == ("की", "का", "POSTP")  # genitive swaps are typed POSTP


def test_spell_min_word_count_skips_rare_words() -> None:
    toks = "उन्होंने प्रसिद्ध पुस्तक लिखी ।".split()
    vocab = {"प्रसिद्ध": 900, "पुस्तक": 3}  # word frequencies
    cands = spelling_candidates(toks, "", vocab, 2, random.Random(0), min_word_count=5)
    assert cands == [Corruption(1, 2, ("प्रसीद्ध",), "SPELL")]          # पुस्तक (3) skipped


def test_ratio_veto_allows_typo_blocks_variant() -> None:
    # शीक्षा: 3 vs 1232 (0.24 %) -> a corpus typo, allowed under the ratio rule
    toks = "उन्होंने शिक्षा पूरी की ।".split()
    vocab = {"शिक्षा": 1232, "शीक्षा": 3}
    def at_1(cs):                                   # पूरी (index 2) has its own candidates
        return [c for c in cs if c.start == 1]
    assert at_1(spelling_candidates(toks, "", vocab, 2, random.Random(0))) == []          # count veto only: vetoed
    assert at_1(spelling_candidates(toks, "", vocab, 2, random.Random(0), veto_ratio=0.05)) == \
        [Corruption(1, 2, ("शीक्षा",), "SPELL")]  # with the ratio veto: allowed
    # गांव: 177 vs गाँव 285 (62 %) -> a real variant, still vetoed
    toks = "वे गाँव लौटे ।".split()
    assert spelling_candidates(toks, "", {"गाँव": 285, "गांव": 177}, 2, random.Random(0), veto_ratio=0.05) == []
