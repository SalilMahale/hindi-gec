# -*- coding: utf-8 -*-
"""One test per explanation template, on real Hindi sentences; expected strings written by hand."""

from __future__ import annotations  # modern type-hint syntax

from hindi_gec.edit_extractor import extract_edits  # edits to explain
from hindi_gec.explain import controller, explain, verb_cell  # functions under test

NAMES = frozenset({"लाल", "बहादुर", "शास्त्री"})


def why(src: str, tgt: str) -> dict:  # explanation of the single edit between two sentences
    s, t = src.split(), tgt.split()
    (e,) = extract_edits(s, t, names=NAMES)
    return {"type": e.type, **explain({"type": e.type, "src": e.src, "tgt": e.tgt, "tgt_span": e.tgt_span}, t, NAMES)}


def test_verb_cell_from_paradigms() -> None:
    assert verb_cell(["चला", "आया"]) == ("M", "sg")
    assert verb_cell(["हुई", "थीं"]) == ("F", "pl")
    assert verb_cell(["हैं"]) == (None, "pl")  # है/हैं carry no gender


def test_controller_subject_object_and_pronoun() -> None:
    t = "उनके बचपन में ही उनका परिवार कोलकाता चला आया ।".split()
    assert controller(t, 7) == ("परिवार", "subject")  # first word after the possessive उनका
    assert controller("उन्होंने कई पुस्तकें लिखीं ।".split(), 3) == ("पुस्तकें", "object")  # ने fused in उन्होंने
    assert controller("वह सेना में शामिल हुई थी ।".split(), 4) == ("वह", "subject")


def test_agr_gen_template() -> None:
    x = why("उनके बचपन में ही उनका परिवार कोलकाता चली आई ।", "उनके बचपन में ही उनका परिवार कोलकाता चला आया ।")
    assert x["type"] == "AGR-GEN"
    assert x["hi"] == "'चली आई' स्त्रीलिंग एकवचन रूप है, पर कर्ता 'परिवार' पुल्लिंग है, इसलिए क्रिया 'चला आया' (पुल्लिंग एकवचन) होगी।"
    assert x["en"] == ("'चली आई' is the feminine singular form, but the subject 'परिवार' is masculine, "
                       "so the verb is 'चला आया' (masculine singular).")


def test_agr_num_template_in_ne_clause_names_the_object() -> None:
    x = why("सिंह ने लंदन स्कूल ऑफ़ इकोनॉमिक्स में अध्ययन किए ।", "सिंह ने लंदन स्कूल ऑफ़ इकोनॉमिक्स में अध्ययन किया ।")
    assert x["type"] == "AGR-NUM"
    assert x["en"] == ("With ने the verb agrees with the object, not the subject. 'किए' is the masculine plural form, "
                       "but the object 'अध्ययन' is singular, so the verb is 'किया' (masculine singular).")


def test_agr_hon_template_names_the_person() -> None:
    x = why("लाल बहादुर शास्त्री एक अद्भुत इंसान था ।", "लाल बहादुर शास्त्री एक अद्भुत इंसान थे ।")
    assert x["type"] == "AGR-HON"
    assert x["hi"] == ("'लाल बहादुर शास्त्री' एक आदरणीय व्यक्ति हैं। आदर के लिए हिंदी में बहुवचन क्रिया आती है, "
                       "इसलिए 'था' की जगह 'थे' होगा।")


def test_postp_templates_substitution_genitive_insertion() -> None:
    x = why("ननिहाल पर रहते हुए उसने प्राथमिक शिक्षा ग्रहण की ।", "ननिहाल में रहते हुए उसने प्राथमिक शिक्षा ग्रहण की ।")
    assert x["en"] == "'ननिहाल में' is right: the meaning needed is 'में' (in, inside), not 'पर' (on, at)."
    g = why("वे पूना के कॉलेज की प्रोफेसर बने ।", "वे पूना के कॉलेज के प्रोफेसर बने ।")
    assert g["type"] == "POSTP" and "('प्रोफेसर')" in g["en"]  # genitive agrees with the next word
    i = why("उनका जन्म गांव हुआ था ।", "उनका जन्म गांव में हुआ था ।")
    assert i["hi"] == "'गांव' के बाद परसर्ग 'में' (के अंदर) छूट गया था।"


def test_erg_templates() -> None:
    x = why("वे 9 जून 1964 को भारत के प्रधानमंत्री का पद भार ग्रहण किया ।",
            "उन्होंने 9 जून 1964 को भारत के प्रधानमंत्री का पद भार ग्रहण किया ।")
    assert x["en"] == "With a past (perfective) transitive verb 'किया', the subject takes ने: 'वे' + ने = 'उन्होंने'."
    d = why("गांधी ने स्कूल गए ।", "गांधी स्कूल गए ।")
    assert d["en"] == "This verb 'गए' does not take ने (intransitive or not perfective), so ने is removed."


def test_aux_template_uses_the_year_anchor() -> None:
    x = why("वे वर्ष 1984 में 8वीं लोकसभा में उत्तरी गोवा से निर्वाचित हुए हैं ।",
            "वे वर्ष 1984 में 8वीं लोकसभा में उत्तरी गोवा से निर्वाचित हुए थे ।")
    assert x["type"] == "AUX"
    assert x["en"] == "The sentence is about the past ('1984'), so the auxiliary 'हैं' becomes 'थे' (past tense)."


def test_spell_templates_matra_and_family() -> None:
    x = why("उन्होंने अपनी शीक्षा दिल्ली में पूरी की ।", "उन्होंने अपनी शिक्षा दिल्ली में पूरी की ।")
    assert x["hi"] == "मानक वर्तनी 'शिक्षा' है: 'शीक्षा' में दीर्घ ई की मात्रा (ी) की जगह ह्रस्व इ की मात्रा (ि) चाहिए।"
    f = why("वे दिल्ली चले गये ।", "वे दिल्ली चले गए ।")
    assert f["type"] == "SPELL"
    assert f["en"] == "'गये' and 'गए' are two spellings of the same form; the standard one is 'गए'."


def test_pron_templates_honorific_and_possessive() -> None:
    def why_ext(src: str, tgt: str) -> dict:
        s, t = src.split(), tgt.split()
        (e,) = extract_edits(s, t, names=NAMES, extended=True)
        return {"type": e.type, **explain({"type": e.type, "src": e.src, "tgt": e.tgt, "tgt_span": e.tgt_span}, t, NAMES)}
    h = why_ext("उसने कई पुस्तकें लिखीं ।", "उन्होंने कई पुस्तकें लिखीं ।")
    assert h["type"] == "PRON"
    assert h["en"] == "Hindi refers to one respected person with the plural pronoun, so 'उसने' becomes 'उन्होंने'."
    p = why_ext("उनका माता एक शिक्षिका थीं ।", "उनकी माता एक शिक्षिका थीं ।")
    assert p["type"] == "PRON"
    assert p["hi"] == ("संबंध-सूचक सर्वनाम का/की/के की तरह अगले शब्द 'माता' से लिंग-वचन में मेल खाता है, इसलिए "
                       "'उनका' की जगह 'उनकी'।")


# ------------------------------------------------------------- honorific subject must be a person
def hon(sentence: str) -> dict:  # explanation of a था -> थे (AGR-HON) edit on the final verb of ``sentence``
    t = sentence.split()
    k = t.index("थे") if "थे" in t else t.index("हैं")
    return explain({"type": "AGR-HON", "src": "था" if t[k] == "थे" else "है", "tgt": t[k], "tgt_span": [k, k + 1]},
                   t, NAMES)


NEUTRAL_EN = "The verb must match its subject in number (Hindi uses the plural verb for one respected person)"


def test_honorific_never_names_a_time_word_or_noun() -> None:
    from hindi_gec.explain import person_subject
    assert person_subject("आज वह एक प्रसिद्ध गायक हैं ।".split(), 5, NAMES) == ""  # आज is a time word; वह may be "it"
    assert person_subject("इसी वजह से वे बहुत लोकप्रिय थे ।".split(), 6, NAMES) == "वे"  # वजह skipped, वे found
    assert person_subject("उस दौरान उनकी रचनाएँ बहुत लोकप्रिय थीं ।".split(), 6, NAMES) == ""
    assert hon("आज वह एक प्रसिद्ध गायक हैं ।")["en"].startswith(NEUTRAL_EN)
    assert hon("उस वर्ष कई पुरस्कार मिले थे ।")["hi"].startswith("क्रिया का वचन कर्ता से मेल खाना चाहिए")


def test_honorific_names_a_clear_person() -> None:
    from hindi_gec.explain import person_subject
    assert person_subject("लाल बहादुर शास्त्री एक अद्भुत इंसान थे ।".split(), 6, NAMES) == "लाल बहादुर शास्त्री"
    assert person_subject("उनके पिता एक शिक्षक थे ।".split(), 4, NAMES) == "पिता"  # person noun after a possessive
    assert person_subject("उन्होंने कई पुस्तकें लिखीं और वे प्रसिद्ध थे ।".split(), 7, NAMES) == "उन्होंने"
    assert person_subject("शास्त्री परिवार बहुत प्रसिद्ध थे ।".split(), 4, NAMES) == ""  # the name only modifies परिवार
    assert hon("उनके पिता एक शिक्षक थे ।")["en"].startswith("'पिता' is one respected person")


def test_agreement_never_names_a_non_subject_word() -> None:
    from hindi_gec.explain import is_subject_word, subject_words
    for w in ("आज", "वजह", "दौरान", "अनुसार", "1995", "जनवरी", "प्राप्त", "दो", "पहली"):
        assert not is_subject_word(w), w
    assert subject_words("आज बहुत गर्मी है ।".split(), 3) == ""  # first run is आज: not named
    assert controller("उन्होंने यह पुरस्कार प्राप्त किया ।".split(), 4) == ("पुरस्कार", "object")  # प्राप्त is part of the verb
