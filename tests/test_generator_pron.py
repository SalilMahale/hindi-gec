# -*- coding: utf-8 -*-
"""Generator pieces for PRON, the forced SPELL pair and corpus normalisation, and the PRON classifier rule."""

from __future__ import annotations  # modern type-hint syntax

import random  # seeded rng

from hindi_gec.edit_extractor import extract_edits  # classifier
from hindi_gec.error_generator import (forced_spell_candidates, normalise_forced,  # functions under test
                                       pronoun_candidates)

GEN = {"का": {"की": 1.0}, "की": {"का": 0.6, "के": 0.4}, "के": {"की": 0.6, "का": 0.4}}  # config generator.genitive_confusion
FORCED = {"उन्होंने": "उन्होने"}


def test_honorific_pronoun_made_singular_but_not_real_plurals() -> None:
    s = "उन्होंने अपनी शिक्षा दिल्ली में पूरी की ।".split()
    c = [x for x in pronoun_candidates(s, GEN, random.Random(0)) if x.start == 0]
    assert c and c[0].new == ("उसने",) and c[0].type == "PRON"  # ergative honorific -> singular
    p = "उनके दोनों पुत्र वकील थे ।".split()
    news = {x.new[0] for k in range(20) for x in pronoun_candidates(p, GEN, random.Random(k)) if x.start == 0}
    assert "उसके" not in news  # उनके दोनों = real plural: never made singular (an ending swap is still allowed)


def test_possessive_ending_follows_genitive_rules() -> None:
    s = "वे अपनी पत्नी के साथ रहते थे ।".split()
    (c,) = [x for x in pronoun_candidates(s, GEN, random.Random(0)) if x.start == 1]
    assert c.new[0] in ("अपना", "अपने")  # की -> का / के
    t = "इसके बाद उनका विवाह हुआ ।".split()  # के before बाद (compound head): के -> का allowed
    outs = {x.new[0] for k in range(20) for x in pronoun_candidates(t, GEN, random.Random(k)) if x.start == 0}
    assert outs <= {"इसकी", "इसका"} and outs


def test_forced_spell_and_normalisation() -> None:
    s = "उन्होंने कविता लिखी ।".split()
    (c,) = forced_spell_candidates(s, FORCED)
    assert (c.start, c.new, c.type) == (0, ("उन्होने",), "SPELL")
    norm, corrs = normalise_forced("उन्होने कविता लिखी ।".split(), FORCED)
    assert norm == s  # the target is fixed
    assert [(x.start, x.new) for x in corrs] == [(0, ("उन्होने",))]  # the corpus original is the wrong source


def test_pron_classified_only_with_extended() -> None:
    src = "उसने कई पुस्तकें लिखीं ।".split()
    tgt = "उन्होंने कई पुस्तकें लिखीं ।".split()
    assert extract_edits(src, tgt)[0].type == "OTHER"  # original 7-type classifier: unchanged
    assert extract_edits(src, tgt, extended=True)[0].type == "PRON"
    poss = extract_edits("उनके जन्म दिल्ली में हुआ ।".split(), "उनका जन्म दिल्ली में हुआ ।".split(), extended=True)
    assert poss[0].type == "PRON"  # possessive agreement
    erg = extract_edits("वे कविता लिखी ।".split(), "उन्होंने कविता लिखी ।".split(), extended=True)
    assert erg[0].type == "ERG"  # वे -> उन्होंने stays ERG


def test_forced_pairs_sit_on_top_of_the_quota() -> None:
    """forced_spell_on_top: n_pairs quota pairs never use the forced pair; n_forced forced pairs are added after."""
    from hindi_gec.config import load_config
    from hindi_gec.error_generator import generate_split
    cfg = load_config()
    g = dict(cfg["generator"])
    sents = [{"page_title": "क", "tokens": t.split()} for t in (
        "उन्होंने 1950 में स्नातक की उपाधि प्राप्त की ।", "उन्होंने कई पुस्तकें लिखीं ।",
        "उन्होंने दिल्ली में पढ़ाई की ।", "वह एक प्रसिद्ध गायक थे ।", "उनका जन्म दिल्ली में हुआ था ।")]
    natural = [({"page_title": "क", "tokens": "उन्होंने गीत गाए ।".split()},
                normalise_forced("उन्होने गीत गाए ।".split(), FORCED)[1])]
    pairs, _ = generate_split(sents, 4, "train", {}, g, random.Random(0), set(), 4, natural, n_forced=3)
    quota, extra = pairs[:-3], pairs[-3:]
    assert len(pairs) == 7  # 4 quota pairs + 3 forced
    assert not any(e.src == "उन्होने" for p in quota for e in p.edits)  # ordinary pairs: no forced edit
    assert extra[0].source == "उन्होने गीत गाए ।"  # the natural pair comes first
    assert all([e.src for e in p.edits] == ["उन्होने"] for p in extra)  # one forced edit each
