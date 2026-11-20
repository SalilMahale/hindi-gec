# -*- coding: utf-8 -*-
"""Person-name gazetteer, subject span, and the AGR-HON vs AGR-NUM relabel — hand-computed values."""

from __future__ import annotations  # modern type-hint syntax

from collections import Counter  # corpus counts

from hindi_gec.edit_extractor import extract_edits  # classifier under test
from hindi_gec.names import build_gazetteer, subject_is_name, subject_run, title_tokens  # functions under test


def test_title_tokens_drop_disambiguator_and_latin() -> None:
    assert title_tokens("राज कपूर (अभिनेता)") == ["राज", "कपूर"]  # "(अभिनेता)" removed
    assert title_tokens("ए. पी. जे. अब्दुल कलाम") == ["अब्दुल", "कलाम"]  # "ए." has a full stop -> not pure Devanagari


def test_build_gazetteer_drops_common_words_unless_in_many_titles() -> None:
    titles = ["विजय सिंह", "राम सिंह", "भारत भूषण", "के आर नारायणन"]
    corpus = Counter({"भारत": 50, "सिंह": 40, "में": 100, "राम": 1})  # most_common(2) = में, भारत
    names = build_gazetteer(titles, corpus, max_common_rank=2, keep_if_titles=2)
    assert "भारत" not in names  # among the 2 most frequent words and in only 1 title
    assert "सिंह" in names  # not in the top 2 (rank 3) -> kept
    assert "के" not in names  # postposition
    assert {"विजय", "राम", "भूषण", "नारायणन", "आर"} <= names
    # with keep_if_titles=1 a frequent title word survives
    assert "भारत" in build_gazetteer(titles, corpus, max_common_rank=2, keep_if_titles=1)


def test_subject_run_skips_postpositional_phrases() -> None:
    s = "उनके बचपन में ही उनका परिवार कोलकाता चला आया ।".split()
    # बचपन is followed by में (adverbial) -> skipped; परिवार कोलकाता runs up to the verb (index 7)
    assert subject_run(s, 7) == ["परिवार", "कोलकाता"]
    t = "लाल बहादुर शास्त्री एक अद्भुत इंसान था ।".split()
    assert subject_run(t, 6) == ["लाल", "बहादुर", "शास्त्री"]  # stops at एक (function word, not a postposition)
    assert subject_run("गांधी के पिता दीवान थे ।".split(), 4) == ["पिता", "दीवान"]  # गांधी के = possessor, not subject


def test_shastri_tha_the_is_honorific_with_gazetteer() -> None:
    src = "लाल बहादुर शास्त्री एक अद्भुत इंसान था ।".split()
    tgt = "लाल बहादुर शास्त्री एक अद्भुत इंसान थे ।".split()
    names = frozenset({"लाल", "बहादुर", "शास्त्री"})
    assert extract_edits(src, tgt)[0].type == "AGR-NUM"  # no names: original rule (no honorific trigger)
    assert extract_edits(src, tgt, names=names)[0].type == "AGR-HON"  # the subject is a person
    assert subject_is_name(tgt, 6, names)


def test_plural_subject_stays_number_agreement() -> None:
    src = "उसके दोनों भाई सेना में था ।".split()
    tgt = "उसके दोनों भाई सेना में थे ।".split()
    names = frozenset({"लाल", "बहादुर", "शास्त्री"})
    assert extract_edits(src, tgt, names=names)[0].type == "AGR-NUM"  # भाई is not a gazetteer name
