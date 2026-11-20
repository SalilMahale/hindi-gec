# -*- coding: utf-8 -*-
"""Behaviour tests for the preprocessing pipeline.

Each test pins a linguistic decision, not just a line of code: that Devanagari
digits are unified, that the danda ends a sentence, that stemming refuses to
eat a short root, and above all that `gec_pipeline` preserves postpositions
and verb inflections (removing them would destroy the very signal the
corrector has to learn).
"""

from collections import Counter  # counting dictionary

from hindi_gec.preprocessing import (  # functions under test
    HINDI_STOPWORDS,
    analysis_pipeline,
    build_parallel_corpus,
    corpus_statistics,
    drop_punctuation,
    export_corpus_csv,
    flatten,
    gec_pipeline,
    load_corpus,
    normalize,
    remove_noise,
    remove_stopwords,
    split_sentences,
    stem,
    tokenize,
)


# ---------------------------------------------------------------- noise
def test_remove_noise_strips_markup_urls_and_latin():
    raw = "<p>भारत में ५ बड़े नगर हैं। See https://hi.wikipedia.org here</p>"
    out = remove_noise(raw)
    assert "<p>" not in out and "http" not in out  # HTML and URL gone
    assert not any("a" <= ch.lower() <= "z" for ch in out)  # no English letters left
    assert "भारत" in out and "नगर" in out  # Hindi kept


def test_remove_noise_drops_brackets_emptied_by_latin_removal():
    """"(MLA)" must not survive as "( )" once the Latin run is stripped."""
    out = remove_noise("वे विधान सभा सदस्य (MLA) थे।")
    assert "(" not in out and ")" not in out  # empty brackets removed
    assert "विधान सभा सदस्य" in out and "थे" in out


def test_remove_noise_keeps_brackets_with_devanagari_content():
    out = remove_noise("वे विधान सभा सदस्य (विधायक) थे।")
    assert "(विधायक)" in out  # brackets with Hindi inside are kept


def test_remove_noise_squeezes_whitespace():
    assert remove_noise("भारत   में    नगर") == "भारत में नगर"  # multiple spaces -> one


# ---------------------------------------------------------------- normalise
def test_normalize_unifies_devanagari_digits():
    assert normalize("सन् १९४७ में") == "सन् 1947 में"  # १९४७ -> 1947


def test_normalize_is_idempotent():
    text = "उनका जन्म १९३१ में हुआ था।"
    assert normalize(normalize(text)) == normalize(text)  # normalising twice changes nothing more


# ---------------------------------------------------------------- segment
def test_split_sentences_breaks_on_danda_and_question_mark():
    para = "राम बाजार गया। सीता घर पर पढ़ रही है। क्या तुम आओगे?"
    assert len(split_sentences(para)) == 3  # three sentences


# ---------------------------------------------------------------- tokenise
def test_tokenize_keeps_danda_as_its_own_token():
    toks = tokenize("राम बाजार गया।")
    assert toks[-1] == "।"  # danda is a separate token
    assert toks[:3] == ["राम", "बाजार", "गया"]  # not glued to गया


def test_drop_punctuation_removes_danda_only():
    assert drop_punctuation(["राम", "गया", "।"]) == ["राम", "गया"]


# ---------------------------------------------------------------- stemming
def test_stem_strips_longest_suffix():
    assert stem("लड़कियों") != "लड़कियों"  # a suffix was removed


def test_stem_protects_short_roots():
    # Stripping would leave fewer than 3 characters, so the word is untouched.
    assert stem("गया") == "गया"


def test_remove_stopwords_drops_function_words():
    tokens = ["उनका", "जन्म", "में", "हुआ"]
    out = remove_stopwords(tokens)
    assert "जन्म" in out  # content word kept
    assert all(t not in HINDI_STOPWORDS for t in out)  # no stop words left


# ---------------------------------------------------------------- pipelines
def test_gec_pipeline_preserves_postpositions_and_inflection():
    """The correction path must not lose में / था — they carry the grammar."""
    sent = "उनका जन्म 1931 में रामेश्वरम में हुआ था।"
    tokens = [t for s in gec_pipeline(sent) for t in s]
    assert tokens.count("में") == 2  # both postpositions kept
    assert "था" in tokens  # auxiliary kept


def test_analysis_pipeline_is_more_aggressive_than_gec_pipeline():
    sent = "उनका जन्म 1931 में रामेश्वरम में हुआ था।"
    assert len(analysis_pipeline(sent)) < len([t for s in gec_pipeline(sent) for t in s])  # removes more tokens


def test_flatten_returns_space_separated_string():
    assert flatten("राम बाजार गया।") == "राम बाजार गया ।"  # danda separated by a space


# ---------------------------------------------------------------- corpus
def test_build_parallel_corpus_applies_every_rejection_rule():
    pairs = [
        ("राम बाजार गया था।", "राम बाजार गई थी।"),      # kept
        ("राम बाजार गया था।", "राम बाजार गई थी।"),      # duplicate
        ("सीता घर गई।", "सीता घर गई।"),                  # no edit left
        ("<p>hello</p>", "<p>world</p>"),                # empty after cleaning
        ("राम गया।", "राम गई।"),                          # too short (3 tokens incl. danda)
    ]
    corpus, stats = build_parallel_corpus(pairs, min_len=4, max_len=60)
    assert len(corpus) == 1  # only the first pair survives
    assert stats["duplicate"] == 1
    assert stats["no edit left"] == 1
    assert stats["empty after cleaning"] == 1
    assert stats["length out of range"] == 1


def test_corpus_statistics_counts_source_side():
    corpus = [("राम बाजार गया ।", "राम बाजार गई ।")]
    stats = corpus_statistics(corpus)
    assert stats["pairs"] == 1
    assert stats["source_tokens"] == 4  # राम बाजार गया ।
    assert stats["avg_sentence_length"] == 4.0


def test_load_corpus_and_export_roundtrip(tmp_path):  # tmp_path = temporary folder provided by pytest
    src = tmp_path / "a.src"
    trg = tmp_path / "a.trg"
    src.write_text("राम बाजार गया था।\nसीता घर गई।\n", encoding="utf-8")  # two incorrect lines
    trg.write_text("राम बाजार गया था।\nसीता घर गया।\n", encoding="utf-8")  # two corrected lines

    pairs = load_corpus(src, trg)
    assert len(pairs) == 2

    out = tmp_path / "out.csv"
    export_corpus_csv([("राम गया", "राम गई")], out)
    text = out.read_text(encoding="utf-8-sig")
    assert text.startswith("ID,")  # header row first
    assert "राम गया" in text
    # utf-8-sig must have written a BOM so Excel on Windows renders Devanagari.
    assert out.read_bytes().startswith(b"\xef\xbb\xbf")  # the 3-byte BOM


def test_rejection_stats_is_a_counter():
    _, stats = build_parallel_corpus([])
    assert isinstance(stats, Counter)
