# -*- coding: utf-8 -*-
"""Tests for the corpus builder — no network required.

The API client is exercised against a fake transport, and every filter is
checked against a hand-written sentence whose verdict is obvious to a human
reader. That is the point: if a filter's behaviour ever drifts, these
sentences say so in Hindi, not in a coverage percentage.
"""

import json  # build fake API responses

import pytest  # testing framework

from hindi_gec.corpus_builder import (  # functions under test
    BULLET_RE,
    CITATION_RE,
    NearDuplicateFilter,
    QualityConfig,
    WikiClient,
    brackets_balanced,
    devanagari_ratio,
    select_sections,
    sentence_rejection_reason,
    sentences_from_extract,
    split_into_sections,
)
from hindi_gec.preprocessing import tokenize
from hindi_gec.schema import CleanSentence, Edit, Pair, count_jsonl, read_jsonl, write_jsonl

QC = QualityConfig()  # default filter settings


def _reason(text: str, qc: QualityConfig = QC):  # helper: why would this sentence be rejected? (None = accepted)
    return sentence_rejection_reason(text, tokenize(text), qc)


# ---------------------------------------------------------------- sections
EXTRACT = """उनका जन्म 1931 में रामेश्वरम में हुआ था। वे एक वैज्ञानिक थे।

== प्रारम्भिक जीवन ==
उनके पिता एक नाविक थे। उन्होंने स्कूल की पढ़ाई रामेश्वरम में पूरी की।

== पुरस्कार एवं सम्मान ==
उन्हें भारत रत्न से सम्मानित किया गया।

== इन्हें भी देखें ==
कुछ और लेख।
"""  # a fake Wikipedia article: lead + 3 sections


def test_split_into_sections_returns_lead_first():
    secs = split_into_sections(EXTRACT)
    assert secs[0][0] == ""  # lead has an empty title
    assert "रामेश्वरम" in secs[0][1]
    assert [t for t, _ in secs[1:]] == ["प्रारम्भिक जीवन", "पुरस्कार एवं सम्मान", "इन्हें भी देखें"]


def test_select_sections_keeps_lead_and_life_sections_only():
    kept = select_sections(split_into_sections(EXTRACT), ["जीवन", "शिक्षा", "करियर"])
    assert [t for t, _ in kept] == ["", "प्रारम्भिक जीवन"]  # awards and "see also" dropped


def test_select_sections_matches_heading_variants():
    secs = [("", "lead"), ("प्रारंभिक जीवन", "a"), ("वैज्ञानिक जीवन", "b"), ("फिल्मोग्राफी", "c")]
    kept = [t for t, _ in select_sections(secs, ["जीवन"])]
    assert kept == ["", "प्रारंभिक जीवन", "वैज्ञानिक जीवन"]  # substring match on "जीवन"


# ---------------------------------------------------------------- filters
def test_good_biography_sentence_is_accepted():
    assert _reason("उनका जन्म 1931 में रामेश्वरम में हुआ था ।") is None


def test_too_short_sentence_is_rejected():
    assert _reason("वे वैज्ञानिक थे ।") == "too short"  # only 3 words


def test_too_long_sentence_is_rejected():
    long_sent = " ".join(["शब्द"] * 45) + " ।"  # 45 words
    assert _reason(long_sent) == "too long"


def test_sentence_without_danda_is_rejected():
    assert _reason("उनका जन्म 1931 में रामेश्वरम में हुआ था") == "does not end in danda"


def test_transliteration_heavy_sentence_is_rejected():
    text = "अब्दुल कलाम (Avul Pakir Jainulabdeen Abdul Kalam, அவுல் பகிர் ஜைனுலாப்தீன்) वैज्ञानिक थे ।"  # mostly English + Tamil
    assert _reason(text) == "not enough Devanagari"


def test_unbalanced_brackets_are_rejected():
    assert _reason("उनका जन्म 1931 में (रामेश्वरम में हुआ था ।") == "unbalanced brackets"


def test_citation_marker_is_rejected():
    assert _reason("उनका जन्म 1931 में रामेश्वरम में हुआ था[1] ।") == "citation marker"


def test_bullet_line_is_rejected():
    assert _reason("- उनका जन्म 1931 में रामेश्वरम में हुआ था ।") == "leading bullet / list item"
    assert BULLET_RE.search("• कुछ")


def test_devanagari_ratio_boundaries():
    assert devanagari_ratio("उनका जन्म हुआ") == 1.0  # all Hindi
    assert devanagari_ratio("abcd efgh") == 0.0  # no Hindi
    assert 0.0 < devanagari_ratio("उनका जन्म Abdul Kalam") < 1.0  # mixed


def test_brackets_balanced_cases():
    assert brackets_balanced("(क) [ख]")
    assert not brackets_balanced("(क]")  # wrong closing bracket
    assert not brackets_balanced("क)")  # closing without opening


def test_citation_regex_handles_devanagari_digits():
    assert CITATION_RE.search("पाठ[२]")  # [२] with a Hindi digit


# ---------------------------------------------------------------- dedup
def test_near_duplicate_filter_catches_effectively_identical_sentences():
    """0.90 removes sentences that differ only in surface form."""
    f = NearDuplicateFilter(shingle_size=3, threshold=0.90, seed=42)
    a = tokenize("उनका जन्म 1931 में रामेश्वरम में हुआ था ।")
    b = tokenize("उनका जन्म 1931 में रामेश्वरम में हुआ था।")    # danda spacing only
    assert f.is_duplicate(a) is False  # first one is new
    assert f.is_duplicate(b) is True  # second one is a duplicate


def test_near_duplicate_filter_keeps_genuinely_different_facts():
    """Two biographies sharing a template are NOT duplicates — see class docstring."""
    f = NearDuplicateFilter(shingle_size=3, threshold=0.90, seed=42)
    a = tokenize("उनका जन्म 1931 में रामेश्वरम में हुआ था ।")
    b = tokenize("उनका जन्म 1935 में इलाहाबाद में हुआ था ।")  # same template, different facts
    c = tokenize("वे भारतीय क्रिकेट टीम के कप्तान बने और कई मैच जीते ।")
    assert f.is_duplicate(a) is False
    assert f.is_duplicate(b) is False
    assert f.is_duplicate(c) is False


def test_near_duplicate_filter_is_deterministic_across_instances():
    toks = [tokenize("उनका जन्म 1931 में रामेश्वरम में हुआ था ।"),
            tokenize("उनका जन्म 1932 में रामेश्वरम में हुआ था ।")]
    runs = []
    for _ in range(2):  # two separate filters with the same seed
        f = NearDuplicateFilter(shingle_size=3, threshold=0.80, seed=42)
        runs.append([f.is_duplicate(t) for t in toks])
    assert runs[0] == runs[1]  # same results


def test_near_duplicate_filter_rejects_bad_band_config():
    with pytest.raises(ValueError):  # 32 hashes can't be split into 7 equal bands
        NearDuplicateFilter(num_hashes=32, bands=7)


# ---------------------------------------------------------------- end to end
def test_sentences_from_extract_keeps_only_life_prose():
    sents, rejects = sentences_from_extract(
        EXTRACT, "ए॰ पी॰ जे॰ अब्दुल कलाम", 123, ["जीवन", "शिक्षा", "करियर"], QC
    )
    texts = [s.text for s in sents]
    assert any("रामेश्वरम" in t for t in texts)
    assert not any("भारत रत्न" in t for t in texts)      # award section excluded
    assert all(s.page_title == "ए॰ पी॰ जे॰ अब्दुल कलाम" and s.revid == 123 for s in sents)  # provenance stored
    assert {s.section for s in sents} <= {"", "प्रारम्भिक जीवन"}  # only lead and life section
    assert isinstance(rejects, dict)


# ---------------------------------------------------------------- API client
class FakeOpener:
    """Stand-in for urllib.request.urlopen returning canned JSON."""

    def __init__(self, payloads):
        self.payloads = list(payloads)  # responses to return, in order
        self.calls = []  # URLs that were requested

    def __call__(self, req, timeout=None):  # called like urlopen(req, timeout=...)
        self.calls.append(req.full_url)
        payload = self.payloads.pop(0)  # next canned response

        class _Resp:  # minimal fake response object usable in a "with" block
            def __enter__(_s):
                return _s

            def __exit__(_s, *a):
                return False

            def read(_s):
                return json.dumps(payload).encode("utf-8")  # response body as bytes

        return _Resp()


def test_category_members_follows_continuation(monkeypatch, tmp_path):  # monkeypatch = pytest tool to temporarily replace functions
    import urllib.request as ur

    payloads = [
        {"query": {"categorymembers": [{"title": "क"}, {"title": "ख"}]},
         "continue": {"cmcontinue": "X", "continue": "-||"}},  # first page says "there is more"
        {"query": {"categorymembers": [{"title": "ग"}]}},  # second (last) page
    ]
    fake = FakeOpener(payloads)
    monkeypatch.setattr(ur, "urlopen", fake)  # replace the real network call with the fake
    c = WikiClient("https://x/api.php", "UA", requests_per_second=0, cache_dir=tmp_path)
    assert c.category_members("श्रेणी:परीक्षण") == ["क", "ख", "ग"]  # both pages combined
    assert len(fake.calls) == 2


def test_client_caches_responses(monkeypatch, tmp_path):
    import urllib.request as ur

    fake = FakeOpener([{"query": {"categorymembers": [{"title": "क"}]}}])
    monkeypatch.setattr(ur, "urlopen", fake)
    c = WikiClient("https://x/api.php", "UA", requests_per_second=0, cache_dir=tmp_path)
    c.category_members("श्रेणी:परीक्षण")
    c.category_members("श्रेणी:परीक्षण")          # served from disk, no second call
    assert len(fake.calls) == 1
    assert c.stats == {"network": 1, "cache": 1, "retries": 0}


def test_page_extract_returns_text_and_revid(monkeypatch, tmp_path):
    import urllib.request as ur

    payload = {"query": {"pages": [{"title": "क", "extract": "पाठ", "revisions": [{"revid": 99}]}]}}
    monkeypatch.setattr(ur, "urlopen", FakeOpener([payload]))
    c = WikiClient("https://x/api.php", "UA", requests_per_second=0, cache_dir=tmp_path)
    assert c.page_extract("क") == ("पाठ", 99)  # (text, revision id)


# ---------------------------------------------------------------- schema
def test_jsonl_roundtrip_preserves_devanagari(tmp_path):
    recs = [CleanSentence(id="bio_000001", page_title="क", revid=1, section="",
                          text="उनका जन्म हुआ ।", tokens=["उनका", "जन्म", "हुआ", "।"])]
    out = tmp_path / "x.jsonl"
    assert write_jsonl(recs, out) == 1
    raw = out.read_text(encoding="utf-8")
    assert "उनका" in raw and "\\u" not in raw       # not escaped
    back = list(read_jsonl(out))
    assert back[0]["text"] == "उनका जन्म हुआ ।"
    assert count_jsonl(out) == 1


def test_pair_schema_serialises_edits(tmp_path):
    pair = Pair(id="syn_000001", origin="synthetic", page_title="क",
                source="सीता बाजार गया ।", target="सीता बाजार गई ।",
                source_tokens=["सीता", "बाजार", "गया", "।"],
                target_tokens=["सीता", "बाजार", "गई", "।"],
                edits=[Edit(src_span=[2, 3], tgt_span=[2, 3], src="गया", tgt="गई", type="AGR-GEN")],
                split="train")
    out = tmp_path / "p.jsonl"
    write_jsonl([pair], out)
    rec = next(read_jsonl(out))  # read back the first record
    assert rec["edits"][0]["type"] == "AGR-GEN"  # nested edit saved correctly
    assert rec["edits"][0]["src_span"] == [2, 3]
