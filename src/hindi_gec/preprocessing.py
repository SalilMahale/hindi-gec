# -*- coding: utf-8 -*-
"""Text preprocessing for Hindi GEC data (ported from `hindi_preprocessing.py`).

Why these particular steps, in this order:

* **Noise removal** first, because HTML/URLs/Latin runs would otherwise be
  normalised and tokenised into junk tokens that inflate the vocabulary.
* **Unicode normalisation (NFC + Indic normaliser)** second, because
  Devanagari has multiple byte sequences for the same visible glyph — nukta
  characters (क़ vs क+़) and pre-composed vs decomposed matras. If this is
  skipped, the *same word* becomes two vocabulary entries and a SPELL edit
  can be a pure encoding artefact rather than a real mistake.
* **Danda segmentation** third: Hindi ends sentences with `।`, not `.`.
* **Tokenisation** last, keeping punctuation as separate tokens so that token
  indices in the edit schema (PLAN §2.5) are reproducible.

Two pipelines are exposed deliberately. `gec_pipeline` is the *correction*
path: it keeps every function word and every inflection, because postpositions
(में, से, को) and verb suffixes (गया/गई/गए) are precisely what the model has to
get right. `analysis_pipeline` additionally strips stop words and stems, and
exists only to profile corpus vocabulary — it must never touch training pairs.

Ported unchanged in behaviour from the original standalone script; the console
driver now lives in `scripts/00_preprocess_demo.py` so that this module is
importable without side effects.
"""

from __future__ import annotations  # modern type-hint syntax

import csv  # write CSV (spreadsheet) files
import re  # regular expressions: pattern-based search/replace on text
import unicodedata  # Unicode helpers: character categories and normalisation (NFC)
from collections import Counter  # a dict that counts things
from pathlib import Path  # file paths
from typing import Dict, Iterable, List, Sequence, Set, Tuple  # type hints

# Optional library. Every function falls back to pure-Python logic if missing,
# so the package imports cleanly on a machine without indic-nlp-library.
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory  # Indic NLP library's script normaliser
    from indicnlp.tokenize import indic_tokenize, sentence_tokenize  # its word tokenizer and sentence splitter

    NORMALIZER = IndicNormalizerFactory().get_normalizer("hi")  # built once
    INDIC_NLP = True  # flag: the library is available
except ImportError:  # pragma: no cover - depends on the environment
    INDIC_NLP = False  # flag: library missing -> use our own regex fallbacks


# ----------------------------------------------------------------------
# 1. RESOURCES : Hindi stop words and light-stemmer suffixes
# ----------------------------------------------------------------------
HINDI_STOPWORDS: Set[str] = {  # very common "function" words (postpositions, pronouns, helpers) - used only for corpus statistics
    "का", "के", "की", "को", "में", "से", "पर", "और", "है", "हैं", "था", "थे",
    "थी", "हूँ", "हो", "ही", "भी", "यह", "वह", "ये", "वे", "एक", "कि", "तो",
    "ना", "नहीं", "कुछ", "जो", "अपने", "अपनी", "इस", "उस", "मैं", "तुम", "आप",
    "हम", "मेरा", "तेरा", "उनका", "लिए", "साथ", "तक", "बाद", "द्वारा", "किया",
    "करने", "करता", "करते", "रहा", "रहे", "रही", "गया", "गई", "गए", "वाला",
}

# Ordered longest-first so the largest matching suffix is stripped.
HINDI_SUFFIXES: List[str] = [  # word endings (verb/noun inflections) that the stemmer cuts off
    "ाओगे", "ाएगी", "ाएगा", "ियाँ", "ियों", "ियां",  # 4-character suffixes
    "ाकर", "ाइए", "ाईं", "ाया", "ेगी", "ेगा", "ोगी", "ोगे", "ाने", "ाना",  # 3-character suffixes
    "ाते", "ाती", "ाता", "तीं", "ाओं", "ाएं", "ुओं", "ुएं",
    "कर", "ाओ", "िए", "ाई", "ाए", "ने", "नी", "ना", "ते", "ीं", "ती", "ता",  # 2-character suffixes
    "ाँ", "ां", "ों", "ें",
    "ी", "ो", "े", "ू", "ु", "ि", "ा",  # single vowel signs (matras)
]


# ----------------------------------------------------------------------
# 2. STEP 1 : Noise removal
# ----------------------------------------------------------------------
def remove_noise(text: str) -> str:
    """Strip HTML tags, URLs, emojis and Latin characters; squeeze spaces."""
    text = re.sub(r"<[^>]+>", " ", text)                 # HTML markup
    text = re.sub(r"http\S+|www\.\S+", " ", text)        # web links
    text = re.sub(r"[A-Za-z]+", " ", text)               # English words
    text = "".join(ch for ch in text
                   if unicodedata.category(ch) != "So")  # emojis / symbols ("So" = Unicode category "Symbol, other")
    # Deleting the Latin run often leaves the brackets that held it behind:
    # "विधान सभा सदस्य (MLA)" becomes "विधान सभा सदस्य ( )". The husk is
    # balanced, so the bracket check never flags it, and it would survive all
    # the way into the training data as a meaningless token pair.
    text = re.sub(r"[(\[{]\s*[,;:।॥.\-–—]*\s*[)\]}]", " ", text)  # delete empty brackets like "( )" or "(, )"
    text = re.sub(r"\s+([।?!,.])", r"\1", text)          # re-attach punctuation
    text = re.sub(r"(?<![\u0900-\u0963\u0966-\u097F0-9])[।?!,.]", " ", text)  # remove punctuation not preceded by a Hindi letter/digit (left-over junk)
    return re.sub(r"\s+", " ", text).strip()             # extra whitespace


# ----------------------------------------------------------------------
# 3. STEP 2 : Unicode normalisation
# ----------------------------------------------------------------------
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")  # translation table: Devanagari digits -> English digits


def normalize(text: str) -> str:
    """Canonically compose Devanagari, fix nukta forms, unify digits."""
    text = unicodedata.normalize("NFC", text)            # single code-point form
    if INDIC_NLP:  # if the Indic NLP library is installed
        text = NORMALIZER.normalize(text)                # nukta, chandrabindu
    return text.translate(DEV_DIGITS)                    # digit unification


# ----------------------------------------------------------------------
# 4. STEP 3 : Sentence segmentation
# ----------------------------------------------------------------------
def split_sentences(text: str) -> List[str]:
    """Break a paragraph on the danda and on the question or exclamation mark."""
    if INDIC_NLP:
        return sentence_tokenize.sentence_split(text, lang="hi")  # library splitter (knows Hindi rules)
    return [s.strip() for s in re.split(r"(?<=[।?!])\s*", text) if s.strip()]  # fallback: split right after । ? ! and drop empty pieces


# ----------------------------------------------------------------------
# 5. STEP 4 : Word tokenisation
# ----------------------------------------------------------------------
#: Devanagari letters and matras, *excluding* danda U+0964 and double danda
#: U+0965. The naive range ऀ-ॿ contains the danda, which would glue
#: it onto the preceding word ("गया।" as one token) — that both contradicts
#: `indic_tokenize.trivial_tokenize` and shifts every token index after it,
#: corrupting the edit spans of the data schema (PLAN §2.5). The same
#: punctuation-excluding range is already used in `remove_noise`.
DEV_WORD_RANGE = r"\u0900-\u0963\u0966-\u097F"  # regex character range for Hindi letters (danda excluded)


def tokenize(sentence: str) -> List[str]:
    """Split a sentence into words, keeping punctuation as separate tokens."""
    if INDIC_NLP:
        return indic_tokenize.trivial_tokenize(sentence, lang="hi")  # library tokenizer
    return re.findall(rf"[{DEV_WORD_RANGE}]+|[0-9]+|[।॥,.?!]", sentence)  # fallback: a run of Hindi letters, OR a number, OR one punctuation mark


# ----------------------------------------------------------------------
# 6. STEP 5 : Punctuation filtering
# ----------------------------------------------------------------------
PUNCT: Set[str] = set("।॥,.?!;:'\"()[]{}-–—…")  # set of punctuation characters


def drop_punctuation(tokens: Sequence[str]) -> List[str]:
    """Keep only tokens that carry lexical content."""
    return [t for t in tokens if t not in PUNCT]  # keep every token that isn't punctuation


# ----------------------------------------------------------------------
# 7. STEP 6 : Stop word removal   (analysis only, see Section 9)
# ----------------------------------------------------------------------
def remove_stopwords(tokens: Sequence[str]) -> List[str]:
    """Drop high-frequency function words that add statistical noise."""
    return [t for t in tokens if t not in HINDI_STOPWORDS]  # keep every token that isn't a stop word


# ----------------------------------------------------------------------
# 8. STEP 7 : Light stemming
# ----------------------------------------------------------------------
def stem(word: str) -> str:
    """Strip the longest matching inflectional suffix, guarding short roots."""
    for suf in HINDI_SUFFIXES:  # longest suffixes are checked first
        if word.endswith(suf) and len(word) - len(suf) >= 3:   # protect the root
            return word[: -len(suf)]  # cut the suffix off, e.g. "लड़कों" -> "लड़क"
    return word  # no suffix matched -> word unchanged


# ----------------------------------------------------------------------
# 9. TWO PIPELINES : correction path versus analysis path
# ----------------------------------------------------------------------
# Grammar correction is a sequence to sequence task, so the model must see
# every function word and every inflection. Stop word removal and stemming
# therefore run only for corpus statistics, never on the training pairs.
def gec_pipeline(text: str) -> List[List[str]]:
    """Preprocessing actually fed to the correction model."""
    clean = normalize(remove_noise(text))  # step 1 (noise) then step 2 (normalise)
    return [tokenize(s) for s in split_sentences(clean)]  # step 3 (sentences) then step 4 (words): list of token lists


def analysis_pipeline(text: str) -> List[str]:
    """Aggressive preprocessing used to profile the corpus vocabulary."""
    tokens = [t for s in gec_pipeline(text) for t in s]  # flatten all sentences into one token list
    tokens = remove_stopwords(drop_punctuation(tokens))  # remove punctuation, then stop words
    return [stem(t) for t in tokens]  # stem each remaining word


# ----------------------------------------------------------------------
# 10. STEP 8 : Corpus level cleaning of the parallel pairs
# ----------------------------------------------------------------------
def flatten(text: str) -> str:
    """Run the GEC pipeline and return one space separated token string."""
    return " ".join(t for s in gec_pipeline(text) for t in s)  # all tokens of all sentences joined by single spaces


def build_parallel_corpus(
    pairs: Iterable[Tuple[str, str]],  # (incorrect, corrected) raw sentence pairs
    min_len: int = 3,  # shortest allowed sentence (in tokens)
    max_len: int = 60,  # longest allowed sentence (in tokens)
) -> Tuple[List[Tuple[str, str]], "Counter[str]"]:
    """Clean both sides, drop unusable pairs, remove duplicates.

    Returns the surviving ``(incorrect, corrected)`` pairs and a counter of
    rejection reasons, so every dropped sentence is accounted for in the
    experiment write-up.
    """
    corpus: List[Tuple[str, str]] = []  # pairs we keep
    seen: Set[Tuple[str, str]] = set()  # pairs already kept (to detect duplicates fast)
    stats: "Counter[str]" = Counter()  # how many pairs were dropped for each reason
    for src_raw, tgt_raw in pairs:  # each raw pair
        src, tgt = flatten(src_raw), flatten(tgt_raw)  # clean both sides
        if not src or not tgt:  # one side became empty
            stats["empty after cleaning"] += 1
            continue  # skip this pair
        if src == tgt:                                   # no edit means no signal
            stats["no edit left"] += 1
            continue
        if not (min_len <= len(src.split()) <= max_len):  # too short or too long
            stats["length out of range"] += 1
            continue
        if (src, tgt) in seen:  # exact same pair already kept
            stats["duplicate"] += 1
            continue
        seen.add((src, tgt))  # remember it
        corpus.append((src, tgt))  # keep it
    return corpus, stats  # cleaned pairs + rejection counts


# ----------------------------------------------------------------------
# 11. DATASET LOADING
# ----------------------------------------------------------------------
def load_corpus(src_path: str | Path, trg_path: str | Path) -> List[Tuple[str, str]]:
    """Read the line aligned source and target files of the HiWikEd dataset."""
    with open(src_path, encoding="utf-8") as f:
        src = [l.strip() for l in f]  # each line = one incorrect sentence
    with open(trg_path, encoding="utf-8") as f:
        trg = [l.strip() for l in f]  # each line = its corrected version (same line number)
    assert len(src) == len(trg), "Files are not line-aligned!"  # both files must have the same number of lines
    return list(zip(src, trg))  # pair line 1 with line 1, line 2 with line 2, ...


# ----------------------------------------------------------------------
# 11B. EXPORT : write the full cleaned corpus to a CSV file
# ----------------------------------------------------------------------
def export_corpus_csv(corpus: Sequence[Tuple[str, str]], out_path: str | Path) -> None:
    """Save every cleaned (incorrect, corrected) pair, one row per pair.

    utf-8-sig is used deliberately: it writes a BOM so Excel on Windows
    renders the Devanagari text correctly instead of showing garbled
    characters (plain utf-8 does not carry this marker).
    """
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:  # newline="" avoids blank rows on Windows
        w = csv.writer(f)  # CSV writer object
        w.writerow(["ID", "Incorrect Sentence", "Corrected Sentence"])  # header row
        for i, (s, t) in enumerate(corpus, 1):  # i starts from 1
            w.writerow([i, s, t])  # one row per pair


def corpus_statistics(corpus: Sequence[Tuple[str, str]]) -> Dict[str, float]:
    """Token count, vocabulary size and mean length of the source side."""
    tokens_src = sum(len(s.split()) for s, _ in corpus)  # total number of words on the incorrect side
    vocab = Counter(t for s, _ in corpus for t in s.split())  # count of every distinct word
    return {
        "pairs": len(corpus),  # number of sentence pairs
        "source_tokens": tokens_src,  # total tokens
        "vocabulary": len(vocab),  # number of distinct words
        "avg_sentence_length": tokens_src / len(corpus) if corpus else 0.0,  # average words per sentence (avoid divide-by-zero)
    }
