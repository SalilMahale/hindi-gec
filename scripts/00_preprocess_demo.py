# -*- coding: utf-8 -*-
"""Step-by-step preprocessing walkthrough on the HiWikEd parallel corpus.

This is the console driver that used to live at the bottom of
`hindi_preprocessing.py`. It is kept as a script (not as library code) because
its job is to *show* each preprocessing stage for the experiment write-up:
raw text -> denoised -> normalised -> segmented -> tokenised -> filtered ->
stemmed, followed by corpus-level statistics.

Run:  python scripts/00_preprocess_demo.py
Reads:  paths.hiwiked_src / paths.hiwiked_trg from configs/config.yaml
Writes: paths.hiwiked_csv (utf-8-sig, so Excel renders Devanagari)
"""

from __future__ import annotations  # modern type-hint syntax

import sys  # system settings (stdout, import path)
from pathlib import Path  # file paths

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # let the Windows console print Devanagari
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # make the hindi_gec package (in src/) importable

from collections import Counter  # noqa: E402  # counting dictionary

from hindi_gec.config import load_config  # noqa: E402  # reads config.yaml
from hindi_gec.preprocessing import (  # noqa: E402  # every preprocessing step we want to demonstrate
    INDIC_NLP,
    analysis_pipeline,
    build_parallel_corpus,
    corpus_statistics,
    drop_punctuation,
    export_corpus_csv,
    load_corpus,
    normalize,
    remove_noise,
    remove_stopwords,
    split_sentences,
    stem,
    tokenize,
)


def head(title: str) -> None:
    """Print a section banner so each stage is visible in the transcript."""
    print("\n" + "=" * 72)  # line of 72 "=" characters
    print(title)
    print("=" * 72)


def main() -> None:
    cfg = load_config()  # load configuration
    src_path, trg_path = cfg.path("hiwiked_src"), cfg.path("hiwiked_trg")  # HiWikEd incorrect / corrected files
    out_csv = cfg.path("hiwiked_csv")  # where to write the cleaned CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)  # create its folder

    print("indic-nlp-library available :", INDIC_NLP)  # True if the Indic NLP library is installed
    print("Source file                 :", src_path)
    print("Target file                 :", trg_path)

    pairs = load_corpus(src_path, trg_path)  # list of (incorrect, corrected) pairs
    print("Sentence pairs loaded       :", len(pairs))

    # A corpus style sentence deliberately polluted to demonstrate cleaning.
    demo = ("<p>भारत   में ५ बड़े नगर हैं। "
            "See https://hi.wikipedia.org for details</p>")  # contains HTML, extra spaces, a Hindi digit, English, a URL

    head("STEP 1 : RAW INPUT")
    print(demo)

    head("STEP 2 : AFTER NOISE REMOVAL")
    denoised = remove_noise(demo)  # HTML, URL and English removed
    print(denoised)

    head("STEP 3 : AFTER UNICODE NORMALISATION")
    normalised = normalize(denoised)  # ५ becomes 5, characters in canonical form
    print(normalised)

    head("STEP 4 : SENTENCE SEGMENTATION")
    para = "राम बाजार गया। सीता घर पर पढ़ रही है। क्या तुम आओगे?"  # paragraph with 3 sentences
    for i, s in enumerate(split_sentences(normalize(para)), 1):  # split on । and ?
        print(f"  S{i} : {s}")

    head("STEP 5 : WORD TOKENISATION")
    toks = tokenize(normalised)  # split into words + punctuation
    print(toks)

    head("STEP 6 : PUNCTUATION REMOVAL")
    toks = drop_punctuation(toks)
    print(toks)

    head("STEP 7 : STOP WORD REMOVAL")
    content = remove_stopwords(toks)  # remove में, हैं etc.
    print(content)

    head("STEP 8 : LIGHT STEMMING")
    for w in content:
        print(f"  {w}  ->  {stem(w)}")  # show each word and its stem

    head("STEP 9 : CLEANED PARALLEL CORPUS  (first 5 pairs shown below)")
    corpus, stats = build_parallel_corpus(  # clean the whole HiWikEd corpus
        pairs,
        min_len=cfg["preprocessing"]["min_tokens"],  # 3
        max_len=cfg["preprocessing"]["max_tokens"],  # 60
    )
    for i, (s, t) in enumerate(corpus[:5], 1):  # show the first 5 pairs
        print(f"\n  Pair {i}")
        print(f"    Incorrect : {s[:100]}")  # first 100 characters
        print(f"    Corrected : {t[:100]}")

    export_corpus_csv(corpus, out_csv)  # save everything to CSV
    print(f"\n  Full cleaned corpus ({len(corpus)} pairs) written to : {out_csv}")

    head("STEP 10 : CORPUS STATISTICS")
    print(f"  Pairs read from files     : {len(pairs)}")
    print(f"  Pairs after cleaning      : {len(corpus)}")
    for reason, count in stats.most_common():  # rejection reasons, most frequent first
        print(f"    rejected - {reason:22s}: {count}")

    numbers = corpus_statistics(corpus)  # token count, vocabulary, average length
    stems = Counter(st for _, t in corpus[:2000] for st in analysis_pipeline(t))  # stem frequencies over the first 2000 corrected sentences
    print(f"  Total tokens (source side): {numbers['source_tokens']}")
    print(f"  Vocabulary size           : {numbers['vocabulary']}")
    print(f"  Average sentence length   : {numbers['avg_sentence_length']:.2f} tokens")
    print(f"  Unique stems (2000 pairs) : {len(stems)}")
    print(f"  Top 5 content stems       : {stems.most_common(5)}")


if __name__ == "__main__":  # run main() only when executed as a script
    main()
