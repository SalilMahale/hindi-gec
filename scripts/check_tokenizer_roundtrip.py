# -*- coding: utf-8 -*-
"""Step 0 gate: IndicBART tokenizer round-trip on every bio_clean sentence.

For each sentence, both the raw ``text`` and the space-joined ``tokens`` form
are encoded as source ("... </s> <2hi>") and as target ("<2hi> ... </s>"),
decoded, and compared with the original. Any mismatch is printed with the
characters that differ, because a lossy tokenizer would make the corrector
learn to "fix" tokenizer damage instead of grammar.

Also reports how often ``<unk>`` appears and the sub-word length distribution,
which sets ``max_source_len`` sensibly.

Run:  .venv\\Scripts\\python.exe scripts\\check_tokenizer_roundtrip.py
"""

from __future__ import annotations  # modern type-hint syntax

import sys  # stdout, import path
import unicodedata  # character names for error messages
from collections import Counter  # counting dictionary
from pathlib import Path  # file paths

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # print Hindi
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # make hindi_gec importable

from hindi_gec.config import load_config  # noqa: E402
from hindi_gec.schema import read_jsonl  # noqa: E402
from hindi_gec.seq2seq_format import (  # noqa: E402
    encode, load_tokenizer, round_trip, source_text, special_ids,
)


def first_diff(a: str, b: str) -> str:
    """Describe the first differing character between two strings."""
    for i, (x, y) in enumerate(zip(a, b)):  # compare character by character
        if x != y:
            return f"pos {i}: U+{ord(x):04X} {unicodedata.name(x, '?')} -> U+{ord(y):04X} {unicodedata.name(y, '?')}"  # e.g. "U+093F DEVANAGARI VOWEL SIGN I -> ..."
    return f"length {len(a)} -> {len(b)}"  # same prefix, different length


def main() -> None:
    cfg = load_config()
    cc = cfg["corrector"]
    lang = cc["lang_tag"]  # "<2hi>"
    tok = load_tokenizer(cc["model_name"], local_files_only=True)  # IndicBART tokenizer
    ids = special_ids(tok, lang)  # special token ids
    rows = list(read_jsonl(cfg.path("bio_clean")))  # all clean sentences
    print(f"  sentences        : {len(rows)}")
    print(f"  special ids      : bos {ids.bos}  eos {ids.eos}  pad {ids.pad}  {lang} {ids.lang}")

    for form in ("text", "tokens"):  # test both the original text and the tokenised form
        fails, unk_sents, lengths = [], 0, []
        for r in rows:
            s = r["text"] if form == "text" else " ".join(r["tokens"])
            src_out, tgt_out = round_trip(tok, s, lang)  # encode -> decode both ways
            if src_out != s or tgt_out != s:  # didn't come back identical
                fails.append((r["id"], s, src_out if src_out != s else tgt_out))
            src_ids = encode(tok, source_text(s, lang))
            lengths.append(len(src_ids))  # number of sub-words
            if tok.unk_token_id in src_ids:  # contains an unknown token
                unk_sents += 1
        lengths.sort()
        pct = lambda q: lengths[min(len(lengths) - 1, int(q * len(lengths)))]  # percentile helper (q = 0.5 -> median)
        over = sum(1 for n in lengths if n > cc["max_source_len"])  # sentences longer than 64 sub-words
        print(f"\n--- form = {form!r} ---")
        print(f"  round-trip OK    : {len(rows) - len(fails)} / {len(rows)}")
        print(f"  round-trip FAIL  : {len(fails)}")
        print(f"  sentences w/ unk : {unk_sents}")
        print(f"  source sub-words : median {pct(0.5)}  p95 {pct(0.95)}  p99 {pct(0.99)}  max {lengths[-1]}")
        print(f"  > max_source_len ({cc['max_source_len']}) : {over}")
        kinds = Counter(first_diff(s, o) .split(': ', 1)[-1] for _, s, o in fails)  # group failures by kind of character change
        for kind, n in kinds.most_common(8):
            print(f"    {n:6d} x {kind}")
        for rid, s, o in fails[:3]:  # show 3 failing examples
            print(f"    [{rid}] in : {s}\n    {' ' * len(rid)}   out: {o}")


if __name__ == "__main__":
    main()
