# -*- coding: utf-8 -*-
"""Training data plumbing: example format, padding/shift, deterministic order."""

from __future__ import annotations  # modern type-hint syntax

import pytest  # testing framework

from hindi_gec.config import load_config
from hindi_gec.seq2seq_format import encode, load_tokenizer, source_text, special_ids, target_text
from hindi_gec.train_corrector import batches_for_epoch, collate, make_example  # functions under test

CC = load_config()["corrector"]  # corrector settings


@pytest.fixture(scope="module")  # a fixture = shared setup; "module" scope = load the tokenizer only once for this file
def tok():
    try:
        return load_tokenizer(CC["model_name"], local_files_only=True)
    except OSError:
        pytest.skip("IndicBART tokenizer not in the HF cache")  # skip (not fail) if the tokenizer isn't downloaded


def test_make_example_equals_encoding_the_formatted_string(tok) -> None:
    ids = special_ids(tok, CC["lang_tag"])
    src, tgt = "उनका जन्म 1931 में हुई थी ।", "उनका जन्म 1931 में हुआ था ।"
    s, t = make_example(tok, ids, src, tgt, 64, 64)
    assert s == encode(tok, source_text(src, CC["lang_tag"]))  # same as encoding "... </s> <2hi>"
    assert t == encode(tok, target_text(tgt, CC["lang_tag"]))  # same as encoding "<2hi> ... </s>"
    assert s[-2:] == [64001, 64006] and t[0] == 64006 and t[-1] == 64001  # </s>=64001, <2hi>=64006 in the right places


def test_truncation_keeps_specials(tok) -> None:
    ids = special_ids(tok, CC["lang_tag"])
    long = " ".join(["उनका जन्म 1931 में हुआ था"] * 20)  # very long sentence
    s, t = make_example(tok, ids, long, long, 16, 16)  # truncate to 16
    assert len(s) == 16 and s[-2:] == [ids.eos, ids.lang]  # special tokens survive truncation
    assert len(t) == 16 and t[0] == ids.lang and t[-1] == ids.eos


def test_collate_shifts_and_pads() -> None:
    # target [L, 5, 6, E] -> decoder reads [L, 5, 6], labels [5, 6, E]
    batch = [([1, 2, 9, 8], [7, 5, 6, 3]), ([4, 9, 8], [7, 3])]  # two (source, target) examples of different lengths
    b = collate(batch, pad=0, device="cpu")
    assert b["input_ids"].tolist() == [[1, 2, 9, 8], [4, 9, 8, 0]]  # shorter source padded with 0
    assert b["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]  # 0 marks the padding
    assert b["decoder_input_ids"].tolist() == [[7, 5, 6], [7, 0, 0]]  # target without its last token
    assert b["labels"].tolist() == [[5, 6, 3], [3, -100, -100]]  # target without its first token; -100 = ignored


def test_epoch_order_is_deterministic_and_complete() -> None:
    a = batches_for_epoch(10, 3, seed=42, epoch=1)
    assert a == batches_for_epoch(10, 3, seed=42, epoch=1)  # same seed+epoch -> same order
    assert a != batches_for_epoch(10, 3, seed=42, epoch=2)  # different epoch -> different order
    assert sorted(i for b in a for i in b) == list(range(10))  # every example appears exactly once
