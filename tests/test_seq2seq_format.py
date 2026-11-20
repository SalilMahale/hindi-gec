# -*- coding: utf-8 -*-
"""IndicBART format and tokenizer round-trip (PLAN §4.3).

The expected ids come straight from the ``ai4bharat/IndicBART`` model card,
which prints them next to its examples, and from ``added_tokens.json``
(``<s>`` 64000, ``</s>`` 64001, ``<2en>`` 64004, ``<2hi>`` 64006).

Tokenizer tests need the ~2 MB tokenizer files in the HF cache
(``scripts/download_indicbart.py --tokenizer-only``); they are skipped, not
failed, on a machine without them. The pure formatting tests always run.
"""

from __future__ import annotations  # modern type-hint syntax

import unicodedata  # NFC normalisation

import pytest  # testing framework

from hindi_gec.config import load_config
from hindi_gec.seq2seq_format import (  # functions under test
    load_tokenizer,
    round_trip,
    source_text,
    special_ids,
    target_text,
    teacher_forcing,
    encode,
)

CFG = load_config()
MODEL = CFG["corrector"]["model_name"]  # "ai4bharat/IndicBART"
LANG = CFG["corrector"]["lang_tag"]  # "<2hi>"


@pytest.fixture(scope="module")  # load the tokenizer once for all tests in this file
def tok():
    try:
        return load_tokenizer(MODEL, local_files_only=True)
    except OSError:
        pytest.skip("IndicBART tokenizer not in the HF cache")  # skip if not downloaded


# ------------------------------------------------------------ pure format
def test_source_puts_language_tag_last() -> None:
    assert source_text("उनका जन्म 1931 में हुआ था ।", "<2hi>") == \
        "उनका जन्म 1931 में हुआ था । </s> <2hi>"  # encoder format: sentence </s> <2hi>


def test_target_puts_language_tag_first() -> None:
    assert target_text("उनका जन्म 1931 में हुआ था ।", "<2hi>") == \
        "<2hi> उनका जन्म 1931 में हुआ था । </s>"  # decoder format: <2hi> sentence </s>


def test_teacher_forcing_shifts_by_one() -> None:
    # [<2hi>, w1, w2, </s>] -> decoder reads [<2hi>, w1, w2], predicts [w1, w2, </s>]
    dec_in, labels = teacher_forcing([64006, 10, 20, 64001])
    assert dec_in == [64006, 10, 20]
    assert labels == [10, 20, 64001]


# ------------------------------------------------------------ tokenizer
def test_special_ids_match_model_not_tokenizer_config(tok) -> None:
    ids = special_ids(tok, LANG)
    assert (ids.bos, ids.eos, ids.lang) == (64000, 64001, 64006)  # ids the model was trained with
    # The trap: the tokenizer's configured eos is [SEP], not </s>.
    assert tok.eos_token == "[SEP]" and tok.eos_token_id != ids.eos


def test_model_card_hindi_target_ids(tok) -> None:
    # Model card: tensor([[64006, 942, 43, 32720, 8384, 64001]])
    assert encode(tok, "<2hi> मैं  एक लड़का हूँ </s>") == [64006, 942, 43, 32720, 8384, 64001]  # matches the official example


def test_model_card_english_source_ids(tok) -> None:
    # Model card: tensor([[466, 1981, 80, 25573, 64001, 64004]])
    assert encode(tok, "I am a boy </s> <2en>") == [466, 1981, 80, 25573, 64001, 64004]


@pytest.mark.parametrize("sentence", [  # run the test below once for each sentence
    "उनका जन्म 1931 में रामेश्वरम में हुआ था ।",           # PLAN §2.5 example
    "उन्होंने अपनी प्रारंभिक शिक्षा गाँव के विद्यालय से प्राप्त की ।",  # chandrabindu, anusvara
    "वे ज़िला परिषद के सदस्य नियुक्त किए गए थे ।",          # nukta (decomposed after NFC)
])
def test_round_trip_returns_sentence_unchanged(tok, sentence: str) -> None:
    sentence = unicodedata.normalize("NFC", sentence)   # as the pipeline stores it
    src_out, tgt_out = round_trip(tok, sentence, LANG)  # encode + decode
    assert src_out == sentence  # nothing lost on the encoder side
    assert tgt_out == sentence  # nothing lost on the decoder side
