# -*- coding: utf-8 -*-
"""IndicBART input/output format — one place, shared by training and inference.

IndicBART (an mBART-architecture model) was pre-trained with a fixed layout,
and a fine-tuned corrector only works if we feed it exactly that layout
(model card, ``ai4bharat/IndicBART``):

    encoder input  :  "<sentence> </s> <2hi>"     language tag LAST
    decoder target :  "<2hi> <sentence> </s>"     language tag FIRST

The language tag tells one multilingual model which language it is reading and
which it must write. For GEC both sides are Hindi, so both tags are ``<2hi>``.
Generation starts the decoder from ``<2hi>`` (``decoder_start_token_id``).
The card's own Hindi generation example passes ``<2en>`` there — a copy-paste
slip in the card; for Hindi output it must be ``<2hi>``.

Three traps this module exists to avoid:

1. **The tokenizer's own special tokens are wrong for this model.** Its
   ``tokenizer_config.json`` (inherited from the Albert tokenizer class) says
   ``eos_token = "[SEP]"`` and ``bos_token = "[CLS]"``, but the model was
   trained with ``</s>`` (id 64001) and ``<s>`` (id 64000) — which is also what
   the model's ``config.json`` says. So ``tokenizer.eos_token_id`` must never be
   used; ids are looked up by token string instead.
2. **``add_special_tokens=False`` always.** We write ``</s>`` and ``<2hi>``
   into the text ourselves; letting the tokenizer add ``[CLS]``/``[SEP]`` too
   would put tokens the model never saw into every example.
3. **Teacher forcing is done by hand.** With target ids ``t = [<2hi>, w1 … wn,
   </s>]`` the decoder reads ``t[:-1]`` and is scored against ``t[1:]``: at
   every step it sees the words so far and must predict the next one.

The round-trip check (PLAN §4.3) guards everything downstream: if encode →
decode does not return a clean sentence unchanged, the corrector would be
"correcting" tokenizer damage and every metric would be meaningless.
"""

from __future__ import annotations  # modern type-hint syntax

from dataclasses import dataclass  # simple record class
from typing import Any, List, Sequence, Tuple  # type hints

#: Token strings fixed by IndicBART's pre-training (see added_tokens.json).
BOS: str = "<s>"  # beginning-of-sentence token
EOS: str = "</s>"  # end-of-sentence token
PAD: str = "<pad>"  # padding token (fills shorter sentences in a batch)


@dataclass(frozen=True)
class SpecialIds:
    """Ids of the tokens the model was actually trained with."""

    bos: int  # id of <s>
    eos: int  # id of </s> (64001)
    pad: int  # id of <pad> (0)
    lang: int  # id of the language tag <2hi> (64006)


def load_tokenizer(model_name: str, local_files_only: bool = False) -> Any:
    """Load IndicBART's SentencePiece tokenizer with the model card's flags.

    ``use_fast=False``: IndicBART ships only a SentencePiece ``spiece.model``
    (slow Albert tokenizer). ``keep_accents=True``: without it the Albert
    normaliser strips combining marks — in Devanagari those are matras and
    the virama, i.e. it would delete half of every word.
    """
    from transformers import AutoTokenizer  # Hugging Face: loads the right tokenizer class automatically

    return AutoTokenizer.from_pretrained(  # load tokenizer files from a folder or the HF hub
        model_name,  # folder path or hub name
        do_lower_case=False,  # don't lowercase (meaningless for Hindi, harmful for Latin names)
        use_fast=False,  # use the slow SentencePiece tokenizer (the only one IndicBART has)
        keep_accents=True,  # keep matras / virama (otherwise Hindi words get destroyed)
        local_files_only=local_files_only,  # True = never download (offline demo)
    )


def special_ids(tokenizer: Any, lang_tag: str) -> SpecialIds:
    """Look up ids by token string — never via ``tokenizer.eos_token_id``."""
    conv = tokenizer._convert_token_to_id_with_added_voc  # function: token string -> id (includes added tokens like <2hi>)
    ids = SpecialIds(bos=conv(BOS), eos=conv(EOS), pad=conv(PAD), lang=conv(lang_tag))  # look up all 4 ids
    unk = tokenizer.unk_token_id  # id used for unknown tokens
    for name, value in vars(ids).items():  # check each special id
        assert value != unk, f"special token {name!r} maps to <unk>; wrong tokenizer?"  # if it's <unk>, we loaded the wrong tokenizer
    return ids


def source_text(sentence: str, lang_tag: str) -> str:
    """Encoder side: sentence, then end-of-sentence, then language tag."""
    return f"{sentence} {EOS} {lang_tag}"  # e.g. "वे निदेशक है । </s> <2hi>"


def target_text(sentence: str, lang_tag: str) -> str:
    """Decoder side: language tag, then sentence, then end-of-sentence."""
    return f"{lang_tag} {sentence} {EOS}"  # e.g. "<2hi> वे निदेशक हैं । </s>"


def encode(tokenizer: Any, text: str) -> List[int]:
    """Token ids for an already-formatted string, no automatic specials."""
    return tokenizer(text, add_special_tokens=False).input_ids  # text -> list of subword ids


def decode(tokenizer: Any, ids: Sequence[int]) -> str:
    """Ids back to text; specials dropped, spacing left exactly as decoded."""
    return tokenizer.decode(
        list(ids), skip_special_tokens=True, clean_up_tokenization_spaces=False  # drop <s> </s> <2hi> <pad>; don't alter spacing
    ).strip()  # remove leading/trailing spaces


def teacher_forcing(target_ids: Sequence[int]) -> Tuple[List[int], List[int]]:
    """Split target ids into (decoder input, labels): ``t[:-1]`` and ``t[1:]``."""
    t = list(target_ids)
    assert len(t) >= 2, "a target needs at least the language tag and </s>"
    return t[:-1], t[1:]  # decoder sees all but the last token; must predict all but the first (i.e. the next token each step)


def round_trip(tokenizer: Any, sentence: str, lang_tag: str) -> Tuple[str, str]:
    """Encode as source and as target, decode both; return (src_out, tgt_out).

    A clean sentence must come back identical from both sides.
    """
    src_out = decode(tokenizer, encode(tokenizer, source_text(sentence, lang_tag)))  # encoder format -> ids -> text
    tgt_out = decode(tokenizer, encode(tokenizer, target_text(sentence, lang_tag)))  # decoder format -> ids -> text
    return src_out, tgt_out  # both should equal the original sentence
