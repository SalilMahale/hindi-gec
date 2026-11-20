# -*- coding: utf-8 -*-
"""Hindi Grammar Error Detection and Correction — biography domain.

This package implements a typed GEC system for Hindi biographical prose
(जीवनी). The scope is deliberately narrow: narrating a person's life puts
past-tense verbs, honorifics and a small closed vocabulary in almost every
sentence, which is exactly where Hindi agreement errors live.

The error taxonomy below is the contract shared by every module in this
package — the synthetic generator emits these codes, the detector predicts
BIO tags over them, the edit classifier assigns them, and evaluation is
broken down by them. Nothing outside this list is trained on.
"""
# This file runs whenever any code does "import hindi_gec". It holds the constants every other module shares.

from typing import Final, Tuple  # Final = "this constant must not be reassigned"; Tuple = type hint for a tuple

__version__: Final[str] = "0.1.0"  # version number of our package

#: The seven ⛔ error types of the original taxonomy (PLAN §1.2); EXTENDED_TYPES below adds PRON. Order is fixed so
#: that label-id mappings stay stable across training runs.
ERROR_TYPES: Final[Tuple[str, ...]] = (  # a tuple (unchangeable list) of the 7 error-type codes
    "AGR-GEN",   # gender agreement:    सीता बाजार गया  -> गई
    "AGR-NUM",   # number agreement:    लड़के जाता है   -> जाते हैं
    "AGR-HON",   # honorific agreement: पिताजी आया     -> आए
    "POSTP",     # postposition:        दुनिया में      -> से
    "ERG",       # ergative ने:         मैं खाना खाया  -> मैंने
    "AUX",       # auxiliary / tense:   आएगा था        -> आया
    "SPELL",     # orthography:         पुस्तिक        -> पुस्तक
)

#: New codes EXTEND the ⛔ taxonomy (they never replace it). The model is trained
#: on all 8. PRON = pronoun form: honorific pronoun (उसने -> उन्होंने) and
#: possessive agreement (उनके -> उनका). Scores are also reported on the original
#: 7 types (PRON then counts as OTHER).
NEW_TYPES: Final[Tuple[str, ...]] = ("PRON",)
EXTENDED_TYPES: Final[Tuple[str, ...]] = ERROR_TYPES + NEW_TYPES

#: Emitted for a sentence left deliberately correct (20 % of generated data).
NO_ERROR: Final[str] = "NONE"  # label used for training sentences that contain no error

#: Assigned by the edit classifier to real edits outside the taxonomy.
#: Reported in the error analysis, never used as a training label (PLAN §5.1).
OTHER: Final[str] = "OTHER"  # label for real-world edits that don't fit any of the 7 types

__all__ = ["ERROR_TYPES", "EXTENDED_TYPES", "NEW_TYPES", "NO_ERROR", "OTHER", "__version__"]  # names exported by "from hindi_gec import *"
