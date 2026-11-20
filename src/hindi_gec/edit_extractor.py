# -*- coding: utf-8 -*-
"""Token alignment -> edits -> error types (PLAN §5.1).

Why one extractor for everything
--------------------------------
Edit-level scoring compares the edits a system made with the edits a human
made. Those two edit sets are only comparable if *both* are cut out of their
sentence pairs by the same procedure — otherwise the score measures
disagreements between two segmenters, not between two corrections. So
metrics never use the generator's stored edits: the reference edits are
``extract_edits(source, target)`` and the system edits are
``extract_edits(source, output)``. The stored generator edits are used only
to check that this classifier agrees with the types we created by
construction.

Alignment
---------
``difflib.SequenceMatcher`` on token lists gives the longest matching blocks;
every non-equal opcode (replace/insert/delete) is one edit. One repair:
two verb-form edits separated only by uninflected light-verb stems
(की जा सकती ← किया जा सकता) are merged, because they are one agreement
error in one verb group — exactly how the generator records it.

Typing (first rule that fires wins)
-----------------------------------
ERG     ने inserted/deleted, or an ergative pronoun ↔ its nominative (उन्होंने ↔ वे)
AUX     an auxiliary changes tense (था ↔ है, थे ↔ हैं), gender/number kept
AGR-*   every token pair is the same verb lexeme in a different cell:
        gender changed -> AGR-GEN; number changed -> AGR-HON when the
        correction is singular -> plural in an honorific context, else AGR-NUM
POSTP   one postposition replaced/inserted/deleted (में से को पर तक का की के)
SPELL   same word up to matra length, nasal sign, nukta, ए/ऐ, ओ/औ
        (or a different spelling family of the same verb form: गये/गए)
PRON    (only with extended=True) one pronoun replaced by the same pronoun in
        another number (उसने -> उन्होंने) or another genitive ending (उनके -> उनका)
OTHER   anything else — reported, never trained on
"""

from __future__ import annotations  # modern type-hint syntax

import difflib  # Python's built-in "diff" tool: finds what changed between two sequences
from typing import FrozenSet, List, Optional, Sequence, Tuple  # type hints

from . import OTHER  # label for edits outside the 7 types
from .error_generator import ERG_PRONOUNS  # उन्होंने -> वे etc.
from .morphology import FORM_INDEX, GROUP_GLUE, Paradigm, is_honorific_context  # verb-form knowledge
from .names import subject_is_name  # bare person name as subject -> honorific
from .other_patterns import pronoun_pattern  # PRON
from .schema import Edit  # Edit record class

POSTPS = frozenset("में से को पर तक का की के".split())  # postpositions recognised by the POSTP rule
_ERG_PAIRS = {(a, b) for a, b in ERG_PRONOUNS.items()} | {(b, a) for a, b in ERG_PRONOUNS.items()}  # (उन्होंने,वे) and (वे,उन्होंने) both count as ERG

# Characters folded away by the spelling key: nukta, both nasal signs.
_SPELL_DROP = str.maketrans("", "", "़ँं")  # translation table that deletes these characters
# Long -> short vowels and ai/au -> e/o, matras and independent vowels.
_SPELL_FOLD = str.maketrans({"ी": "ि", "ू": "ु", "ै": "े",
                             "ौ": "ो", "ऐ": "ए", "औ": "ओ",
                             "ई": "इ", "ऊ": "उ"})  # map each long/ai/au vowel to its short partner


def spell_key(word: str) -> str:
    """Fold the orthographic distinctions people confuse into one key."""
    return word.translate(_SPELL_DROP).translate(_SPELL_FOLD)  # e.g. "शीक्षा" and "शिक्षा" give the same key -> spelling variants


# ----------------------------------------------------------------------
# Verb-form comparison
# ----------------------------------------------------------------------
def _cells(form: str) -> List[Tuple[Paradigm, Tuple[str, str]]]:  # every (paradigm, gender-number cell) this form can be
    return [(p, c) for p in FORM_INDEX.get(form, ()) for c, f in p.forms if f == form]


def _verb_change(a: str, b: str) -> Optional[str]:
    """How verb form ``a`` (wrong) relates to ``b`` (right).

    Returns "tense", "gender", "number", "number+gender", "family", or None.
    """
    best: Optional[str] = None  # best explanation found so far
    for pa, ca in _cells(a):  # every analysis of the wrong form
        for pb, cb in _cells(b):  # every analysis of the right form
            if {pa.kind, pb.kind} == {"aux_past", "aux_pres"}:  # one is था-type, other is है-type
                same_n = ca[1] == cb[1]  # same number?
                if same_n:
                    return "tense"  # only tense changed -> AUX error
                continue
            if pa.lexeme != pb.lexeme or pa.kind != pb.kind:  # different verbs -> not an agreement change
                continue
            g_diff = ca[0] != cb[0] and pa.kind != "aux_pres"  # gender differs (है/हैं have no gender)
            n_diff = ca[1] != cb[1]  # number differs
            if not g_diff and not n_diff:  # same cell
                if a != b:
                    best = best or "family"     # गये vs गए: same cell, other spelling
                continue
            kind = "number+gender" if g_diff and n_diff else ("gender" if g_diff else "number")  # what changed
            if best is None or best == "family" or (kind != "number+gender" and best == "number+gender"):  # prefer the simplest explanation
                best = kind
    return best  # None if the two words aren't forms of the same verb


def classify(
    src: Sequence[str], tgt: Sequence[str], target_tokens: Sequence[str], page_title: str = "",
    names: FrozenSet[str] = frozenset(), tgt_start: Optional[int] = None, extended: bool = False,
) -> str:
    """Type of one edit: ``src`` tokens (wrong) -> ``tgt`` tokens (right).

    ``names`` (person-name gazetteer, names.py) and ``tgt_start`` (where the edit
    starts in the target) let a bare person name as subject count as honorific
    context (शास्त्री … था → थे is AGR-HON). Without them the result is exactly
    the original rule, so stored evaluation numbers reproduce.
    """
    s, t = list(src), list(tgt)  # wrong words, right words
    # ERG
    if (s, t) in (([], ["ने"]), (["ने"], [])):  # ने inserted or deleted
        return "ERG"
    if len(s) == len(t) == 1 and (s[0], t[0]) in _ERG_PAIRS:  # वे <-> उन्होंने
        return "ERG"
    # AUX / AGR on aligned verb forms
    if len(s) == len(t) and s:  # same number of words on both sides
        changes = []  # what changed for each word pair
        for a, b in zip(s, t):  # compare word by word
            if a == b:
                continue  # unchanged word
            if a in GROUP_GLUE and b in GROUP_GLUE:  # a glue word changed -> not a simple inflection
                changes = []
                break
            ch = _verb_change(a, b)  # describe the change
            if ch is None:  # not the same verb
                changes = []
                break
            changes.append(ch)
        if changes:  # all changed words are inflections of the same verbs
            if all(c == "tense" for c in changes):
                return "AUX"  # tense only
            if any("gender" in c for c in changes):
                return "AGR-GEN"  # gender changed
            if any("number" in c for c in changes):
                # Correction makes it plural in an honorific context -> AGR-HON.
                tgt_plural = any(c for p, c in _cells(t[-1]) if c[1] == "pl")  # is the corrected last word plural?
                named = tgt_start is not None and subject_is_name(target_tokens, tgt_start, names)  # subject is a known person
                if tgt_plural and (is_honorific_context(target_tokens, page_title) or named):
                    return "AGR-HON"  # respectful plural
                return "AGR-NUM"  # ordinary number agreement
            if all(c == "family" for c in changes):
                return "SPELL"  # गये -> गए type spelling difference
    # POSTP
    if (len(s) == len(t) == 1 and s[0] in POSTPS and t[0] in POSTPS) or \
       (not s and len(t) == 1 and t[0] in POSTPS) or (not t and len(s) == 1 and s[0] in POSTPS):  # postposition replaced, inserted or deleted
        return "POSTP"
    # SPELL
    if len(s) == len(t) and s and all(a == b or spell_key(a) == spell_key(b) for a, b in zip(s, t)):  # words differ only in confusable vowels/nasals
        return "SPELL"
    # PRON (extended taxonomy only)
    if extended and len(s) == len(t) == 1 and pronoun_pattern(s[0], t[0]) in ("PRON-HON", "POSS-AGR"):
        return "PRON"
    return OTHER  # none of the rules fired


# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------
def _is_verb_edit(src: Sequence[str], tgt: Sequence[str]) -> bool:  # True if every changed word is an inflection of the same verb
    return len(src) == len(tgt) and bool(src) and all(
        a == b or _verb_change(a, b) is not None for a, b in zip(src, tgt))


def extract_edits(
    source: Sequence[str], target: Sequence[str], page_title: str = "", names: FrozenSet[str] = frozenset(),
    extended: bool = False,
) -> List[Edit]:
    """Align two token lists and return typed edits (half-open spans); ``names``, ``extended``: see classify."""
    sm = difflib.SequenceMatcher(a=list(source), b=list(target), autojunk=False)  # align the two word lists (autojunk off = treat all words equally)
    spans = [(i1, i2, j1, j2) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]  # keep only changed regions: source[i1:i2] -> target[j1:j2]
    # Merge verb-group edits separated only by light-verb glue (same length gap).
    merged: List[Tuple[int, int, int, int]] = []
    for sp in spans:
        if merged:  # there is a previous edit
            pi1, pi2, pj1, pj2 = merged[-1]  # previous edit
            i1, i2, j1, j2 = sp  # current edit
            gap = list(source[pi2:i1])  # unchanged words between them
            if (gap and i1 - pi2 == j1 - pj2 and all(g in GROUP_GLUE for g in gap)  # gap consists only of glue words
                    and _is_verb_edit(source[pi1:pi2], target[pj1:pj2])  # both edits are verb inflection changes
                    and _is_verb_edit(source[i1:i2], target[j1:j2])):
                merged[-1] = (pi1, i2, pj1, j2)  # join them into one edit covering the whole verb group
                continue
        merged.append(sp)
    edits = []
    for i1, i2, j1, j2 in merged:  # build Edit objects
        s, t = list(source[i1:i2]), list(target[j1:j2])  # wrong words, right words
        edits.append(Edit(src_span=[i1, i2], tgt_span=[j1, j2], src=" ".join(s), tgt=" ".join(t),
                          type=classify(s, t, target, page_title, names, j1, extended)))  # with its error type
    return edits
