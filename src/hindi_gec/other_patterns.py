# -*- coding: utf-8 -*-
"""What are the real edits the 7 types miss? Rule-based sub-typing of OTHER edits.

42.6 % of real HiWikEd-bio edits fall outside the taxonomy. Before any new
error type is generated, this module sorts those OTHER edits into linguistic
patterns, so that only patterns with real support are implemented. It is a
measuring tool: it runs on HiWikEd **dev** only (never test).

Patterns (first rule that fires wins)
-------------------------------------
Pronouns. Hindi third-person pronouns come in a distal series (वह / वे, उस- / उन-),
a proximal series (यह / ये, इस- / इन-), a relative series (जो, जिस- / जिन-) and
the reflexive अपना. Each form has a number (singular / plural-or-honorific)
and a case (nominative, ergative -ने, genitive -का/की/के, dative -े/-हें …).

    PRON-HON     same series and case, number changed: उसने → उन्होंने, उसकी → उनकी
                 (a respected person is referred to with the plural pronoun)
    POSS-AGR     same pronoun, genitive ending changed: उनके → उनका, अपना → अपनी
                 (a possessive agrees with the possessed noun, like का/की/के)
    PRON-PROX    proximal ↔ distal, same number and case: इन्होंने → उन्होंने
    PRON-OTHER   any other pronoun ↔ pronoun change

Inflection of a word outside the verb paradigms, same stem, final vowel
ा / ी / े (± nasal) changed:
    AGR-OBJ      at a verb position (followed by an auxiliary, a light verb or
                 the danda) in a ने clause: the verb agrees with the object
    VERB-AGR     at a verb position without ने
    NOUN-OBL     followed by a postposition: oblique form (जिला → जिले में)
    AGR-ADJ      otherwise: an adjective or participle agreeing with its noun
                 (दूसरा → दूसरे, वाली → वाला)
    NOUN-INFL    same stem, plural / oblique-plural suffix (कार्य → कार्यों, पुस्तक → पुस्तकें)
    VERB-FORM    both words are verb or auxiliary forms (गया → था, है → रहे)

Everything else:
    ORTH         spelling outside the SPELL rule: ≤ 2 character edits, all of
                 them signs, vowels, halant or a confusable consonant pair
                 (कृष्णा → कृष्ण, गृह → ग्रह, बन → वन)
    FACT         a number, month or digit changed (सितंबर → दिसंबर): content, not grammar
    WORD-CHOICE  one word replaced by a different word (लेखक → कवि)
    INSERT / DELETE / MULTI / PUNCT   word inserted, deleted, multi-word rewrite, punctuation
"""

from __future__ import annotations  # modern type-hint syntax

import re  # regular expressions
from typing import Dict, Optional, Sequence, Tuple  # type hints

from .error_generator import MONTHS  # month names
from .morphology import FORM_INDEX, GROUP_GLUE, POSTPOSITIONS  # verb forms, glue, postpositions

# pronoun form -> (series, number, case)
PRONOUNS: Dict[str, Tuple[str, str, str]] = {}


def _add(series: str, number: str, forms: Dict[str, str]) -> None:
    for case, words in forms.items():
        for w in words.split():
            PRONOUNS[w] = (series, number, case)


_add("dist", "sg", {"nom": "वह वो", "erg": "उसने", "gen": "उसका उसकी उसके", "dat": "उसे उसको", "obl": "उस उससे उसमें उसपर"})
_add("dist", "pl", {"nom": "वे", "erg": "उन्होंने उन्होने", "gen": "उनका उनकी उनके", "dat": "उन्हें उनको",
                    "obl": "उन उनसे उनमें उनपर"})
_add("prox", "sg", {"nom": "यह", "erg": "इसने", "gen": "इसका इसकी इसके", "dat": "इसे इसको", "obl": "इस इससे इसमें इसपर"})
_add("prox", "pl", {"nom": "ये", "erg": "इन्होंने इन्होने", "gen": "इनका इनकी इनके", "dat": "इन्हें इनको",
                    "obl": "इन इनसे इनमें इनपर"})
_add("rel", "sg", {"nom": "जो", "erg": "जिसने", "gen": "जिसका जिसकी जिसके", "dat": "जिसे जिसको", "obl": "जिस जिससे जिसमें"})
_add("rel", "pl", {"erg": "जिन्होंने जिन्होने", "gen": "जिनका जिनकी जिनके", "dat": "जिन्हें जिनको", "obl": "जिन जिनसे जिनमें"})
_add("refl", "-", {"gen": "अपना अपनी अपने"})

AUX_NEXT = frozenset("था थी थे थीं है हैं हूँ हो गया गई गए गयी गये गईं रहा रही रहे चुका चुकी चुके । ,".split()) | GROUP_GLUE
ERGATIVE = frozenset("ने उन्होंने इन्होंने जिन्होंने उसने इसने जिसने मैंने हमने आपने".split())
_END = re.compile(r"^(.*?)(ा|ी|े|ीं|ें|ां)$")  # word = stem + final vowel (± nasal)
_NOUN_SUFFIX = ("ियों", "ियाँ", "ियां", "ाओं", "ाएँ", "ाएं", "ओं", "ों", "ें", "एँ", "एं", "यों", "याँ")
_ORTH_CHARS = set("ािीुूृेैोौंँःॅॉ़्अआइईउऊएऐओऔऑ")
_CONFUSABLE = [{"ब", "व"}, {"श", "ष", "स"}, {"ण", "न"}, {"ड", "ड़"}, {"ढ", "ढ़"}, {"य", "ए"}, {"र", "्र"}]
_NUMBER_WORDS = frozenset("एक दो तीन चार पाँच पांच छह छः सात आठ नौ दस सौ हज़ार हजार लाख करोड़ पहला पहली पहले "
                          "दूसरा दूसरी तीसरा तीसरी चौथा".split())
_PUNCT = re.compile(r"^[^\wऀ-ॣ०-ॿ]+$")


def char_edits(a: str, b: str) -> Tuple[int, set]:
    """Levenshtein distance between two words, and the set of characters it touched."""
    prev = [(j, set(b[:j])) for j in range(len(b) + 1)]
    for i, x in enumerate(a, 1):
        cur = [(i, set(a[:i]))]
        for j, y in enumerate(b, 1):
            opts = [(prev[j][0] + 1, prev[j][1] | {x}), (cur[j - 1][0] + 1, cur[j - 1][1] | {y}),
                    (prev[j - 1][0] + (x != y), prev[j - 1][1] | ({x, y} if x != y else set()))]
            cur.append(min(opts, key=lambda o: o[0]))
        prev = cur
    return prev[-1]


def _orthographic(a: str, b: str) -> bool:
    d, touched = char_edits(a, b)
    if d == 0 or d > 2 or max(len(a), len(b)) < 3:
        return False
    if touched <= _ORTH_CHARS:
        return True
    rest = touched - _ORTH_CHARS
    return any(rest <= pair for pair in _CONFUSABLE)


def pronoun_pattern(a: str, b: str) -> Optional[str]:
    pa, pb = PRONOUNS.get(a), PRONOUNS.get(b)
    if not pa or not pb:
        return None
    (sa, na, ca), (sb, nb, cb) = pa, pb
    if sa == sb and ca == cb and na != nb:
        return "PRON-HON"
    if sa == sb and na == nb and ca == cb == "gen":
        return "POSS-AGR"
    if {sa, sb} == {"dist", "prox"} and na == nb and ca == cb:
        return "PRON-PROX"
    return "PRON-OTHER"


def pattern(src: Sequence[str], tgt: Sequence[str], target: Sequence[str], tgt_start: int) -> str:
    """Pattern of one OTHER edit: ``src`` (wrong) -> ``tgt`` (right) at ``tgt_start`` in ``target``."""
    s, t = list(src), list(tgt)
    if all(_PUNCT.match(w) for w in s + t):
        return "PUNCT"
    if not s:
        return "INSERT"
    if not t:
        return "DELETE"
    if len(s) != 1 or len(t) != 1:
        return "MULTI"
    a, b = s[0], t[0]
    p = pronoun_pattern(a, b)
    if p:
        return p
    if a in FORM_INDEX and b in FORM_INDEX:
        return "VERB-FORM"
    ma, mb = _END.match(a), _END.match(b)
    if ma and mb and ma.group(1) == mb.group(1) and len(ma.group(1)) >= 2:  # same stem, other final vowel
        end = tgt_start + 1
        nxt = target[end] if end < len(target) else "।"
        if nxt in AUX_NEXT:
            return "AGR-OBJ" if any(w in ERGATIVE for w in target[:tgt_start]) else "VERB-AGR"
        if nxt in POSTPOSITIONS:
            return "NOUN-OBL"
        return "AGR-ADJ"
    for x, y in ((a, b), (b, a)):  # noun plural / oblique plural
        if any(y == x + suf or (x[-1:] in "ाीे" and y == x[:-1] + suf) for suf in _NOUN_SUFFIX):
            return "NOUN-INFL"
    if any(c.isdigit() for c in a + b) or a in MONTHS or b in MONTHS or a in _NUMBER_WORDS or b in _NUMBER_WORDS:
        return "FACT"
    if _orthographic(a, b):
        return "ORTH"
    return "WORD-CHOICE"
