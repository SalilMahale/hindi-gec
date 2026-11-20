# -*- coding: utf-8 -*-
"""Rule-based Hindi morphology for the error generator (PLAN §3.1, no Stanza).

The linguistics
---------------
Hindi verbs agree with a noun phrase in **gender** (M/F) and **number**
(sg/pl), and the agreement is spread over a whole *verb group* — the main verb
plus its light verbs and auxiliaries::

    उनका जन्म 1931 में हुआ था ।       जन्म (M sg)  ->  हुआ था   (M sg)
    उन्होंने शिक्षा प्राप्त की थी ।    शिक्षा (F sg) -> की थी    (F sg)
    वे वैज्ञानिक थे ।                 वे (honorific) -> थे      (M pl)

A human who makes an agreement error changes the *whole* group coherently
(हुआ था -> हुई थी), not one stray token, so the generator works on groups:
read the group's (gender, number), pick a new cell, and re-render every token
from its paradigm. Before corrupting, the group is re-rendered at its *own*
cell; if that does not reproduce the original tokens exactly, our analysis is
wrong for this group and it is skipped. That self-check is what lets a
rule-based analyser be trusted without a POS tagger.

Feminine plural is marked once, on the last inflecting element
(की गई थीं, जाती हैं — never *कीं गईं थीं), and the renderer follows that rule.

Two closed-class homographs need context:

* ``की`` is the genitive (उनकी शिक्षा) or the perfective of करना (प्राप्त की).
  It is verbal only when followed by an auxiliary, a passive light verb,
  a danda or a clause boundary.
* ``लिए/लिये`` after a ``-के`` word is the postposition "for" (के लिए).

Honorific plural: Hindi uses plural verbs for one respected person
(गांधी जी कहते थे). In biographies that is the default, so a plural -> singular
flip in an honorific context is an AGR-HON error, not AGR-NUM.

Spelling: Wikipedia tolerates real variant spellings (गांव/गाँव, जिला/ज़िला).
A spelling "error" that is itself a common corpus form would teach the model
to change correct text, so SPELL variants are vetoed when they occur in the
clean corpus (see ``error_generator``).
"""

from __future__ import annotations  # modern type-hint syntax

import re  # regular expressions
from dataclasses import dataclass  # auto-generates simple record classes
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple  # type hints

Cell = Tuple[str, str]            # (gender "M"/"F", number "sg"/"pl")
CELLS: Tuple[Cell, ...] = (("M", "sg"), ("F", "sg"), ("M", "pl"), ("F", "pl"))  # all 4 gender-number combinations

STD, YFAM = "std", "y"            # spelling families: गई/गए vs गयी/गये


# ----------------------------------------------------------------------
# Paradigms
# ----------------------------------------------------------------------
@dataclass(frozen=True)  # frozen = objects can't be changed after creation (safe to share)
class Paradigm:
    """One inflecting lexeme in one spelling family."""

    lexeme: str                   # e.g. "गया", "रहा", "था"
    kind: str                     # "perf" | "impf" | "aux_past" | "aux_pres"
    family: str                   # STD | YFAM
    forms: Tuple[Tuple[Cell, str], ...]  # the 4 forms: ((("M","sg"),"गया"), (("F","sg"),"गई"), ...)

    def form(self, cell: Cell) -> Optional[str]:  # get the word form for a given (gender, number)
        for c, f in self.forms:  # look through the 4 (cell, form) pairs
            if c == cell:  # found the requested cell
                return f  # return its form, e.g. ("F","sg") -> "गई"
        return None  # cell not found


def _cells(msg: str, fsg: str, mpl: str, fpl: str) -> Tuple[Tuple[Cell, str], ...]:  # helper: 4 forms -> table of (cell, form)
    return ((("M", "sg"), msg), (("F", "sg"), fsg), (("M", "pl"), mpl), (("F", "pl"), fpl))  # masc-sg, fem-sg, masc-pl, fem-pl


def _build_paradigms() -> List[Paradigm]:  # build the full list of verb paradigms (our small "dictionary" of verb forms)
    p: List[Paradigm] = []  # will collect all paradigms
    # Auxiliaries. Present है/हैं has no gender: both gender cells share a form.
    p.append(Paradigm("था", "aux_past", STD, _cells("था", "थी", "थे", "थीं")))  # past auxiliary "was/were"
    p.append(Paradigm("है", "aux_pres", STD, _cells("है", "है", "हैं", "हैं")))  # present auxiliary "is/are"

    # Irregular perfectives, both spelling families.
    irregular = {  # lexeme -> (standard spellings, y-spellings), each in order M-sg, F-sg, M-pl, F-pl
        "गया": (("गया", "गई", "गए", "गईं"), ("गया", "गयी", "गये", "गयीं")),  # went
        "हुआ": (("हुआ", "हुई", "हुए", "हुईं"), ("हुआ", "हुयी", "हुये", "हुयीं")),  # happened / was (born)
        "किया": (("किया", "की", "किए", "कीं"), ("किया", "की", "किये", "कीं")),  # did
        "दिया": (("दिया", "दी", "दिए", "दीं"), ("दिया", "दी", "दिये", "दीं")),  # gave
        "लिया": (("लिया", "ली", "लिए", "लीं"), ("लिया", "ली", "लिये", "लीं")),  # took
    }
    for lex, (std, yf) in irregular.items():  # for each irregular verb
        p.append(Paradigm(lex, "perf", STD, _cells(*std)))  # add its standard-spelling paradigm
        p.append(Paradigm(lex, "perf", YFAM, _cells(*yf)))  # and its y-spelling paradigm

    # -आया perfectives (बनाया, निभाया …): stem + ाया/ाई/ाए/ाईं or ाया/ायी/ाये/ायीं.
    # Explicit list: the bare suffix ाई also ends nouns (भाई, पढ़ाई, लड़ाई), and
    # दिखाई/सुनाई are invariant inside दिखाई देना / सुनाई देना, so दिखा/सुना are left out.
    for stem in ("आ", "पा", "बना", "निभा", "बता", "बिता", "चला", "लगा",
                 "करा", "गा", "ला", "उठा", "अपना", "मना", "पहुँचा",
                 "पहुंचा", "हरा", "सिखा", "समझा", "बुला", "जिता", "छपा"):
        std = (stem + "या", stem + "ई", stem + "ए", stem + "ईं")       # बनाया/बनाई/बनाए/बनाईं
        yf = (stem + "या", stem + "यी", stem + "ये", stem + "यीं")     # बनाया/बनायी/बनाये/बनायीं
        p.append(Paradigm(std[0], "perf", STD, _cells(*std)))  # standard family
        p.append(Paradigm(std[0], "perf", YFAM, _cells(*yf)))  # y family

    # Regular consonant-stem perfectives: stem + ा/ी/े/ीं  (रहा/रही/रहे/रहीं).
    for stem in ("रह", "चुक", "सक", "बन", "मिल", "लग", "रख", "पढ़", "लिख", "कह",
                 "मान", "जीत", "हार", "बैठ", "उठ", "निकल", "पहुँच", "पहुंच", "बदल",
                 "खेल", "बोल", "सुन", "समझ", "छोड़", "जोड़", "मर", "जन्म", "देख",
                 "भेज", "बच", "चल", "लौट", "बस", "उतर", "पल", "बढ़", "हट", "टूट"):
        p.append(Paradigm(stem + "ा", "perf", STD,  # e.g. रहा
                          _cells(stem + "ा", stem + "ी", stem + "े", stem + "ीं")))  # रहा / रही / रहे / रहीं

    # Imperfective participles: root + ता/ती/ते/तीं  (जाता/जाती/जाते/जातीं).
    for root in ("जा", "कर", "हो", "रह", "बन", "मिल", "ले", "दे", "लिख", "पढ़",
                 "मान", "जान", "कह", "आ", "रख", "लग", "पा", "सक", "दिखा", "बता",
                 "सिखा", "चल", "खेल", "बोल", "सुन", "समझ", "कहला", "जी", "बिता",
                 "गा", "लड़", "बैठ", "निभा", "चाह", "सोच", "मर", "बच", "उठ",
                 "लौट", "बदल", "पहुँच", "पहुंच", "भेज", "छोड़", "चला", "बना",
                 "पढ़ा", "लिखा", "रचा", "दिख", "बस", "पड़", "पहन", "चुन"):
        p.append(Paradigm(root + "ता", "impf", STD,  # e.g. करता (habitual "does")
                          _cells(root + "ता", root + "ती", root + "ते", root + "तीं")))  # करता / करती / करते / करतीं
    return p  # all paradigms


PARADIGMS: Tuple[Paradigm, ...] = tuple(_build_paradigms())  # build once when the module is imported

#: form -> paradigms containing it (a form like गया sits in both families).
FORM_INDEX: Dict[str, Tuple[Paradigm, ...]] = {}  # reverse lookup: word form -> which paradigms it belongs to
for _p in PARADIGMS:  # each paradigm
    for _c, _f in _p.forms:  # each of its 4 forms
        FORM_INDEX[_f] = FORM_INDEX.get(_f, ()) + (_p,)  # add this paradigm to that form's tuple

#: Uninflected stems that sit inside a verb group (कर दिया, जा सकता, हो गया).
GROUP_GLUE: FrozenSet[str] = frozenset(  # words that don't change form but belong to the verb group
    "जा कर हो दे ले बन पा रह उठ बैठ चल निकल आ लग सुन मिल दिख सक".split()
)

#: What may follow a verbal की (else it is the genitive).
_VERBAL_KI_NEXT: FrozenSet[str] = frozenset(  # if की is followed by one of these, it's the verb "did" (not "of")
    "। , थी थीं है हैं गई गयी गईं गयीं जाती जा और तथा एवं".split()
)

POSTPOSITIONS: FrozenSet[str] = frozenset("में से को पर तक ने का की के".split())  # Hindi postpositions (like English prepositions, but after the noun)

#: Heads of compound postpositions (के बाद, के लिए, के रूप में …). Before these,
#: के is fixed, so के -> का is certainly an error; before an ordinary noun it
#: may be correct (राम का पुत्र / राम के पुत्र "Ram's sons").
COMPOUND_POSTP_HEADS: FrozenSet[str] = frozenset("""
बाद लिए लिये साथ रूप द्वारा बारे पास अनुसार कारण बीच दौरान तहत ऊपर नीचे पहले बजाय
अलावा सामने पीछे विरुद्ध खिलाफ खिलाफ़ ख़िलाफ़ प्रति बाहर अंदर अन्दर उपरांत उपरान्त माध्यम
पश्चात समय यहाँ यहां
""".split())

#: Pronouns and particles that are never content words.
FUNCTION_WORDS: FrozenSet[str] = frozenset("""
और तथा एवं या अथवा कि भी ही तो न नहीं ना एक यह वह ये वे इस उस इन उन जो जिस जिन
किसी कुछ सभी सब बहुत अपने अपनी अपना साथ बाद लिए लिये द्वारा उनका उनकी उनके इनका
इनकी इनके उन्होंने इन्होंने जिन्होंने उन्हें इन्हें उन्हीं इन्हीं उसका उसकी उसके
इसका इसकी इसके उसने इसने जिसने आप मैं मैंने हम हमने वहाँ वहां यहाँ यहां जब तब फिर
लेकिन परंतु परन्तु किंतु किन्तु क्योंकि अतः इसलिए यदि
""".split())

#: Honorific triggers (PLAN §3.1 item 5).
HONORIFIC_TOKENS: FrozenSet[str] = frozenset("""
उन्होंने इन्होंने जिन्होंने वे आप उनका उनकी उनके इनका इनकी इनके उन्हें इन्हें उन्हीं
इन्हीं श्री श्रीमती महात्मा पंडित पण्डित डॉ॰ डॉ डॉक्टर स्वामी गुरु सर जी साहब सरदार
""".split())

_DEVANAGARI_WORD = re.compile(r"^[ऀ-ॣ॰-ॿ]+$")  # matches a word made only of Devanagari letters (no danda)
_YEAR = re.compile(r"^(1[89]\d\d|20[01]\d)$")         # 1800–2019


def analyses(tokens: Sequence[str], i: int) -> Tuple[Paradigm, ...]:
    """Paradigms the token at ``i`` can belong to, with homograph checks."""
    tok = tokens[i]  # the word we're analysing
    cands = FORM_INDEX.get(tok, ())  # which verb paradigms contain this exact form
    if not cands:  # not a known verb form
        return ()
    if tok == "की":  # की is ambiguous: "of" (genitive) or "did" (verb)
        nxt = tokens[i + 1] if i + 1 < len(tokens) else "।"  # the next word (end of sentence counts as ।)
        if nxt not in _VERBAL_KI_NEXT:  # not followed by a verb-ish word
            return ()  # so it's the genitive "of", not a verb
    if tok in ("लिए", "लिये") and i > 0 and tokens[i - 1].endswith("के"):  # "के लिए" = "for", not the verb "took"
        return ()
    return cands  # the possible paradigms


def sentence_family(tokens: Sequence[str]) -> str:
    """YFAM if the sentence already uses a y-spelling (गयी, किये …), else STD."""
    for t in tokens:  # each word
        ps = FORM_INDEX.get(t, ())  # its paradigms
        fams = {p.family for p in ps}  # spelling families it appears in
        if fams == {YFAM}:  # only in the y-family -> the writer uses y-spellings
            return YFAM
    return STD  # default: standard spelling


# ----------------------------------------------------------------------
# Verb groups
# ----------------------------------------------------------------------
@dataclass
class VerbGroup:
    """A contiguous verb group and the analysis of each inflecting token."""

    start: int  # index of the first word of the group
    end: int                                   # half-open
    slots: List[Tuple[int, Paradigm]]          # (token index, chosen paradigm)
    gender: Optional[str]  # "M" / "F" (None if no gendered form, e.g. only है)
    number: str  # "sg" / "pl"


def find_verb_groups(tokens: Sequence[str]) -> List[VerbGroup]:
    """Return verb groups whose analysis re-renders to the original tokens."""
    groups: List[VerbGroup] = []  # result list
    fam = sentence_family(tokens)  # standard or y-spelling for this sentence
    i, n = 0, len(tokens)  # i = current position, n = number of words
    while i < n:  # scan the sentence left to right
        if not analyses(tokens, i):  # this word is not an inflecting verb form
            i += 1
            continue  # move on
        # A group may start one glue stem earlier (हो गया, कर दिया).
        start = i - 1 if i > 0 and tokens[i - 1] in GROUP_GLUE else i  # include a preceding glue word
        j = i  # extend the group to the right
        while j < n and (analyses(tokens, j) or tokens[j] in GROUP_GLUE):  # while words are verb forms or glue
            j += 1
        # Drop trailing glue: "सक" at the end of a group is not inflecting.
        end = j
        while end > i and not analyses(tokens, end - 1):  # trim glue words at the end
            end -= 1
        group = _analyse_group(tokens, start, end, fam)  # work out gender/number and self-check
        if group is not None:  # analysis passed
            groups.append(group)
        i = j  # continue scanning after this group
    return groups


def _pick(ps: Tuple[Paradigm, ...], fam: str, tok: str) -> Optional[Paradigm]:
    """Choose one paradigm for a token, preferring the sentence's family."""
    if not ps:
        return None
    for p in ps:
        if p.family == fam:  # same spelling family as the sentence
            return p
    return ps[0]  # otherwise just take the first


def _analyse_group(tokens: Sequence[str], start: int, end: int, fam: str) -> Optional[VerbGroup]:  # build a VerbGroup or None if unsure
    slots: List[Tuple[int, Paradigm]] = []  # inflecting words of the group
    for k in range(start, end):  # each word in the group
        ps = analyses(tokens, k)  # its possible paradigms
        if ps:  # it inflects
            p = _pick(ps, fam, tokens[k])  # choose one paradigm
            assert p is not None
            slots.append((k, p))  # remember (position, paradigm)
    if not slots:  # nothing inflects
        return None
    # Number from the last inflecting token; gender from the last gendered one.
    last_idx, last_p = slots[-1]  # last inflecting word
    number = _cell_of(last_p, tokens[last_idx])[1]  # its number (sg/pl)
    gender: Optional[str] = None
    for k, p in reversed(slots):  # from the end backwards
        if p.kind != "aux_pres":  # है/हैं carry no gender, skip them
            gender = _cell_of(p, tokens[k])[0]  # gender of the last gendered word
            break
    g = VerbGroup(start, end, slots, gender, number)  # build the group object
    if render_group(tokens, g, gender or "M", number) != list(tokens[start:end]):  # self-check: regenerate at its own gender/number
        return None                            # self-check failed: skip
    return g  # analysis trusted


def _cell_of(p: Paradigm, form: str) -> Cell:
    """The cell a form occupies; for ambiguous forms the first listed wins."""
    for c, f in p.forms:
        if f == form:
            return c  # e.g. "गई" -> ("F","sg")
    raise KeyError(form)  # form not in this paradigm (shouldn't happen)


def render_group(tokens: Sequence[str], g: VerbGroup, gender: str, number: str) -> List[str]:
    """Re-inflect every slot of ``g`` to (gender, number); glue is unchanged.

    Feminine plural is marked only on the last inflecting slot; earlier
    gendered slots take the feminine singular form.
    """
    out = list(tokens[g.start:g.end])  # copy of the group's words
    last = g.slots[-1][0]  # position of the last inflecting word
    for k, p in g.slots:  # each inflecting word
        cell: Cell = (gender, number)  # target gender/number
        if gender == "F" and number == "pl" and k != last:  # Hindi rule: fem-plural only on the last word
            cell = ("F", "sg")  # earlier words use fem-singular (की गई थीं)
        form = p.form(cell)  # look up the new form
        assert form is not None
        out[k - g.start] = form  # replace the word
    return out  # re-inflected group


def has_past_anchor(tokens: Sequence[str]) -> bool:
    """A year 1800–2019 or a life-event word fixes the sentence in the past."""
    return any(_YEAR.match(t) or t in ("जन्म", "निधन", "मृत्यु", "देहांत", "देहान्त")  # year, or birth/death words
               for t in tokens)


def has_present_anchor(tokens: Sequence[str]) -> bool:
    """Words that force present tense (वर्तमान में, आजकल, अभी, संप्रति)."""
    return any(t in ("वर्तमान", "आजकल", "अभी", "संप्रति", "सम्प्रति") for t in tokens)  # "currently/nowadays/now"


def is_honorific_context(tokens: Sequence[str], page_title: str = "") -> bool:
    """True if the sentence refers to someone with honorific plural agreement.

    Triggers: honorific pronouns (वे, उन्होंने, उनके …), titles (श्री, डॉ॰ …),
    any -जी word, or a word of the biography's own page title — in a
    biography, the page's subject is referred to with honorific plural.
    """
    title_words = {w for w in page_title.split() if len(w) >= 2 and _DEVANAGARI_WORD.match(w)}  # Hindi words from the person's name
    for t in tokens:
        if t in HONORIFIC_TOKENS or t in title_words:  # respectful pronoun/title, or the subject's name
            return True
        if len(t) > 3 and t.endswith("जी") and _DEVANAGARI_WORD.match(t):  # e.g. "गांधीजी", "पिताजी"
            return True
    return False


# ----------------------------------------------------------------------
# Spelling confusions (PLAN §3.1 item 7)
# ----------------------------------------------------------------------
SPELL_SWAPS: Tuple[Tuple[str, str], ...] = (  # pairs of vowel signs people commonly confuse
    ("ि", "ी"),   # ि  ↔ ी
    ("ु", "ू"),   # ु  ↔ ू
    ("े", "ै"),   # े  ↔ ै
    ("ो", "ौ"),   # ो  ↔ ौ
    ("ए", "ऐ"),   # ए  ↔ ऐ
    ("ं", "ँ"),   # ं  ↔ ँ   anusvara ↔ chandrabindu
)
NUKTA = "़"  # the dot under letters like ज़, फ़ (often dropped by writers)


def is_content_word(tok: str, closed: Iterable[str] = ()) -> bool:
    """Devanagari-only, ≥ 3 code points, not a function word/postposition/verb form."""
    return (
        len(tok) >= 3  # long enough to be a real content word
        and bool(_DEVANAGARI_WORD.match(tok))  # only Hindi letters
        and tok not in FUNCTION_WORDS  # not a pronoun/particle
        and tok not in POSTPOSITIONS  # not a postposition
        and tok not in FORM_INDEX  # not a verb form we handle elsewhere
        and tok not in closed  # not in any extra excluded list
    )


def spelling_variants(word: str) -> List[str]:
    """Every single-character confusion of ``word`` (one swap or a nukta drop)."""
    out: List[str] = []  # candidate misspellings
    for i, ch in enumerate(word):  # each character with its position
        for a, b in SPELL_SWAPS:  # each confusable pair
            if ch == a:
                out.append(word[:i] + b + word[i + 1:])  # replace a with b at position i
            elif ch == b:
                out.append(word[:i] + a + word[i + 1:])  # replace b with a
        if ch == NUKTA:
            out.append(word[:i] + word[i + 1:])  # drop the nukta
    seen, uniq = set(), []  # remove duplicates while keeping order
    for v in out:
        if v != word and v not in seen:  # skip unchanged and repeated variants
            seen.add(v)
            uniq.append(v)
    return uniq  # e.g. "शिक्षा" -> ["शीक्षा", ...]
