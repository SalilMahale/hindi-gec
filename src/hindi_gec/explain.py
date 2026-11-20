# -*- coding: utf-8 -*-
"""A short bilingual reason for every edit, one rule template per type.

What an explanation may claim
-----------------------------
The system has no dictionary of noun genders, so it cannot know on its own that
परिवार is masculine. What it does know exactly is the verb paradigm: चला आया is
the masculine singular cell of चलना + आना. Hindi agreement says the verb takes
the gender and number of its controller, so the explanation states the verb's
new cell and names the controller only when a rule finds it:

* in a ने clause the verb agrees with the object (उन्होंने पुस्तक लिखी), taken
  as the content word right before the verb group;
* otherwise the subject: a sentence-initial pronoun (वे, वह, उन्होंने …) or the
  first unmarked noun phrase (names.subject_run).

A word is never named as a subject or object if it cannot be one: time words
(आज, वर्ष, दौरान), connectives and compound-postposition heads (वजह, अनुसार,
बाद), verb forms (किए, करने), month names and numbers (NOT_SUBJECT,
is_subject_word). Then the template says "the subject" without naming it.

AGR-HON is stricter, because it calls the subject "one respected person": it
names only a clear person (person_subject): an honorific pronoun (वे, ये, आप,
उन्होंने …) that is not followed by a postposition, gazetteer name words, or a
person noun or title (पिता, अध्यक्ष, श्री …, a word ending in -जी). With no
clear person it falls back to a neutral reason: the verb must match its
subject in number.

Templates (slots in braces)
---------------------------
AGR-GEN / AGR-NUM  '{src}' is the {cell} form, but {controller} is {gender|number},
                   so the verb is '{tgt}' ({cell})
AGR-HON            {person} is one respected person; Hindi uses the plural for
                   respect, so '{src}' becomes '{tgt}' (no clear person: the verb
                   must match its subject in number …)
POSTP              genitive का/की/के agrees with the next noun; other
                   postpositions: '{tgt}' ({meaning}) fits after '{noun}'
ERG                a past (perfective) transitive verb '{verb}' makes the subject take ने
AUX                the sentence is in the past/present ('{anchor}'), so the
                   auxiliary is '{tgt}'
SPELL              which sign differs (matra length, nasal sign, nukta …)
PRON               singular -> plural pronoun: Hindi refers to one respected
                   person with the plural pronoun (उसने -> उन्होंने); plural ->
                   singular: the referent is one person or thing, not honoured;
                   possessive ending: उनका/उनकी/उनके agrees with the next noun,
                   like का/की/के
"""

from __future__ import annotations  # modern type-hint syntax

import difflib  # character-level diff for spelling
import re  # year pattern
from typing import Dict, FrozenSet, Optional, Sequence, Tuple  # type hints

from .error_generator import MONTHS  # month names
from .morphology import (COMPOUND_POSTP_HEADS, FORM_INDEX, FUNCTION_WORDS, GROUP_GLUE, POSTPOSITIONS,  # verb knowledge
                         find_verb_groups)
from .names import subject_span  # first unmarked noun phrase
from .other_patterns import _NUMBER_WORDS, PRONOUNS, pronoun_pattern  # number words; pronoun series / number / case

Cell = Tuple[str, str]
CELL_HI = {("M", "sg"): "पुल्लिंग एकवचन", ("F", "sg"): "स्त्रीलिंग एकवचन",
           ("M", "pl"): "पुल्लिंग बहुवचन", ("F", "pl"): "स्त्रीलिंग बहुवचन"}
CELL_EN = {("M", "sg"): "masculine singular", ("F", "sg"): "feminine singular",
           ("M", "pl"): "masculine plural", ("F", "pl"): "feminine plural"}
GENDER = {"M": ("पुल्लिंग", "masculine"), "F": ("स्त्रीलिंग", "feminine")}
NUMBER = {"sg": ("एकवचन", "singular"), "pl": ("बहुवचन", "plural")}
POSTP_MEANING = {"में": ("के अंदर", "in, inside"), "पर": ("के ऊपर / स्थान पर", "on, at"),
                 "से": ("से (स्रोत, साधन)", "from, with, by"), "को": ("को (लक्ष्य)", "to"),
                 "तक": ("तक (सीमा)", "up to, until")}
GENITIVE = ("का", "की", "के")
SUBJECT_PRONOUNS = frozenset("वे वह यह ये आप".split())
ERGATIVE_PRONOUNS = frozenset("उन्होंने इन्होंने जिन्होंने उसने इसने जिसने मैंने हमने आपने तुमने किसने".split())  # ने fused into the pronoun
#: Honorific pronouns: they always refer to people (वह / यह can mean "it", so they are not here).
HONORIFIC_PRONOUNS = frozenset("वे ये आप उन्होंने इन्होंने आपने".split())
#: Nouns and titles that always denote a person (closed list: a gender-agreement subject such as
#: परिवार may be any noun, but "X is one respected person" must be about a person).
PERSON_NOUNS = frozenset((
    "पिता पिताजी माता माताजी माँ मां पति पत्नी पुत्र पुत्री बेटा बेटी बेटे दादा दादी नाना नानी भाई बहन चाचा "
    "चाची मामा मामी गुरु गुरुजी स्वामी स्वामीजी महात्मा पंडित राजा रानी राजकुमार राजकुमारी सम्राट महाराज "
    "महाराजा अध्यक्ष सदस्य सांसद विधायक मंत्री मुख्यमंत्री प्रधानमंत्री राष्ट्रपति राज्यपाल न्यायाधीश "
    "प्रोफेसर प्राध्यापक शिक्षक शिक्षिका लेखक लेखिका कवि कवयित्री गायक गायिका अभिनेता अभिनेत्री निर्देशक "
    "वैज्ञानिक खिलाड़ी कप्तान संस्थापक नेता संत ऋषि मुनि आचार्य डॉक्टर डॉ डॉ. श्री श्रीमती सुश्री").split())
TITLES = frozenset("श्री श्रीमती सुश्री डॉ डॉ. डॉक्टर प्रो प्रो. स्वामी महात्मा पंडित".split())
#: Words that can never be a subject or object: time words, connectives, adverbs (NOT_SUBJECT).
NOT_SUBJECT = frozenset((
    "आज कल अब तब जब फिर बाद पहले पश्चात पश्चात् दौरान बीच समय वर्ष वर्षों साल सालों सन सन् ई ईस्वी दिन रात "
    "सुबह शाम महीना महीने सदी शताब्दी दशक वजह कारण अनुसार अलावा बावजूद शायद संप्रति सम्प्रति वर्तमान आजकल "
    "अभी लगभग केवल सिर्फ़ सिर्फ भी ही तथा एवं और साथ द्वारा लिए लिये तरह रूप प्रकार बहुत अधिक कई कुछ सभी "
    "इसलिए किंतु किन्तु परंतु परन्तु लेकिन क्योंकि यद्यपि तो अगला अगली अगले पिछला पिछली पिछले").split())
#: Hosts of conjunct verbs and predicate participles (प्राप्त करना, शुरू होना, सम्मानित किया): they sit
#: right before the verb but are part of the predicate, never its subject or object.
PREDICATE_WORDS = frozenset((
    "प्राप्त शुरू शुरु आरंभ आरम्भ प्रारंभ प्रारम्भ सम्मानित निर्वाचित नियुक्त मनोनीत जारी पूरा पूरी पूरे अलग "
    "शामिल उत्तीर्ण स्थापित प्रकाशित प्रसिद्ध लोकप्रिय सक्रिय उपलब्ध सफल असफल मौजूद समाप्त खत्म ख़त्म तैयार "
    "पैदा प्रभावित आयोजित घोषित पुरस्कृत सेवानिवृत्त कार्यरत पदस्थ विवाहित संबद्ध सम्बद्ध जुड़े जुड़ा जुड़ी "
    "छोड़ याद पसंद ग्रहण धारण प्रदान दर्ज जाना").split())
#: Nouns for a group: "नेहरू परिवार" is about the family, "सिंधी भाई" about many people, not one person.
GROUP_NOUNS = frozenset("परिवार वंश कुल समुदाय समाज लोग लोगों जाति टीम दल पार्टी संघ सरकार कंपनी भाई बहन".split())
_PLURAL_END = ("एँ", "एं", "ओं", "ों", "ियाँ", "ियां", "याँ", "यां")
ORDINALS = frozenset("प्रथम द्वितीय तृतीय चतुर्थ पहला पहली दूसरा दूसरी तीसरा तीसरी चौथा चौथी".split())
_NUMBER = re.compile(r"^[\d०-९.,]+$")
POSSESSIVES = frozenset("का की के उनका उनकी उनके इनका इनकी इनके उसका उसकी उसके इसका इसकी इसके "
                        "अपना अपनी अपने जिनका जिनकी जिनके".split())
SIGN = {  # character -> (Hindi name, English name)
    "ि": ("ह्रस्व इ की मात्रा (ि)", "the short i sign (ि)"), "ी": ("दीर्घ ई की मात्रा (ी)", "the long ī sign (ी)"),
    "ु": ("ह्रस्व उ की मात्रा (ु)", "the short u sign (ु)"), "ू": ("दीर्घ ऊ की मात्रा (ू)", "the long ū sign (ू)"),
    "े": ("ए की मात्रा (े)", "the e sign (े)"), "ै": ("ऐ की मात्रा (ै)", "the ai sign (ै)"),
    "ो": ("ओ की मात्रा (ो)", "the o sign (ो)"), "ौ": ("औ की मात्रा (ौ)", "the au sign (ौ)"),
    "ं": ("अनुस्वार (ं)", "the anusvara (ं)"), "ँ": ("चंद्रबिंदु (ँ)", "the chandrabindu (ँ)"),
    "़": ("नुक्ता (़)", "the nukta (़)"), "इ": ("इ", "इ"), "ई": ("ई", "ई"), "उ": ("उ", "उ"), "ऊ": ("ऊ", "ऊ"),
    "ए": ("ए", "ए"), "ऐ": ("ऐ", "ऐ"), "ओ": ("ओ", "ओ"), "औ": ("औ", "औ"),
}
_YEAR = re.compile(r"^(1[89]\d\d|20[0-2]\d)$")
PAST_WORDS = ("जन्म", "निधन", "मृत्यु", "देहांत", "देहान्त")
PRESENT_WORDS = ("वर्तमान", "आजकल", "अभी", "संप्रति", "सम्प्रति")


# ----------------------------------------------------------------------
# Small analyses
# ----------------------------------------------------------------------
def verb_cell(words: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    """(gender, number) of a verb group: number from the last inflecting word, gender
    from the last gendered one (है/हैं have none). None where unknown."""
    gender = number = None
    for w in reversed(list(words)):
        cells = [(p.kind, c) for p in FORM_INDEX.get(w, ()) for c, f in p.forms if f == w]
        if not cells:
            continue
        if number is None:
            number = cells[0][1][1]
        if gender is None and cells[0][0] != "aux_pres":
            genders = {c[0] for k, c in cells}
            gender = genders.pop() if len(genders) == 1 else None
        if gender is not None and number is not None:
            break
    return gender, number


def cell_text(gender: Optional[str], number: Optional[str]) -> Tuple[str, str]:  # ("पुल्लिंग एकवचन", "masculine singular")
    if gender and number:
        return CELL_HI[(gender, number)], CELL_EN[(gender, number)]
    if number:
        return NUMBER[number]
    return "", ""


def is_subject_word(w: str) -> bool:
    """Can ``w`` be (part of) a subject or object at all? Not a time word, connective, verb form,
    conjunct-verb host or predicate participle, compound-postposition head, month or number."""
    return not (w in NOT_SUBJECT or w in PREDICATE_WORDS or w in FORM_INDEX or w in COMPOUND_POSTP_HEADS
                or w in MONTHS or w in _NUMBER_WORDS or w in ORDINALS or w in FUNCTION_WORDS
                or w in POSTPOSITIONS or _NUMBER.match(w))


def is_person_word(w: str) -> bool:  # a noun or title that always denotes a person (पिता, अध्यक्ष, शास्त्रीजी)
    return w in PERSON_NOUNS or (len(w) > 2 and w.endswith("जी") and is_subject_word(w))


def is_name_word(w: str, names: FrozenSet[str] = frozenset()) -> bool:
    """A gazetteer name word that can stand for a person: not a short token (टी), not an ordinal or
    other non-subject word. Page titles also hold words such as गीत or उद्योग, so a name word is
    only trusted at the start of a sentence-initial subject (person_subject)."""
    return (w in names and len(w) > 2 and not w.endswith(_PLURAL_END)  # रचनाएँ: a plural noun, not a name
            and is_subject_word(w))


def person_subject(target: Sequence[str], start: int, names: FrozenSet[str] = frozenset()) -> str:
    """The respected person the verb at ``start`` is about, or "" if no rule finds a clear person.

    In order: (1) the first honorific pronoun before the verb that is not followed by a
    postposition (वे, ये, आप; उन्होंने with ने fused in); (2) a sentence-initial subject run:
    its leading gazetteer name words (with any title in front: डॉ संबित), or a leading person
    noun (राजा …); (3) a person noun or -जी word right after a possessive (उनके पिता).
    Never a time word, noun of reason or other non-person (आज, वजह, दौरान).
    """
    before = list(target[:start])
    for i, w in enumerate(before):
        nxt = before[i + 1] if i + 1 < len(before) else ""
        if w in HONORIFIC_PRONOUNS and (nxt not in POSTPOSITIONS or nxt == "ने"):
            return w
    a, b = subject_span(target, start)
    run = list(target[a:b])
    if not run:
        return ""
    if a == 0:  # a sentence-initial run: name words, after any title (डॉ संबित …)
        lo = 0
        while lo < len(run) and run[lo] in TITLES:
            lo += 1
        k = lo
        while k < len(run) and is_name_word(run[k], names):
            k += 1
        if k > lo and not (k < len(run) and (run[k] in GROUP_NOUNS or run[k] in PERSON_NOUNS)):
            return " ".join(run[:k])  # (नेहरू परिवार, सिंधी भाई: the name only modifies the subject noun)
        if lo < len(run) and is_person_word(run[lo]):  # राजा हरिश्चंद्र …, पिताजी …
            return " ".join(run[:lo + 1])
    if a > 0 and target[a - 1] in POSSESSIVES and is_person_word(run[0]):  # उनके पिता, उनकी माँ
        return run[0]
    return ""


def subject_words(target: Sequence[str], end: int, names: FrozenSet[str] = frozenset()) -> str:
    """The subject's words when a rule identifies them, else "".

    A run of content words can hold more than the subject (परिवार कोलकाता चला
    आया: subject + destination), so a run is only named when (a) it contains
    gazetteer names (the contiguous name words are the subject), (b) it follows
    a possessive (उनका परिवार: the possessed noun is the first word), or (c) it
    is a single word.
    """
    a, b = subject_span(target, end)
    run = list(target[a:b])
    if not run:
        return ""
    idx = [k for k, w in enumerate(run) if w in names]
    if idx:
        words = run[idx[0]: idx[-1] + 1]
    elif a > 0 and target[a - 1] in POSSESSIVES:
        words = run[:1]
    else:
        words = run[:1] if len(run) == 1 else []
    return " ".join(words) if words and all(is_subject_word(w) for w in words) else ""


def controller(target: Sequence[str], start: int, names: FrozenSet[str] = frozenset()) -> Tuple[str, str]:
    """(word(s), role) the verb at ``start`` agrees with; role "object" in a ने clause."""
    before = list(target[:start])
    if "ने" in before or any(w in ERGATIVE_PRONOUNS for w in before):  # ergative clause: the verb agrees with the object
        k = start - 1
        while k >= 0 and (target[k] in GROUP_GLUE or target[k] in PREDICATE_WORDS):  # skip light-verb parts (प्राप्त कर लिया)
            k -= 1
        if k >= 0 and target[k] != "ने" and is_subject_word(target[k]):
            return target[k], "object"
        return "", "object"
    if before and before[0] in SUBJECT_PRONOUNS:
        return before[0], "subject"
    return subject_words(target, start, names), "subject"


def main_verb(target: Sequence[str]) -> str:  # the last verb group, as text
    groups = find_verb_groups(target)
    return " ".join(target[groups[-1].start:groups[-1].end]) if groups else ""


def _q(w: str) -> str:  # quote a Hindi word
    return f"'{w}'"


# ----------------------------------------------------------------------
# Templates
# ----------------------------------------------------------------------
def _agreement(e: Dict, target: Sequence[str], feature: str, names: FrozenSet[str]) -> Tuple[str, str]:
    src, tgt = e["src"], e["tgt"]
    sg, sn = verb_cell(src.split())
    tg, tn = verb_cell(tgt.split())
    s_hi, s_en = cell_text(sg, sn)
    t_hi, t_en = cell_text(tg, tn)
    who, role = controller(target, e["tgt_span"][0], names)
    role_hi = "कर्म" if role == "object" else "कर्ता"
    who_hi = f"{role_hi} {_q(who)}" if who else role_hi
    who_en = f"the {role} {_q(who)}" if who else f"the {role}"
    if feature == "gender" and tg:
        val_hi, val_en = GENDER[tg]
    elif tn:
        val_hi, val_en = NUMBER[tn]
    else:
        val_hi, val_en = t_hi, t_en
    lead_hi = "ने वाले वाक्य में क्रिया कर्ता से नहीं, कर्म से मेल खाती है। " if role == "object" else ""
    lead_en = "With ने the verb agrees with the object, not the subject. " if role == "object" else ""
    hi = f"{lead_hi}{_q(src)} {s_hi} रूप है, पर {who_hi} {val_hi} है, इसलिए क्रिया {_q(tgt)} ({t_hi}) होगी।"
    en = f"{lead_en}{_q(src)} is the {s_en} form, but {who_en} is {val_en}, so the verb is {_q(tgt)} ({t_en})."
    return hi, en


def _honorific(e: Dict, target: Sequence[str], names: FrozenSet[str]) -> Tuple[str, str]:
    who = person_subject(target, e["tgt_span"][0], names)
    src, tgt = _q(e["src"]), _q(e["tgt"])
    if not who:  # no clear person: say only what the grammar guarantees
        return (f"क्रिया का वचन कर्ता से मेल खाना चाहिए (आदरणीय व्यक्ति के लिए बहुवचन क्रिया आती है), "
                f"इसलिए {src} की जगह {tgt} होगा।",
                f"The verb must match its subject in number (Hindi uses the plural verb for one "
                f"respected person), so {src} becomes {tgt}.")
    hi = (f"{_q(who)} एक आदरणीय व्यक्ति हैं। आदर के लिए हिंदी में बहुवचन क्रिया आती है, "
          f"इसलिए {src} की जगह {tgt} होगा।")
    en = (f"{_q(who)} is one respected person, and Hindi uses the plural verb for respect, "
          f"so {src} becomes {tgt}.")
    return hi, en


def _postposition(e: Dict, target: Sequence[str]) -> Tuple[str, str]:
    src, tgt = e["src"], e["tgt"]
    a, b = e["tgt_span"]
    if src in GENITIVE and tgt in GENITIVE:
        nxt = target[b] if b < len(target) else ""
        hi = (f"संबंध-सूचक का/की/के अगले शब्द {_q(nxt)} के लिंग-वचन से मेल खाता है, इसलिए {_q(src)} की जगह "
              f"{_q(tgt)} (का: पुल्लिंग एकवचन, की: स्त्रीलिंग, के: पुल्लिंग बहुवचन या परसर्ग से पहले)।")
        en = (f"The possessive का/की/के agrees with the word that follows ({_q(nxt)}), so {_q(src)} becomes "
              f"{_q(tgt)} (का masculine singular, की feminine, के masculine plural or before a postposition).")
        return hi, en
    noun = target[a - 1] if a > 0 else ""
    if not src:  # inserted
        m_hi, m_en = POSTP_MEANING.get(tgt, ("", ""))
        return (f"{_q(noun)} के बाद परसर्ग {_q(tgt)} ({m_hi}) छूट गया था।",
                f"The postposition {_q(tgt)} ({m_en}) was missing after {_q(noun)}.")
    if not tgt:  # deleted
        return (f"यहाँ परसर्ग {_q(src)} की ज़रूरत नहीं है।", f"The postposition {_q(src)} is not needed here.")
    m_hi, m_en = POSTP_MEANING.get(tgt, ("", ""))
    s_hi, s_en = POSTP_MEANING.get(src, ("", ""))
    hi = f"{_q(noun + ' ' + tgt)} सही है: यहाँ {_q(tgt)} ({m_hi}) का अर्थ चाहिए, {_q(src)} ({s_hi}) का नहीं।"
    en = f"{_q(noun + ' ' + tgt)} is right: the meaning needed is {_q(tgt)} ({m_en}), not {_q(src)} ({s_en})."
    return hi, en


def _ergative(e: Dict, target: Sequence[str]) -> Tuple[str, str]:
    verb = main_verb(target)
    v = f" {_q(verb)}" if verb else ""
    if e["tgt"] and not e["src"]:  # ने inserted
        return (f"भूतकाल (पूर्ण) की सकर्मक क्रिया{v} के साथ कर्ता के बाद 'ने' आता है।",
                f"With a past (perfective) transitive verb{v}, the subject takes ने.")
    if e["src"] and not e["tgt"]:  # ने deleted
        return (f"यह क्रिया{v} 'ने' नहीं लेती (अकर्मक या अपूर्ण), इसलिए 'ने' हटेगा।",
                f"This verb{v} does not take ने (intransitive or not perfective), so ने is removed.")
    return (f"भूतकाल (पूर्ण) की सकर्मक क्रिया{v} के साथ कर्ता ने-रूप लेता है: {_q(e['src'])} + ने = {_q(e['tgt'])}।",
            f"With a past (perfective) transitive verb{v}, the subject takes ने: {_q(e['src'])} + ने = {_q(e['tgt'])}.")


def _auxiliary(e: Dict, target: Sequence[str]) -> Tuple[str, str]:
    past = next((t for t in target if _YEAR.match(t) or t in PAST_WORDS), "")
    now = next((t for t in target if t in PRESENT_WORDS), "")
    tense = "past" if any(f in e["tgt"].split() for f in ("था", "थी", "थे", "थीं")) else "present"
    if tense == "past":
        why_hi = f"वाक्य भूतकाल की बात करता है{' (' + _q(past) + ')' if past else ''}"
        why_en = f"The sentence is about the past{' (' + _q(past) + ')' if past else ''}"
        hi, en = "भूतकाल", "past"
    else:
        why_hi = f"वाक्य वर्तमान की बात करता है{' (' + _q(now) + ')' if now else ''}"
        why_en = f"The sentence is about the present{' (' + _q(now) + ')' if now else ''}"
        hi, en = "वर्तमान", "present"
    return (f"{why_hi}, इसलिए सहायक क्रिया {_q(e['src'])} की जगह {_q(e['tgt'])} ({hi}) होगी।",
            f"{why_en}, so the auxiliary {_q(e['src'])} becomes {_q(e['tgt'])} ({en} tense).")


def _spelling(e: Dict) -> Tuple[str, str]:
    src, tgt = e["src"], e["tgt"]
    gs, ns = verb_cell(src.split())
    gt, nt = verb_cell(tgt.split())
    if gs and (gs, ns) == (gt, nt):  # same verb cell, other spelling family (गये / गए)
        return (f"{_q(src)} और {_q(tgt)} एक ही रूप की दो वर्तनियाँ हैं; मानक वर्तनी {_q(tgt)} है।",
                f"{_q(src)} and {_q(tgt)} are two spellings of the same form; the standard one is {_q(tgt)}.")
    ops = [op for op in difflib.SequenceMatcher(a=src, b=tgt, autojunk=False).get_opcodes() if op[0] != "equal"]
    if ops:
        op, i1, i2, j1, j2 = ops[0]
        a, b = src[i1:i2], tgt[j1:j2]
        a_hi, a_en = SIGN.get(a, (a, a)) if a else ("", "")
        b_hi, b_en = SIGN.get(b, (b, b)) if b else ("", "")
        if a and b:
            return (f"मानक वर्तनी {_q(tgt)} है: {_q(src)} में {a_hi} की जगह {b_hi} चाहिए।",
                    f"{_q(src)} has {a_en} where the standard spelling {_q(tgt)} has {b_en}.")
        if b:
            return (f"मानक वर्तनी {_q(tgt)} है: {_q(src)} में {b_hi} भी चाहिए।",
                    f"{_q(src)} is missing {b_en}: the standard spelling is {_q(tgt)}.")
        return (f"मानक वर्तनी {_q(tgt)} है: {_q(src)} में {a_hi} नहीं चाहिए।",
                f"{_q(src)} should not have {a_en}: the standard spelling is {_q(tgt)}.")
    return f"मानक वर्तनी {_q(tgt)} है।", f"The standard spelling is {_q(tgt)}."


def _pronoun(e: Dict, target: Sequence[str]) -> Tuple[str, str]:
    src, tgt = e["src"], e["tgt"]
    if pronoun_pattern(src, tgt) == "POSS-AGR":
        b = e["tgt_span"][1]
        nxt = target[b] if b < len(target) else ""
        return (f"संबंध-सूचक सर्वनाम का/की/के की तरह अगले शब्द {_q(nxt)} से लिंग-वचन में मेल खाता है, इसलिए "
                f"{_q(src)} की जगह {_q(tgt)}।",
                f"Like का/की/के, a possessive pronoun agrees with the word that follows ({_q(nxt)}), "
                f"so {_q(src)} becomes {_q(tgt)}.")
    if PRONOUNS[tgt][1] == "pl":  # singular -> honorific plural
        return (f"आदरणीय व्यक्ति के लिए एक व्यक्ति होने पर भी बहुवचन सर्वनाम आता है, इसलिए {_q(src)} की जगह {_q(tgt)}।",
                f"Hindi refers to one respected person with the plural pronoun, so {_q(src)} becomes {_q(tgt)}.")
    return (f"यहाँ सर्वनाम एक व्यक्ति या वस्तु के लिए है और आदरसूचक नहीं है, इसलिए {_q(src)} की जगह एकवचन {_q(tgt)}।",
            f"The pronoun refers to one person or thing that is not spoken of with respect, "
            f"so {_q(src)} becomes the singular {_q(tgt)}.")


def explain(edit: Dict, target_tokens: Sequence[str], names: FrozenSet[str] = frozenset()) -> Dict[str, str]:
    """{"hi": ..., "en": ...} for one edit (dict with type, src, tgt, tgt_span); ``names``: gazetteer."""
    t = edit["type"]
    if t == "AGR-GEN":
        hi, en = _agreement(edit, target_tokens, "gender", names)
    elif t == "AGR-NUM":
        hi, en = _agreement(edit, target_tokens, "number", names)
    elif t == "AGR-HON":
        hi, en = _honorific(edit, target_tokens, names)
    elif t == "POSTP":
        hi, en = _postposition(edit, target_tokens)
    elif t == "ERG":
        hi, en = _ergative(edit, target_tokens)
    elif t == "AUX":
        hi, en = _auxiliary(edit, target_tokens)
    elif t == "SPELL":
        hi, en = _spelling(edit)
    elif t == "PRON":
        hi, en = _pronoun(edit, target_tokens)
    else:
        hi, en = ("यह बदलाव आठ समर्थित प्रकारों में नहीं आता।", "This change is outside the eight supported types.")
    return {"hi": hi, "en": en}
