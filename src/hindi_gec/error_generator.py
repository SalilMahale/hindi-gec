# -*- coding: utf-8 -*-
"""Synthetic error generator (PLAN §3.2, rule-based, no Stanza).

Why synthetic errors
--------------------
Real annotated Hindi errors in the biography domain number about a thousand —
enough to *test* on, far too few to train a seq2seq model. So we take clean
sentences and break them in the ways people actually break Hindi, keeping the
clean sentence as the target. Because every error is created by one
reversible token edit, its span and its type are known exactly: the label is
correct *by construction*, not by a classifier's guess.

What each type does (on the CLEAN tokens; the corrupted side is the source)
--------------------------------------------------------------------------
AGR-GEN  re-inflect a whole verb group to the other gender   हुआ था -> हुई थी
AGR-NUM  sg -> pl anywhere, or pl -> sg outside honorific context  था -> थे
AGR-HON  pl -> sg in an honorific context (वे / उन्होंने / page subject)  थे -> था
AUX      past aux -> present in a past-anchored sentence (year, जन्म, निधन),
         present -> past with वर्तमान/आजकल                    हुआ था -> हुआ है
POSTP    swap among में/से/को/पर/तक with confusion weights (config);
         never right after a number or month (1931 से / 1931 तक are both fine)
ERG      उन्होंने -> वे (drop ने), delete a standalone ने, or वे -> उन्होंने
GENITIVE (typed POSTP) का/की/के swap: का->की, की->का/के, के->की always; के->का
         only before a compound-postposition head (के बाद -> का बाद). का->के is
         never generated: "राम के पुत्र" (Ram's sons) is correct Hindi.
SPELL    one matra / vowel / nasal / nukta confusion in a content word, vetoed if
         the "misspelling" is itself a common corpus form (गाँव -> गांव is a
         legitimate variant, not an error)
PRON     an honorific pronoun made singular where the verb does not agree
         with it (ergative, genitive, dative: उन्होंने -> उसने, उनके -> उसके,
         उन्हें -> उसे), or a possessive pronoun's ending swapped by the का/की/के
         rules (उनका -> उनकी, अपने -> अपनी). Real dev edits go 88 % singular ->
         honorific, so only that direction is generated.
SPELL+   data-driven forced pairs (config generator.forced_spell):
         उन्होंने -> उन्होने is 76 of 145 real dev SPELL edits. The corpus itself
         contains उन्होने 402 times, so the generator first normalises those targets and
         emits each original sentence as a real (wrong -> right) pair.
         With ``forced_spell_on_top`` these pairs sit OUTSIDE the type quota:
         ordinary misspellings keep their own SPELL share (0.1785) and
         ``forced_spell_pairs`` forced pairs (natural ones first, then
         one-edit synthetic ones) are added on top. In an earlier training run
         they were drawn inside the SPELL share, which left fewer ordinary
         misspellings and SPELL F0.5 fell from 0.762 to 0.589.

Sampling
--------
A sentence is NONE (left clean) with p = ``p_clean``; otherwise it gets 1 or 2
errors. For each error slot the type is chosen *greedily by deficit*: of the
types applicable to this sentence, the one furthest below its target share so
far. This keeps the final distribution within a fraction of a point of PLAN
§3.3 even though some types (ERG) are applicable far less often than others.
Two errors in one sentence never touch the same or adjacent tokens, so a diff
of (source, target) recovers them as two separate edits.
"""

from __future__ import annotations  # modern type-hint syntax

import random  # random choices (always through a seeded rng, so results are reproducible)
import re  # regular expressions
from collections import Counter  # counting dictionary
from dataclasses import dataclass  # simple record classes
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple  # type hints

from . import ERROR_TYPES, EXTENDED_TYPES  # the 7 error types (+ PRON)
from .morphology import (  # our rule-based Hindi grammar helpers
    COMPOUND_POSTP_HEADS,  # words like बाद/लिए that follow के in compound postpositions
    analyses,  # which verb paradigms a word belongs to
    find_verb_groups,  # find verb groups (main verb + helpers) in a sentence
    has_past_anchor,  # does the sentence clearly talk about the past?
    has_present_anchor,  # does it clearly talk about the present?
    is_content_word,  # is a word a "real" noun/adjective (safe to misspell)?
    is_honorific_context,  # is the subject a respected person (plural verb)?
    render_group,  # re-inflect a verb group to a new gender/number
    spelling_variants,  # list of plausible misspellings of a word
)
from .schema import Edit, Pair  # record classes for edits and sentence pairs

_HAS_DIGIT = re.compile(r"\d")  # pattern: contains any digit
MONTHS = frozenset("""
जनवरी फ़रवरी फरवरी मार्च अप्रैल अप्रेल मई जून जुलाई अगस्त सितंबर सितम्बर अक्टूबर अक्तूबर
नवंबर नवम्बर दिसंबर दिसम्बर
""".split())  # Hindi month names (with spelling variants)

#: Ergative pronoun -> nominative (drop ने).
ERG_PRONOUNS: Dict[str, str] = {  # pronoun+ने form -> plain form (used to create a missing-ने error)
    "उन्होंने": "वे", "इन्होंने": "ये", "जिन्होंने": "जो",
    "उसने": "वह", "इसने": "यह", "मैंने": "मैं", "हमने": "हम",
}

#: PRON: honorific (plural) pronoun -> singular, only in cases the verb does
#: not agree with (ergative, genitive, dative/oblique), so no other word changes.
HON_TO_SG: Dict[str, str] = {
    "उन्होंने": "उसने", "उनका": "उसका", "उनकी": "उसकी", "उनके": "उसके", "उन्हें": "उसे", "उनको": "उसको",
    "उनसे": "उससे", "इन्होंने": "इसने", "इनका": "इसका", "इनकी": "इसकी", "इनके": "इसके", "इन्हें": "इसे",
}
#: Next words that make a plural pronoun a real plural (उन सभी, उनके दोनों …): never swapped.
PLURAL_MARKERS = frozenset("लोग लोगों सभी सब सबको दोनों तीनों चारों अनेक कई".split())
#: Possessive pronouns: form -> (stem, genitive ending). अपना/अपनी/अपने behave like का/की/के.
POSSESSIVE_FORMS: Dict[str, Tuple[str, str]] = {
    **{f"{s}{e}": (s, e) for s in ("उन", "उस", "इन", "इस", "जिन", "जिस") for e in ("का", "की", "के")},
    "अपना": ("अप", "का"), "अपनी": ("अप", "की"), "अपने": ("अप", "के"),
}
_POSS_RENDER = {("अप", "का"): "अपना", ("अप", "की"): "अपनी", ("अप", "के"): "अपने"}


@dataclass(frozen=True)  # immutable record
class Corruption:
    """Replace clean tokens [start, end) with ``new`` (may be empty = deletion)."""

    start: int  # first clean-token index to replace
    end: int  # one past the last index (half-open)
    new: Tuple[str, ...]  # the wrong replacement words
    type: str  # which of the 7 error types this is


# ----------------------------------------------------------------------
# Candidate enumeration
# ----------------------------------------------------------------------
def _diff_span(old: Sequence[str], new: Sequence[str], offset: int, typ: str) -> Optional[Corruption]:
    """Minimal [first changed, last changed] span between equal-length lists."""
    changed = [k for k, (a, b) in enumerate(zip(old, new)) if a != b]  # positions where the word differs
    if not changed:  # nothing changed
        return None
    a, b = changed[0], changed[-1] + 1  # smallest span covering all changes
    return Corruption(offset + a, offset + b, tuple(new[a:b]), typ)  # convert to sentence positions


def agreement_candidates(tokens: Sequence[str], page_title: str) -> List[Corruption]:
    """AGR-GEN, AGR-NUM, AGR-HON and AUX corruptions of every verb group."""
    out: List[Corruption] = []  # all possible corruptions
    hon = is_honorific_context(tokens, page_title)  # respected subject?
    past, present = has_past_anchor(tokens), has_present_anchor(tokens)  # tense clues
    for g in find_verb_groups(tokens):  # each verb group
        orig = list(tokens[g.start:g.end])  # its original words
        gender = g.gender or "M"  # default to masculine if no gender marking
        if g.gender is not None:  # group has gender -> can make a gender error
            flip = "F" if g.gender == "M" else "M"  # opposite gender
            c = _diff_span(orig, render_group(tokens, g, flip, g.number), g.start, "AGR-GEN")  # e.g. हुआ था -> हुई थी
            if c:
                out.append(c)
        other_n = "pl" if g.number == "sg" else "sg"  # opposite number
        typ = "AGR-NUM" if g.number == "sg" or not hon else "AGR-HON"  # plural->singular for a respected person = honorific error
        # A bare singular copula after an honorific subject (… पुत्र है) is often
        # itself the corpus's error; "corrupting" it to हैं would train the model
        # to turn correct honorific plurals into singulars. Skip that flip.
        bare_copula = len(g.slots) == 1 and g.slots[0][1].kind in ("aux_past", "aux_pres")  # group is just है/था
        if not (g.number == "sg" and hon and bare_copula):  # skip the risky case above
            c = _diff_span(orig, render_group(tokens, g, gender, other_n), g.start, typ)  # flip singular <-> plural
            if c:
                out.append(c)
        # AUX: swap the tense of the auxiliary, keeping gender and number.
        for k, p in g.slots:  # each inflecting word of the group
            if p.kind == "aux_past" and past:  # था/थे in a clearly-past sentence
                new = "हैं" if g.number == "pl" else "है"  # make it present (wrong)
            elif p.kind == "aux_pres" and present and g.gender is not None:  # है/हैं in a clearly-present sentence
                new = {("M", "sg"): "था", ("F", "sg"): "थी", ("M", "pl"): "थे",
                       ("F", "pl"): "थीं"}[(g.gender, g.number)]  # make it past (wrong), matching gender/number
            else:
                continue  # no safe tense error here
            if new != tokens[k]:
                out.append(Corruption(k, k + 1, (new,), "AUX"))  # single-word AUX error
    return out


def postposition_candidates(
    tokens: Sequence[str], confusion: Mapping[str, Mapping[str, float]], rng: random.Random
) -> List[Corruption]:
    """POSTP: one weighted substitute per eligible postposition."""
    out: List[Corruption] = []
    for i, t in enumerate(tokens):
        if t not in confusion or i == 0:  # not a swappable postposition, or first word
            continue
        prev = tokens[i - 1]  # word before the postposition
        if _HAS_DIGIT.search(prev) or prev in MONTHS:
            continue                       # 1931 से / 1931 तक / 1931 में: often all valid
        if t == "पर" and prev in (",", "।", "और"):
            continue                       # पर = "but"
        alts = list(confusion[t].items())  # possible wrong replacements with their weights
        alt = rng.choices([a for a, _ in alts], weights=[w for _, w in alts])[0]  # pick one, more likely ones more often
        out.append(Corruption(i, i + 1, (alt,), "POSTP"))  # e.g. में -> पर
    return out


def genitive_candidates(
    tokens: Sequence[str], confusion: Mapping[str, Mapping[str, float]], rng: random.Random
) -> List[Corruption]:
    """POSTP: genitive का/की/के swaps that are always wrong (see module docstring)."""
    out: List[Corruption] = []
    for i, t in enumerate(tokens):
        if t not in confusion or i == 0 or tokens[i - 1] in (",", "(", "।"):  # not का/की/के, or bad position
            continue
        if t == "की" and analyses(tokens, i):
            continue                       # verbal की (प्राप्त की), not the genitive
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""  # next word
        alts = [(a, w) for a, w in confusion[t].items()  # allowed replacements...
                if not (t == "के" and a == "का" and nxt not in COMPOUND_POSTP_HEADS)]  # ...but के->का only before बाद/लिए etc.
        if not alts:
            continue
        alt = rng.choices([a for a, _ in alts], weights=[w for _, w in alts])[0]  # weighted random choice
        out.append(Corruption(i, i + 1, (alt,), "POSTP"))
    return out


def ergative_candidates(tokens: Sequence[str]) -> List[Corruption]:
    """ERG: drop ने from a pronoun, delete a standalone ने, or insert one on वे."""
    out: List[Corruption] = []
    has_erg = any(t == "ने" or t in ERG_PRONOUNS for t in tokens)  # does the sentence already use ने?
    for i, t in enumerate(tokens):
        if t in ERG_PRONOUNS:  # उन्होंने -> वे (remove ने)
            out.append(Corruption(i, i + 1, (ERG_PRONOUNS[t],), "ERG"))
        elif t == "ने" and i > 0 and tokens[i - 1] not in (",", "(", ")", "।"):  # standalone ने after a noun
            out.append(Corruption(i, i + 1, (), "ERG"))  # delete it (empty replacement)
        elif (t == "वे" and not has_erg  # वे in a sentence without ने
              and (i == 0 or tokens[i - 1] in (",", "और", "तथा"))  # at the start of a clause
              and (i + 1 < len(tokens) and tokens[i + 1] not in ("लोग", "सभी", "सब", "दोनों"))):  # not "वे लोग" (those people)
            out.append(Corruption(i, i + 1, ("उन्होंने",), "ERG"))  # wrongly add ने
    return out


def spelling_candidates(
    tokens: Sequence[str], page_title: str, vocab: Mapping[str, int],
    veto_min_count: int, rng: random.Random,
    min_word_count: int = 0, veto_ratio: Optional[float] = None,
) -> List[Corruption]:
    """SPELL: one confusion per content word, never on the page subject's name.

    Without ``veto_ratio`` a variant seen >= ``veto_min_count`` times is vetoed. With it
    (the setting in use), it is vetoed only if it is also >= veto_ratio × as frequent as the word: a
    3-in-1232 "शीक्षा" is a typo in the corpus, not a legitimate spelling.
    Words seen < ``min_word_count`` times are skipped — names and rare words
    whose spelling the model cannot know (recall on them without this rule: 4 of 93).
    """
    out: List[Corruption] = []
    title_words = set(page_title.split())  # words of the person's name (never misspell names)
    for i, t in enumerate(tokens):
        if t in title_words or not is_content_word(t) or vocab.get(t, 0) < min_word_count:  # skip names, function words, rare words
            continue
        if veto_ratio is None:  # count veto only
            variants = [v for v in spelling_variants(t) if vocab.get(v, 0) < veto_min_count]  # drop variants that are common in the corpus
        else:  # count + ratio veto (in use)
            variants = [v for v in spelling_variants(t)
                        if vocab.get(v, 0) < veto_min_count or vocab.get(v, 0) < veto_ratio * vocab.get(t, 0)]  # keep if rare overall OR rare relative to the word
        if variants:
            out.append(Corruption(i, i + 1, (rng.choice(variants),), "SPELL"))  # choose one misspelling at random
    return out


def pronoun_candidates(
    tokens: Sequence[str], confusion: Mapping[str, Mapping[str, float]], rng: random.Random
) -> List[Corruption]:
    """PRON: honorific pronoun -> singular, and possessive-ending swaps (see module docstring)."""
    out: List[Corruption] = []
    for i, t in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        cands: List[str] = []
        if t in HON_TO_SG and nxt not in PLURAL_MARKERS:  # उनके दोनों पुत्र is a real plural
            cands.append(HON_TO_SG[t])
        if t in POSSESSIVE_FORMS:  # unlike a bare का/की/के, a possessive pronoun carries its own possessor (उनका जन्म …)
            stem, end = POSSESSIVE_FORMS[t]
            alts = [(a, w) for a, w in confusion.get(end, {}).items()  # same rules as का/की/के
                    if not (end == "के" and a == "का" and nxt not in COMPOUND_POSTP_HEADS)]
            if alts:
                a = rng.choices([x for x, _ in alts], weights=[w for _, w in alts])[0]
                cands.append(_POSS_RENDER.get((stem, a), stem + a))
        if cands:
            out.append(Corruption(i, i + 1, (rng.choice(cands),), "PRON"))
    return out


def forced_spell_candidates(tokens: Sequence[str], forced: Mapping[str, str]) -> List[Corruption]:
    """Data-driven SPELL pairs: right -> wrong (उन्होंने -> उन्होने), seen in real edits."""
    return [Corruption(i, i + 1, (forced[t],), "SPELL") for i, t in enumerate(tokens) if t in forced]


def normalise_forced(tokens: Sequence[str], forced: Mapping[str, str]) -> Tuple[List[str], List[Corruption]]:
    """Fix corpus tokens that are a forced pair's wrong form (उन्होने -> उन्होंने).

    Returns (normalised tokens, corruptions that turn the normalised sentence back
    into the original one): the original sentence is a real error pair.
    """
    right_of = {w: r for r, w in forced.items()}
    norm = [right_of.get(t, t) for t in tokens]
    corrs = [Corruption(i, i + 1, (t,), "SPELL") for i, t in enumerate(tokens) if t in right_of]
    return norm, corrs


def all_candidates(
    tokens: Sequence[str], page_title: str, vocab: Mapping[str, int],
    gcfg: Mapping[str, Any], rng: random.Random,
) -> Dict[str, List[Corruption]]:
    """Every applicable corruption of a clean sentence, grouped by type."""
    types = EXTENDED_TYPES if gcfg.get("pron_swaps") else ERROR_TYPES  # with pron_swaps: + PRON
    by_type: Dict[str, List[Corruption]] = {t: [] for t in types}  # one empty list per type
    genitive = (genitive_candidates(tokens, gcfg["genitive_confusion"], rng)
                if gcfg.get("genitive_swaps") else [])  # genitive swaps (config generator.genitive_swaps)
    for c in (agreement_candidates(tokens, page_title)  # collect candidates from every generator
              + postposition_candidates(tokens, gcfg["postp_confusion"], rng)
              + genitive
              + ergative_candidates(tokens)
              + spelling_candidates(tokens, page_title, vocab, gcfg["spell_veto_min_count"], rng,
                                    gcfg.get("spell_min_word_count", 0), gcfg.get("spell_veto_ratio"))):
        by_type[c.type].append(c)  # file each candidate under its type
    if gcfg.get("pron_swaps"):  # only when enabled (no extra rng calls otherwise: older data reproduces byte for byte)
        by_type["PRON"] = pronoun_candidates(tokens, gcfg["genitive_confusion"], rng)
    if gcfg.get("forced_spell") and not gcfg.get("forced_spell_on_top"):  # earlier run: forced pairs inside the SPELL quota
        forced = forced_spell_candidates(tokens, gcfg["forced_spell"])
        if forced:
            prefer = rng.random() < gcfg["forced_spell_prefer"]
            by_type["SPELL"] = forced if prefer else by_type["SPELL"] + forced
    return by_type


# ----------------------------------------------------------------------
# Applying corruptions
# ----------------------------------------------------------------------
def overlaps(a: Corruption, b: Corruption) -> bool:
    """True if two corruptions touch the same or adjacent clean tokens."""
    return not (a.end < b.start or b.end < a.start)  # they're separate only if there's a gap of at least one word


def apply_corruptions(
    clean: Sequence[str], corrs: Iterable[Corruption]
) -> Tuple[List[str], List[Edit]]:
    """Corrupt ``clean``; return (source tokens, edits with half-open spans).

    ``src_span`` indexes the corrupted source, ``tgt_span`` the clean target.
    """
    src: List[str] = []  # the corrupted sentence being built
    edits: List[Edit] = []  # matching edit records
    pos = 0  # how far we've copied from the clean sentence
    for c in sorted(corrs, key=lambda c: c.start):  # left to right
        src.extend(clean[pos:c.start])  # copy unchanged words up to the corruption
        s0 = len(src)  # where the wrong words start in the source
        src.extend(c.new)  # insert the wrong words
        edits.append(Edit(
            src_span=[s0, len(src)], tgt_span=[c.start, c.end],  # spans in the wrong and in the correct sentence
            src=" ".join(c.new), tgt=" ".join(clean[c.start:c.end]), type=c.type,  # wrong text, correct text, type
        ))
        pos = c.end  # continue after the replaced words
    src.extend(clean[pos:])  # copy the rest of the sentence
    return src, edits


def undo_edits(source: Sequence[str], edits: Sequence[Edit]) -> List[str]:
    """Rebuild the target from the source and edits (used by the quality gate)."""
    out = list(source)
    for e in sorted(edits, key=lambda e: e.src_span[0], reverse=True):  # right to left so earlier positions don't shift
        s0, s1 = e.src_span
        out[s0:s1] = e.tgt.split() if e.tgt else []  # put the correct words back
    return out


# ----------------------------------------------------------------------
# Sampling with a type quota
# ----------------------------------------------------------------------
class TypeQuota:
    """Greedy-by-deficit type chooser that tracks PLAN §3.3 shares."""

    def __init__(self, targets: Mapping[str, float]) -> None:
        self.targets = dict(targets)  # desired share for each type (e.g. AGR-NUM 0.231)
        self.counts: Counter = Counter()  # how many of each type generated so far

    def choose(self, available: Iterable[str], rng: random.Random) -> str:  # pick the type most "behind" its target
        avail = list(available)  # types possible for this sentence
        total = sum(self.counts.values()) + 1  # total errors after adding one more
        deficit = {t: self.targets[t] * total - self.counts[t] for t in avail}  # how many we "should" have minus how many we have
        best = max(deficit.values())  # the largest shortfall
        return rng.choice([t for t in avail if deficit[t] >= best - 1e-9])  # random among the most-behind types (ties)

    def add(self, types: Iterable[str]) -> None:
        self.counts.update(types)  # record the types just generated


def sample_corruptions(
    cands: Mapping[str, List[Corruption]], k: int, quota: TypeQuota, rng: random.Random
) -> List[Corruption]:
    """Pick up to ``k`` non-overlapping corruptions, types chosen by quota."""
    chosen: List[Corruption] = []
    for _ in range(k):  # k = 1 or 2 errors
        avail = {t: [c for c in cs if not any(overlaps(c, x) for x in chosen)]  # drop candidates that clash with already-chosen ones
                 for t, cs in cands.items()}
        avail = {t: cs for t, cs in avail.items() if cs}  # keep only types that still have candidates
        if not avail:  # nothing left
            break
        t = quota.choose(avail, rng)  # choose the type
        chosen.append(rng.choice(avail[t]))  # choose one candidate of that type
    return chosen


def generate_split(
    sentences: Sequence[Mapping[str, Any]], n_pairs: int, split: str,
    vocab: Mapping[str, int], gcfg: Mapping[str, Any], rng: random.Random,
    seen_sources: Set[str], max_passes: int = 12,
    natural: Sequence[Tuple[Mapping[str, Any], Sequence[Corruption]]] = (),
    n_forced: int = 0,
) -> Tuple[List[Pair], Counter]:
    """Generate ``n_pairs`` pairs for one split; returns (pairs, rejection counts).

    ``natural``: (normalised sentence, corruptions) whose source is the
    corpus's own original text. Without ``forced_spell_on_top`` they are emitted
    first and counted in the type quota. With it, they and ``n_forced`` forced
    pairs in all (natural first, then one forced edit per sentence) are added
    after the ``n_pairs`` quota pairs, outside the quota.
    """
    on_top = bool(gcfg.get("forced_spell_on_top"))
    rejects: Counter = Counter()  # reasons sentences were rejected
    quota = TypeQuota(gcfg["type_distribution"])  # keeps the type mix on target
    k_values = [int(k) for k in gcfg["errors_per_sentence"]]  # [1, 2]
    k_weights = [float(w) for w in gcfg["errors_per_sentence"].values()]  # [0.75, 0.25]
    n_none = min(round(gcfg["p_clean"] * n_pairs), len(sentences))  # number of error-free pairs (20 %)
    n_err = n_pairs - n_none  # number of pairs that get errors
    pairs: List[Pair] = []

    def make(s: Mapping[str, Any], src: List[str], edits: List[Edit]) -> Pair:  # helper: build a Pair record
        return Pair(id="", origin="synthetic", page_title=s["page_title"],
                    source=" ".join(src), target=" ".join(s["tokens"]),  # source = corrupted, target = clean
                    source_tokens=list(src), target_tokens=list(s["tokens"]),
                    edits=edits, split=split)

    for s in rng.sample(list(sentences), n_none):  # pick random sentences to keep clean
        text = " ".join(s["tokens"])
        if text in seen_sources:  # already used
            rejects["duplicate"] += 1
            continue
        seen_sources.add(text)
        pairs.append(make(s, list(s["tokens"]), []))  # source == target, no edits

    made = 0  # error pairs created so far
    for s, corrs in (() if on_top else natural):  # the corpus's own misspellings, as real error pairs
        src, edits = apply_corruptions(s["tokens"], corrs)
        source = " ".join(src)
        if made >= n_err or source in seen_sources:
            continue
        seen_sources.add(source)
        quota.add(e.type for e in edits)
        pairs.append(make(s, src, edits))
        made += 1
    for _ in range(max_passes):  # may loop over the corpus several times to reach the target count
        order = list(sentences)
        rng.shuffle(order)  # random order each pass
        for s in order:
            if made >= n_err:  # enough
                break
            cands = all_candidates(s["tokens"], s["page_title"], vocab, gcfg, rng)  # all possible errors
            if not any(cands.values()):  # no error type applies
                rejects["no applicable error"] += 1
                continue
            k = rng.choices(k_values, weights=k_weights)[0]  # 1 error (75 %) or 2 errors (25 %)
            chosen = sample_corruptions(cands, k, quota, rng)  # pick them
            src, edits = apply_corruptions(s["tokens"], chosen)  # build the wrong sentence
            source = " ".join(src)
            if source == " ".join(s["tokens"]):  # corruption changed nothing
                rejects["no-op"] += 1
                continue
            if source in seen_sources:  # same wrong sentence already produced
                rejects["duplicate"] += 1
                continue
            seen_sources.add(source)
            quota.add(e.type for e in edits)  # update type counts
            pairs.append(make(s, src, edits))  # save the pair
            made += 1
        if made >= n_err:
            break
    if made < n_err:  # couldn't make enough
        rejects[f"shortfall ({split})"] = n_err - made
    if on_top:  # forced pairs on top of the quota pairs
        forced = 0
        for s, corrs in natural:
            src, edits = apply_corruptions(s["tokens"], corrs)
            source = " ".join(src)
            if forced >= n_forced or source in seen_sources:
                continue
            seen_sources.add(source)
            pairs.append(make(s, src, edits))
            forced += 1
        order = [s for s in sentences if any(t in gcfg["forced_spell"] for t in s["tokens"])]
        rng.shuffle(order)
        for s in order:
            if forced >= n_forced:
                break
            src, edits = apply_corruptions(s["tokens"], [rng.choice(forced_spell_candidates(s["tokens"],
                                                                                           gcfg["forced_spell"]))])
            source = " ".join(src)
            if source in seen_sources:
                rejects["duplicate"] += 1
                continue
            seen_sources.add(source)
            pairs.append(make(s, src, edits))
            forced += 1
        if forced < n_forced:
            rejects[f"forced shortfall ({split})"] = n_forced - forced
    return pairs, rejects


def split_by_page(
    sentences: Sequence[Mapping[str, Any]], val_share: float, seed: int
) -> Dict[str, List[Mapping[str, Any]]]:
    """Assign whole biographies to train/val so no page leaks across (PLAN §2.4)."""
    pages = sorted({s["page_title"] for s in sentences})  # unique page titles, sorted for reproducibility
    random.Random(seed).shuffle(pages)  # shuffle with a fixed seed
    val_pages = set(pages[: round(val_share * len(pages))])  # first 5 % of pages go to validation
    out: Dict[str, List[Mapping[str, Any]]] = {"train": [], "val": []}
    for s in sentences:
        out["val" if s["page_title"] in val_pages else "train"].append(s)  # a page's sentences all go to the same split
    return out


# ----------------------------------------------------------------------
# Quality gate (PLAN §3.4, automated part)
# ----------------------------------------------------------------------
def quality_gate(
    pairs: Sequence[Pair], targets: Mapping[str, float], p_clean: float, tolerance: float,
    types: Sequence[str] = ERROR_TYPES,
) -> Dict[str, Any]:
    """Check no-ops, duplicates, span validity, type shares, NONE share, leakage (``types``: the 7 types + PRON)."""
    ERROR_TYPES = tuple(types)  # noqa: N806  the checks below run over these types
    noop = sum(1 for p in pairs if p.edits and p.source == p.target)  # pairs with edits but no real change (bad)
    src_counts = Counter(p.source for p in pairs)  # how often each source sentence occurs
    dups = sum(c - 1 for c in src_counts.values() if c > 1)  # number of extra copies
    bad_spans = 0  # pairs whose edit records are inconsistent
    for p in pairs:
        ok = True
        for e in p.edits:
            s0, s1 = e.src_span
            t0, t1 = e.tgt_span
            ok &= 0 <= s0 <= s1 <= len(p.source_tokens) and 0 <= t0 <= t1 <= len(p.target_tokens)  # spans inside the sentence
            ok &= " ".join(p.source_tokens[s0:s1]) == e.src  # span text matches the stored wrong text
            ok &= " ".join(p.target_tokens[t0:t1]) == e.tgt  # span text matches the stored correct text
            ok &= e.type in ERROR_TYPES  # valid type
        ok &= undo_edits(p.source_tokens, p.edits) == list(p.target_tokens)  # undoing the edits gives back the clean sentence
        bad_spans += not ok  # True counts as 1
    type_counts = Counter(e.type for p in pairs for e in p.edits)  # edits per type
    n_edits = sum(type_counts.values()) or 1  # total edits (avoid divide-by-zero)
    shares = {t: type_counts[t] / n_edits for t in ERROR_TYPES}  # share of each type
    type_ok = all(abs(shares[t] - targets[t]) <= tolerance for t in ERROR_TYPES)  # each within ±2 points of target
    none_share = sum(1 for p in pairs if not p.edits) / max(len(pairs), 1)  # share of error-free pairs
    pages = {sp: {p.page_title for p in pairs if p.split == sp} for sp in ("train", "val")}  # pages in each split
    leak = len(pages["train"] & pages["val"])  # pages appearing in both (should be 0)
    return {  # summary report
        "pairs": len(pairs), "edits": sum(type_counts.values()),
        "noop": noop, "duplicates": dups, "bad_spans": bad_spans,
        "type_counts": dict(type_counts), "type_shares": shares, "type_ok": type_ok,
        "none_share": none_share, "none_ok": abs(none_share - p_clean) <= tolerance,
        "page_leak": leak,
        "errors_per_pair": Counter(len(p.edits) for p in pairs),  # how many pairs have 0, 1, 2 errors
        "passed": (noop == 0 and dups == 0 and bad_spans == 0 and type_ok  # overall pass only if every check passes
                   and abs(none_share - p_clean) <= tolerance and leak == 0),
    }
