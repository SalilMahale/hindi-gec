# -*- coding: utf-8 -*-
"""System A — classical baseline: rule-generated candidates scored by an n-gram language model.

The idea
--------
A language model (LM) trained on clean biography sentences knows which word
sequences are usual. For a possibly wrong sentence, the rules propose
alternatives at the places where the 8 error types happen (verb forms,
postpositions, ने, auxiliaries, pronouns, spelling); the LM scores each
alternative sentence; a change is accepted only if it makes the sentence
clearly more probable. No neural network, CPU only.

Stupid backoff (Brants et al., 2007)
------------------------------------
For a word w after the context h = (w_{i-n+1} … w_{i-1}):

    S(w | h) = c(h w) / c(h)               if the n-gram h w was seen
             = α · S(w | h')               otherwise, h' = h without its first word
    S(w)     = c(w) / N                    unigram (N = number of training tokens)
    S(unseen w) = 1 / (N + V)              a floor (V = vocabulary size)

c(·) are counts, α = 0.4 the backoff penalty. S is not a normalised
probability, but it ranks sentences well, which is all we need. The sentence
score is Σ_i log S(w_i | h_i) over the tokens and the closing </s>, with two
<s> … padding tokens at the start.

Decision rule
-------------
gain(candidate) = score(candidate sentence) − score(current sentence).
The best candidate is accepted if gain > margin; then candidates are
proposed again on the changed sentence (positions already changed are not
touched again), up to ``max_changes`` changes. The margin trades precision
against recall and is tuned on HiWikEd dev_select.
"""

from __future__ import annotations  # modern type-hint syntax

import math  # log
from collections import Counter  # n-gram counts
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple  # type hints

from .error_generator import ERG_PRONOUNS, HON_TO_SG, POSSESSIVE_FORMS  # pronoun tables
from .morphology import (CELLS, find_verb_groups, is_content_word,  # verb knowledge
                         render_group, spelling_variants)

BOS, EOS = "<s>", "</s>"
POSTP_SET = ("में", "पर", "से", "को", "तक")
GENITIVE_SET = ("का", "की", "के")
AUX_TENSE = {"था": "है", "थे": "हैं", "थी": "है", "थीं": "हैं", "है": "था", "हैं": "थे"}  # tense swap (number kept)
_ERG_BACK = {v: k for k, v in ERG_PRONOUNS.items()}  # वे -> उन्होंने
_SG_TO_HON = {v: k for k, v in HON_TO_SG.items()}  # उसने -> उन्होंने

Candidate = Tuple[int, int, Tuple[str, ...]]  # replace tokens[start:end] with the new tokens


class NGramLM:
    """Stupid-backoff n-gram LM (see module docstring)."""

    def __init__(self, order: int, alpha: float) -> None:
        self.order, self.alpha = order, alpha
        self.counts: Counter = Counter()  # n-gram tuple -> count, all orders 1..n
        self.n_tokens = 0

    def fit(self, sentences: Iterable[Sequence[str]]) -> "NGramLM":
        for s in sentences:
            seq = [BOS] * (self.order - 1) + list(s) + [EOS]
            self.n_tokens += len(s) + 1
            for i in range(self.order - 1, len(seq)):
                for n in range(1, self.order + 1):
                    self.counts[tuple(seq[i - n + 1:i + 1])] += 1
            for n in range(1, self.order):  # context counts that start with <s> padding
                self.counts[tuple(seq[self.order - 1 - n:self.order - 1])] += 1
        self.vocab = sum(1 for k in self.counts if len(k) == 1)
        return self

    def logscore(self, word: str, context: Tuple[str, ...]) -> float:
        """log S(word | context), backing off with penalty alpha."""
        penalty = 0.0
        for k in range(len(context), 0, -1):  # longest context first
            h = context[-k:]
            c = self.counts.get(h + (word,), 0)
            if c:
                return penalty + math.log(c / self.counts[h])
            penalty += math.log(self.alpha)
        c = self.counts.get((word,), 0)
        return penalty + math.log((c or 1) / (self.n_tokens + (0 if c else self.vocab)))

    def sentence_score(self, tokens: Sequence[str]) -> float:
        seq = [BOS] * (self.order - 1) + list(tokens) + [EOS]
        return sum(self.logscore(seq[i], tuple(seq[i - self.order + 1:i]))
                   for i in range(self.order - 1, len(seq)))


def candidates(tokens: Sequence[str], vocab: Mapping[str, int], spell_min_count: int,
               forced_right_of: Mapping[str, str]) -> List[Candidate]:
    """Every rule alternative of the sentence (one change each)."""
    out: Set[Candidate] = set()
    for g in find_verb_groups(tokens):  # AGR: every other gender-number cell of each verb group
        orig = tuple(tokens[g.start:g.end])
        for gender, number in CELLS:
            new = tuple(render_group(tokens, g, gender, number))
            if new != orig:
                out.add((g.start, g.end, new))
    for i, t in enumerate(tokens):
        alts: List[Tuple[str, ...]] = []
        if t in POSTP_SET and i > 0:
            alts += [(p,) for p in POSTP_SET if p != t]
        if t in GENITIVE_SET and i > 0:
            alts += [(p,) for p in GENITIVE_SET if p != t]
        if t in AUX_TENSE:
            alts.append((AUX_TENSE[t],))
        if t in ERG_PRONOUNS:
            alts.append((ERG_PRONOUNS[t],))
        if t in _ERG_BACK:
            alts.append((_ERG_BACK[t],))
        if t == "ने":
            out.add((i, i + 1, ()))  # delete ने
        if t in HON_TO_SG:
            alts.append((HON_TO_SG[t],))
        if t in _SG_TO_HON:
            alts.append((_SG_TO_HON[t],))
        if t in POSSESSIVE_FORMS:
            stem, end = POSSESSIVE_FORMS[t]
            for e in GENITIVE_SET:
                if e != end:
                    alts.append(({"का": "अपना", "की": "अपनी", "के": "अपने"}[e] if stem == "अप" else stem + e,))
        if t in forced_right_of:
            alts.append((forced_right_of[t],))
        if is_content_word(t):  # spelling: only toward a clearly more frequent corpus form
            c = vocab.get(t, 0)
            alts += [(v,) for v in spelling_variants(t) if vocab.get(v, 0) >= max(spell_min_count, 2 * c + 1)]
        for a in alts:
            out.add((i, i + 1, a))
    return sorted(out)


def apply(tokens: Sequence[str], c: Candidate) -> List[str]:
    a, b, new = c
    return list(tokens[:a]) + list(new) + list(tokens[b:])


def trajectory(tokens: Sequence[str], lm: NGramLM, vocab: Mapping[str, int], max_changes: int,
               spell_min_count: int, forced_right_of: Mapping[str, str],
               margin: float = -math.inf) -> List[Tuple[float, List[str]]]:
    """The greedy steps: [(gain, sentence after the step), …], stopping when gain <= margin.

    The steps do not depend on the margin (each is the best remaining candidate),
    so one trajectory with margin = -inf serves every margin: the correction for
    margin m is the sentence after the last of the leading steps whose gain > m.
    """
    steps: List[Tuple[float, List[str]]] = []
    cur = list(tokens)
    cur_score = lm.sentence_score(cur)
    touched: Set[int] = set()  # positions (in the current sentence) already changed
    for _ in range(max_changes):
        best: Optional[Tuple[float, List[str], Candidate]] = None
        for c in candidates(cur, vocab, spell_min_count, forced_right_of):
            a, b, _ = c
            if touched & set(range(a, max(b, a + 1))):
                continue
            new = apply(cur, c)
            gain = lm.sentence_score(new) - cur_score
            if best is None or gain > best[0]:
                best = (gain, new, c)
        if best is None or best[0] <= margin:
            break
        gain, cur, (a, b, new) = best
        cur_score += gain
        shift = len(new) - (b - a)
        touched = {p if p < a else p + shift for p in touched} | set(range(a, a + max(len(new), 1)))
        steps.append((gain, list(cur)))
    return steps


def at_margin(tokens: Sequence[str], steps: Sequence[Tuple[float, List[str]]], margin: float) -> List[str]:
    """The correction a given margin produces from a margin-free trajectory."""
    out = list(tokens)
    for gain, sent in steps:
        if gain <= margin:
            break
        out = sent
    return out


def correct(tokens: Sequence[str], lm: NGramLM, vocab: Mapping[str, int], margin: float,
            max_changes: int, spell_min_count: int, forced_right_of: Mapping[str, str]) -> List[str]:
    """Greedy LM-scored correction (see module docstring)."""
    return at_margin(tokens, trajectory(tokens, lm, vocab, max_changes, spell_min_count, forced_right_of, margin),
                     margin)
