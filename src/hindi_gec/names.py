# -*- coding: utf-8 -*-
"""Person-name gazetteer: tells honorific plural apart from ordinary plural.

The linguistics
---------------
Hindi uses the plural verb for one respected person: लाल बहादुर शास्त्री एक
अद्भुत इंसान थे ("were") is about one man. So a correction था → थे is an
honorific-agreement fix (AGR-HON) when the subject is a person, and a number
fix (AGR-NUM) when the subject really is several things. The edit classifier
already recognises honorific pronouns (वे, उन्होंने), titles (श्री, डॉ॰) and
-जी words; what it could not see is a bare name, because the live app never
knows the biography's page title. This module supplies that knowledge.

How the list is built
---------------------
Every word of every biography page title in bio_clean (4 551 people) is a
candidate name token, after removing parenthesised disambiguators
("(अभिनेता)"). A candidate is dropped when it is

* not pure Devanagari, or a single character (initials such as ए);
* a function word or postposition (के, की, और …: some titles are phrases);
* among the ``max_common_rank`` most frequent corpus words, unless it is in at
  least ``keep_if_titles`` titles. Frequent words that are rarely in titles
  (भारत, पुरस्कार, सरकार) are ordinary vocabulary; frequent words that are in
  many titles (सिंह, कुमार, शर्मा, गांधी) are names.

Subject
-------
``subject_run`` returns the noun phrase the verb most likely agrees with: the
first run of content words, counted from the start of the sentence, that is
not followed by a postposition (a word followed by का/की/के/में/से … is a
modifier or an adverbial, not the subject; ने still marks the subject).
"""

from __future__ import annotations  # modern type-hint syntax

import re  # regular expressions
from collections import Counter  # word counts
from pathlib import Path  # file paths
from typing import FrozenSet, Iterable, List, Sequence, Tuple  # type hints

from .morphology import FUNCTION_WORDS, POSTPOSITIONS  # closed-class words

_DEVA = re.compile(r"^[ऀ-ॣ॰-ॿ]+$")  # Devanagari letters only (no danda, no digits)
_PARENS = re.compile(r"\(.*?\)")  # "(अभिनेता)" style disambiguators in titles
_STOP = FUNCTION_WORDS | POSTPOSITIONS  # words that are never (part of) a name


def title_tokens(title: str) -> List[str]:
    """Name tokens of one page title (disambiguator removed)."""
    return [w for w in _PARENS.sub(" ", title).split() if _DEVA.match(w)]


def build_gazetteer(titles: Iterable[str], corpus_counts: Counter, max_common_rank: int,
                    keep_if_titles: int) -> FrozenSet[str]:
    """Name tokens from page titles, minus common vocabulary (see module docstring)."""
    in_titles = Counter(w for t in set(titles) for w in set(title_tokens(t)))  # number of titles containing each word
    common = {w for w, _ in corpus_counts.most_common(max_common_rank)}  # the most frequent corpus words
    return frozenset(w for w, n in in_titles.items()
                     if len(w) >= 2 and w not in _STOP and (w not in common or n >= keep_if_titles))


def save_gazetteer(names: Iterable[str], path: Path) -> None:
    Path(path).write_text("".join(w + "\n" for w in sorted(names)), encoding="utf-8")


def load_gazetteer(path: Path) -> FrozenSet[str]:
    """Empty set if the file does not exist (the classifier then behaves as before)."""
    p = Path(path)
    if not p.exists():
        return frozenset()
    return frozenset(w.strip() for w in p.read_text(encoding="utf-8").splitlines() if w.strip())


def subject_span(tokens: Sequence[str], end: int) -> Tuple[int, int]:
    """(start, stop) of the first run of content words in tokens[:end] not followed by a
    postposition other than ने; (0, 0) if there is none."""
    start = None
    for i in range(min(end, len(tokens)) + 1):  # one step past the end closes the last run
        t = tokens[i] if i < end and i < len(tokens) else None
        if t is not None and t not in _STOP and _DEVA.match(t):  # a content word: extend the run
            start = i if start is None else start
            continue
        if start is not None:  # a run just ended at t
            if t is None or t == "ने" or t not in POSTPOSITIONS:  # not followed by का/की/में/से …
                return start, i
            start = None  # a modifier or adverbial: keep looking
    return 0, 0


def subject_run(tokens: Sequence[str], end: int) -> List[str]:
    """The words of subject_span."""
    a, b = subject_span(tokens, end)
    return list(tokens[a:b])


def subject_is_name(tokens: Sequence[str], end: int, names: FrozenSet[str]) -> bool:
    """True if the subject run before position ``end`` contains a known person-name token."""
    return bool(names) and any(w in names for w in subject_run(tokens, end))
