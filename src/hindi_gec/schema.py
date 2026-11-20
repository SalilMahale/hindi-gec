# -*- coding: utf-8 -*-
"""On-disk record schemas and JSONL I/O.

Two record types travel through this project.

`CleanSentence` is a single well-formed biography sentence with the Wikipedia
page and revision it came from. Provenance is not bureaucracy here: Wikipedia
is CC BY-SA, so the report has to attribute it, and a revid makes any sentence
in the final corpus re-checkable years later.

`Pair` is the (source, target) record of PLAN §2.5 — a possibly-erroneous
sentence, its correction, and the typed edits between them. Spans are **token
indices, half-open**, which is why tokenisation is frozen in `preprocessing`:
change the tokeniser and every stored span silently shifts.

Everything is written as JSONL — one JSON object per line, UTF-8, no BOM.
JSONL is used rather than CSV because records nest (a list of edit objects)
and because a corrupt line costs you one sentence instead of the whole file.
"""

from __future__ import annotations  # modern type-hint syntax

import json  # convert Python dicts <-> JSON text
from dataclasses import asdict, dataclass, field  # dataclass = auto-generates __init__ etc.; asdict = dataclass -> dict
from pathlib import Path  # file paths
from typing import Any, Dict, Iterable, Iterator, List, Optional  # type hints


@dataclass  # makes a simple "record" class: fields below become constructor arguments
class CleanSentence:
    """One clean biography sentence with its Wikipedia provenance."""

    id: str  # unique id of the sentence
    page_title: str  # Wikipedia page it came from (e.g. a person's name)
    revid: Optional[int]  # Wikipedia revision number (lets us re-check the exact page version); may be None
    section: str                 # "" for the lead section
    text: str                    # detokenised, normalised sentence
    tokens: List[str]  # the sentence split into words/tokens

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)  # convert to a plain dict so it can be saved as JSON


@dataclass
class Edit:
    """A single typed correction between source and target token sequences."""

    src_span: List[int]          # [start, end) token indices into source_tokens
    tgt_span: List[int]  # [start, end) token indices into target_tokens (the corrected sentence)
    src: str  # the wrong text (words in src_span joined)
    tgt: str  # the correct text (words in tgt_span joined)
    type: str                    # one of hindi_gec.ERROR_TYPES (or OTHER)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)  # convert to dict for JSON


@dataclass
class Pair:
    """A (source, target) GEC record — PLAN §2.5."""

    id: str  # unique id of this pair
    origin: str                  # "synthetic" | "hiwiked" | "gold"
    page_title: str  # Wikipedia page the sentence came from (used to keep a page in one split only)
    source: str  # the possibly-wrong sentence (model input)
    target: str  # the correct sentence (what the model should output)
    source_tokens: List[str]  # source split into tokens
    target_tokens: List[str]  # target split into tokens
    edits: List[Edit] = field(default_factory=list)  # list of corrections; default = new empty list for each Pair
    split: str = ""              # "train" | "val" | "dev" | "test"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)  # asdict also converts the nested Edit objects into dicts
        return d


# ----------------------------------------------------------------------
# JSONL I/O
# ----------------------------------------------------------------------
def write_jsonl(records: Iterable[Any], path: str | Path) -> int:
    """Write dataclass records (or plain dicts) to JSONL. Returns the count.

    `ensure_ascii=False` is essential: with the default, every Devanagari
    character would be written as a \\uXXXX escape, making the file unreadable
    for the manual verification steps this project depends on.
    """
    path = Path(path)  # accept str or Path
    path.parent.mkdir(parents=True, exist_ok=True)  # create the output folder if it doesn't exist
    n = 0  # count of records written
    with open(path, "w", encoding="utf-8") as f:  # open file for writing in UTF-8
        for rec in records:  # each record
            obj = rec.to_dict() if hasattr(rec, "to_dict") else rec  # dataclass -> dict; plain dicts stay as they are
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")  # one JSON object per line, Hindi kept readable
            n += 1  # count it
    return n  # how many lines were written


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Stream a JSONL file as dicts, skipping blank lines."""
    with open(path, encoding="utf-8") as f:  # open for reading
        for line in f:  # read one line at a time (memory-friendly for big files)
            line = line.strip()  # remove newline/space at the ends
            if line:  # skip empty lines
                yield json.loads(line)  # "yield" makes this a generator: hand back one dict at a time


def count_jsonl(path: str | Path) -> int:
    """Number of records in a JSONL file."""
    return sum(1 for _ in read_jsonl(path))  # add 1 for every record read
