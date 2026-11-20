# -*- coding: utf-8 -*-
"""Build a clean Hindi biographical corpus from Wikipedia (PLAN §2.2).

Why this file is long: a language model is only as good as the sentences it
never had to guess about. Every filter below exists to keep one specific kind
of garbage out of `bio_clean.jsonl`.

**Why the extracts API and not wikitext.** `prop=extracts&explaintext=1`
returns the rendered article as plain prose with infoboxes, tables, reference
markers and templates already stripped. Parsing raw wikitext would mean
reimplementing that, badly. The cost is that MediaWiki refuses to batch full
article extracts — it answers `"exlimit" was too large ... lowered to 1` — so
this is strictly one HTTP request per biography, which is what makes the rate
limiter and the on-disk cache load-bearing rather than decorative.

**Why only some sections.** A biography page is prose at the top ("उनका जन्म
1931 में रामेश्वरम में हुआ था।") and increasingly list-like at the bottom
(bibliographies, award tables, external links). Keeping the lead plus the
life sections keeps the register narrative and past-tense, which is the entire
reason the domain was narrowed.

**Why the quality filters.** In order:

* *6–40 tokens* — under six tokens is usually a stub fragment with no verb to
  agree with; over forty and Stanza's dependency parse degrades, which the
  agreement rules in System A depend on.
* *≥ 80 % Devanagari* — Hindi Wikipedia leads are full of parenthetical
  transliterations (`अब्दुल कलाम (तमिल: ..., अंग्रेज़ी: Avul Pakir ...)`).
  Those sentences are majority non-Devanagari and would teach the model
  nothing about Hindi agreement.
* *must end in danda* — a sentence that does not is almost always a fragment
  the segmenter cut badly, or a heading.
* *balanced brackets / no citation markers / no leading bullet* — all signs of
  list or template residue that survived the extractor.

**Why near-duplicate removal, not just exact.** Wikipedia biographies share
boilerplate ("उनका जन्म X में हुआ था।" with only the year differing). Exact
dedup misses these, and they would let the language model memorise a template
instead of learning agreement. Sentences are shingled over *stems*, so
inflectional variation does not hide a duplicate, and compared with MinHash
LSH so the cost stays linear instead of quadratic in corpus size.
"""

from __future__ import annotations  # modern type-hint syntax

import hashlib  # hash functions (sha1 for cache file names, blake2b for shingles)
import json  # JSON encode/decode
import random  # seeded random numbers (MinHash coefficients)
import re  # regular expressions
import time  # sleeping / timing for the rate limiter
import urllib.error  # HTTP error classes
import urllib.parse  # build URL query strings
import urllib.request  # make HTTP requests
from dataclasses import dataclass  # simple record class
from pathlib import Path  # file paths
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple  # type hints

from .preprocessing import normalize, remove_noise, split_sentences, stem, tokenize  # cleaning steps
from .schema import CleanSentence  # output record type

CATEGORY_PREFIX = "श्रेणी:"  # Hindi Wikipedia's word for "Category:"

#: Residue the extracts API does not always remove: [1], [२], [citation needed].
CITATION_RE = re.compile(r"\[\s*[\d०-९]+\s*\]|\[[^\]]{0,40}(needed|उद्धरण)[^\]]{0,40}\]")  # matches [12] or [citation needed]-style markers
#: A line that is a list item rather than prose.
BULLET_RE = re.compile(r"^\s*[-*•—–·]|^\s*[\d०-९]+[.)]\s")  # line starting with a bullet or "1." / "1)"
BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}", "«": "»", "“": "”"}  # opening -> closing bracket


# ======================================================================
# 1. API client
# ======================================================================
class WikiClient:
    """Polite MediaWiki API client with disk cache and rate limiting.

    Three behaviours matter:

    * **Cache first.** Every response is stored under a hash of its query
      parameters, so re-running the build costs no network at all. Wikipedia
      asks that bots not re-fetch what they already have, and it makes
      iterating on the *filters* free.
    * **Rate limit.** A simple spacing limiter holds requests to at most
      `requests_per_second` (PLAN §2.2 caps this at 5).
    * **Back off on 429.** Wikimedia returns HTTP 429 on bursts; retrying
      immediately makes it worse, so each retry waits twice as long.
    """

    def __init__(
        self,
        api_url: str,  # e.g. https://hi.wikipedia.org/w/api.php
        user_agent: str,  # identifies our program (Wikipedia requires a contact)
        requests_per_second: float = 4.0,  # maximum request rate
        cache_dir: Optional[Path] = None,  # folder for cached responses
        max_retries: int = 8,  # how many times to retry a failed request
        timeout: int = 60,  # seconds to wait for a response
    ) -> None:
        self.api_url = api_url
        self.user_agent = user_agent
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0  # minimum seconds between requests (0.25 s)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)  # create the cache folder
        self.max_retries = max_retries
        self.timeout = timeout
        self._last_request = 0.0  # time of the previous request
        self.stats = {"network": 0, "cache": 0, "retries": 0}  # counters for the build summary

    # -------------------------------------------------- cache key
    def _cache_path(self, params: Dict[str, str]) -> Optional[Path]:  # file name for a given set of query parameters
        if not self.cache_dir:
            return None
        key = json.dumps(params, sort_keys=True, ensure_ascii=False)  # parameters as a stable string
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()  # short unique fingerprint of that string
        return self.cache_dir / f"{digest}.json"  # e.g. wiki_cache/0021480996....json

    # -------------------------------------------------- request
    def get(self, **params: str) -> Dict:
        """Perform one API call, honouring cache, rate limit and backoff."""
        params.setdefault("format", "json")  # ask for JSON responses
        params.setdefault("formatversion", "2")  # newer, simpler JSON format

        cache_file = self._cache_path(params)
        if cache_file and cache_file.exists():  # we already have this response
            self.stats["cache"] += 1
            with open(cache_file, encoding="utf-8") as f:
                return json.load(f)  # return it without using the network

        url = self.api_url + "?" + urllib.parse.urlencode(params)  # full request URL
        delay = 5.0  # initial wait before a retry (seconds)
        for attempt in range(self.max_retries):  # try up to max_retries times
            elapsed = time.monotonic() - self._last_request  # time since the last request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)  # wait so we don't exceed the rate limit
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})  # build the request with our user agent
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # send it
                    data = json.loads(resp.read().decode("utf-8"))  # read and parse the JSON reply
                self._last_request = time.monotonic()
                self.stats["network"] += 1
                if cache_file:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False)  # save to cache for next time
                return data
            except urllib.error.HTTPError as exc:  # server returned an error code
                self._last_request = time.monotonic()
                if exc.code in (429, 503) and attempt < self.max_retries - 1:  # "too many requests" / "service unavailable"
                    self.stats["retries"] += 1
                    # Wikimedia tells you how long to wait; obey it rather than
                    # guessing, and fall back to exponential backoff if absent.
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        wait = float(retry_after) if retry_after else delay
                    except (TypeError, ValueError):
                        wait = delay
                    time.sleep(min(wait, 120.0))  # wait (max 2 minutes)
                    delay = min(delay * 2, 120.0)  # double the delay for next time (exponential backoff)
                    continue
                raise  # other HTTP errors: give up
            except (urllib.error.URLError, TimeoutError):  # network problem or timeout
                self._last_request = time.monotonic()
                if attempt < self.max_retries - 1:
                    self.stats["retries"] += 1
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise RuntimeError("unreachable")  # the loop always returns or raises

    # -------------------------------------------------- categories
    def category_members(self, category: str, cmtype: str = "page", hard_limit: int = 5000) -> List[str]:
        """All member titles of a category, following continuation tokens."""
        title = category if category.startswith(CATEGORY_PREFIX) else CATEGORY_PREFIX + category  # add "श्रेणी:" if missing
        out: List[str] = []
        cont: Dict[str, str] = {}  # continuation token (the API returns results in pages of 500)
        while True:
            data = self.get(
                action="query", list="categorymembers", cmtitle=title,
                cmlimit="500", cmtype=cmtype, **cont,  # cmtype = "page" for articles, "subcat" for subcategories
            )
            members = data.get("query", {}).get("categorymembers", [])
            out.extend(m["title"] for m in members)  # collect titles
            if "continue" not in data or len(out) >= hard_limit:  # no more pages of results
                break
            cont = {k: v for k, v in data["continue"].items()}  # ask for the next page
        return out[:hard_limit]

    def walk_categories(self, seeds: Sequence[str], depth: int = 1) -> List[str]:
        """Collect article titles from seed categories and their subcategories.

        Depth 1 means: the seed's own pages, plus the pages of each direct
        subcategory. Going deeper on Wikipedia's category graph quickly drifts
        off-domain (a biography category three levels down can be a list of
        films), which is exactly the drift the narrowed scope is avoiding.
        """
        seen: Set[str] = set()  # titles already collected
        titles: List[str] = []
        for seed in seeds:  # e.g. "भारतीय वैज्ञानिक"
            cats = [seed]
            if depth >= 1:
                cats.extend(self.category_members(seed, cmtype="subcat"))  # add direct subcategories
            for cat in cats:
                for t in self.category_members(cat, cmtype="page"):  # each article in the category
                    if t not in seen and ":" not in t:   # skip other namespaces
                        seen.add(t)
                        titles.append(t)
        return titles

    # -------------------------------------------------- page text
    def page_extract(self, title: str) -> Tuple[str, Optional[int]]:
        """Plain-text article extract and its current revision id.

        `exlimit` is pinned to 1 because MediaWiki silently lowers it anyway
        for whole-article extracts; sending 1 keeps the warning out of the log.
        """
        data = self.get(
            action="query", prop="extracts|revisions", rvprop="ids",  # ask for plain text + revision id
            titles=title, explaintext="1", exsectionformat="wiki", exlimit="1",  # plain text, headings as "== title =="
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages:  # page not found
            return "", None
        page = pages[0]
        revid = None
        revs = page.get("revisions") or []
        if revs:
            revid = revs[0].get("revid")  # revision number
        return page.get("extract") or "", revid  # (article text, revision id)


# ======================================================================
# 2. Section selection
# ======================================================================
SECTION_RE = re.compile(r"^(=+)\s*(.+?)\s*\1\s*$", re.M)  # matches heading lines like "== प्रारंभिक जीवन =="


def split_into_sections(extract: str) -> List[Tuple[str, str]]:
    """Split a plain-text extract into (section_title, body) pairs.

    The lead section is returned first with an empty title.
    """
    matches = list(SECTION_RE.finditer(extract))  # all heading lines
    sections: List[Tuple[str, str]] = []
    lead_end = matches[0].start() if matches else len(extract)  # lead = text before the first heading
    lead = extract[:lead_end].strip()
    if lead:
        sections.append(("", lead))  # lead section has an empty title
    for i, m in enumerate(matches):  # each heading
        body_start = m.end()  # body starts after the heading
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(extract)  # ...and ends at the next heading
        body = extract[body_start:body_end].strip()
        if body:
            sections.append((m.group(2).strip(), body))  # (heading text, body)
    return sections


def select_sections(
    sections: Sequence[Tuple[str, str]], keep_titles: Sequence[str]
) -> List[Tuple[str, str]]:
    """Keep the lead plus any section whose title matches a wanted one.

    Matching is substring-based on the NFC-normalised title, so
    "प्रारम्भिक जीवन", "प्रारंभिक जीवन" and "वैज्ञानिक जीवन" all match the
    keyword "जीवन". Hindi Wikipedia has no single house style for these
    headings, and an exact-match list would silently discard most pages.
    """
    keys = [normalize(k) for k in keep_titles]  # normalised wanted headings (जीवन, शिक्षा, करियर ...)
    out: List[Tuple[str, str]] = []
    for title, body in sections:
        if title == "":  # always keep the lead
            out.append((title, body))
            continue
        norm = normalize(title)
        if any(k in norm or norm in k for k in keys):  # substring match either way
            out.append((title, body))
    return out


# ======================================================================
# 3. Sentence quality filters
# ======================================================================
@dataclass
class QualityConfig:  # thresholds for the sentence filters (filled from config.yaml)
    min_tokens: int = 6  # shortest allowed sentence
    max_tokens: int = 40  # longest allowed sentence
    min_devanagari_ratio: float = 0.80  # at least 80 % Hindi letters
    must_end_in_danda: bool = True  # sentence must end with ।
    reject_unbalanced_brackets: bool = True
    reject_citation_markers: bool = True
    reject_leading_bullet: bool = True


def devanagari_ratio(text: str) -> float:
    """Share of non-space, non-digit, non-punctuation chars that are Devanagari."""
    considered = [c for c in text if not c.isspace() and not c.isdigit()
                  and c not in "।॥,.?!;:'\"()[]{}-–—…%/"]  # letters only
    if not considered:
        return 0.0
    dev = sum(1 for c in considered if "ऀ" <= c <= "ॿ")  # count Devanagari characters (Unicode block 0900-097F)
    return dev / len(considered)  # fraction


def brackets_balanced(text: str) -> bool:
    """True if every opening bracket has its matching close, in order."""
    stack: List[str] = []  # open brackets not yet closed
    closers = {v: k for k, v in BRACKET_PAIRS.items()}  # closing -> opening
    for ch in text:
        if ch in BRACKET_PAIRS:  # opening bracket
            stack.append(ch)
        elif ch in closers:  # closing bracket
            if not stack or stack[-1] != closers[ch]:  # nothing open, or wrong kind
                return False
            stack.pop()  # matched -> remove
    return not stack  # balanced only if nothing is left open


def sentence_rejection_reason(
    text: str, tokens: Sequence[str], qc: QualityConfig
) -> Optional[str]:
    """Return why a sentence should be dropped, or None if it is good.

    Returning the reason rather than a bool is deliberate: the build script
    reports a rejection histogram, which is how you notice that a filter is
    misconfigured instead of silently losing 90 % of the corpus.
    """
    if qc.reject_leading_bullet and BULLET_RE.search(text):
        return "leading bullet / list item"
    if qc.reject_citation_markers and CITATION_RE.search(text):
        return "citation marker"
    # Script ratio is checked before length on purpose: it reads the raw text,
    # so the reported reason does not depend on which tokeniser is installed.
    # (The pure-Python fallback tokeniser drops non-Devanagari runs, which would
    # otherwise turn a transliteration-heavy sentence into a "too short" one.)
    if devanagari_ratio(text) < qc.min_devanagari_ratio:
        return "not enough Devanagari"
    content = [t for t in tokens if t not in "।॥,.?!;:"]  # tokens excluding punctuation
    if len(content) < qc.min_tokens:
        return "too short"
    if len(content) > qc.max_tokens:
        return "too long"
    if qc.must_end_in_danda and not text.rstrip().endswith("।"):
        return "does not end in danda"
    if qc.reject_unbalanced_brackets and not brackets_balanced(text):
        return "unbalanced brackets"
    return None  # passed every filter


# ======================================================================
# 4. Near-duplicate removal (MinHash + LSH over stem shingles)
# ======================================================================
_MERSENNE = (1 << 61) - 1  # a large prime (2^61 - 1) used for the hash functions


class NearDuplicateFilter:
    """Streaming near-duplicate detector over stem shingles.

    How it works, in one breath: each sentence becomes a set of overlapping
    *stem* n-grams (shingles); each set is squashed into a short MinHash
    signature whose collision probability equals the sets' Jaccard similarity;
    the signature is cut into bands, and two sentences are only ever compared
    if a whole band matches. That turns an O(n²) all-pairs comparison into
    something close to O(n), which matters at 40 000+ sentences.

    Stems rather than surface words, because "गया" / "गई" / "गए" are the same
    template with a different agreement suffix — and boilerplate is exactly
    what we are trying to catch.

    **What the threshold actually means.** Sentences are short, so a single
    differing content word moves Jaccard a long way. Measured on real
    biography sentences with `shingle_size=3`:

    ==========================================  ==============
    difference between the two sentences        Jaccard
    ==========================================  ==============
    only spacing / punctuation                  1.00
    same stems, different inflection            0.60
    one content word (a year, a place)          0.33
    ==========================================  ==============

    At the configured 0.90 this therefore removes sentences that are
    *effectively identical* and leaves near-template sentences alone. That is
    deliberate: "उनका जन्म 1931 में हुआ था।" and "उनका जन्म 1935 में हुआ था।"
    are two genuine biographical facts, not a duplicate, and train/val leakage
    is already prevented by splitting on source page (PLAN §2.4). Lowering the
    threshold below ~0.5 would start deleting real data.
    """

    def __init__(
        self,
        shingle_size: int = 4,  # words per shingle (n-gram size)
        threshold: float = 0.90,  # Jaccard similarity at/above which a sentence is a duplicate
        num_hashes: int = 32,  # length of the MinHash signature
        bands: int = 8,  # LSH bands (32 hashes / 8 bands = 4 rows per band)
        seed: int = 42,  # seed for the random hash coefficients
    ) -> None:
        if num_hashes % bands != 0:
            raise ValueError("num_hashes must be divisible by bands")
        self.shingle_size = shingle_size
        self.threshold = threshold
        self.num_hashes = num_hashes
        self.bands = bands
        self.rows = num_hashes // bands  # hash values per band
        rng = random.Random(seed)
        self._a = [rng.randrange(1, _MERSENNE) for _ in range(num_hashes)]  # random multipliers: hash_i(x) = (a_i*x + b_i) mod p
        self._b = [rng.randrange(0, _MERSENNE) for _ in range(num_hashes)]  # random offsets
        self._buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}  # (band no., band values) -> sentence indices
        self._signatures: List[Tuple[int, ...]] = []  # signature of every sentence kept so far

    # -------------------------------------------------- helpers
    def _shingles(self, tokens: Sequence[str]) -> Set[int]:  # sentence -> set of hashed stem n-grams
        stems = [stem(t) for t in tokens if t not in "।॥,.?!;:"]  # stem every non-punctuation word
        k = self.shingle_size
        if len(stems) < k:  # sentence shorter than one shingle
            grams = [" ".join(stems)] if stems else []
        else:
            grams = [" ".join(stems[i : i + k]) for i in range(len(stems) - k + 1)]  # overlapping k-word windows
        return {int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")  # hash each shingle to a 64-bit integer
                for g in grams}

    def _signature(self, shingles: Set[int]) -> Tuple[int, ...]:  # MinHash signature
        if not shingles:
            return tuple([0] * self.num_hashes)
        sig = []
        for a, b in zip(self._a, self._b):  # for each of the 32 hash functions
            sig.append(min(((a * h + b) % _MERSENNE) for h in shingles))  # keep the minimum hash value over all shingles
        return tuple(sig)

    @staticmethod
    def _estimated_jaccard(s1: Tuple[int, ...], s2: Tuple[int, ...]) -> float:  # fraction of equal signature positions ≈ Jaccard similarity
        return sum(1 for x, y in zip(s1, s2) if x == y) / len(s1)

    # -------------------------------------------------- api
    def is_duplicate(self, tokens: Sequence[str]) -> bool:
        """Check against everything seen so far; if new, remember it."""
        sig = self._signature(self._shingles(tokens))  # signature of this sentence
        candidates: Set[int] = set()  # earlier sentences sharing at least one band
        band_keys = []
        for band in range(self.bands):  # cut the signature into 8 bands of 4 values
            key = (band, sig[band * self.rows : (band + 1) * self.rows])
            band_keys.append(key)
            candidates.update(self._buckets.get(key, ()))  # sentences in the same bucket are candidates
        for idx in candidates:  # compare only with candidates (fast)
            if self._estimated_jaccard(sig, self._signatures[idx]) >= self.threshold:
                return True  # near-duplicate found
        idx = len(self._signatures)  # new sentence -> remember it
        self._signatures.append(sig)
        for key in band_keys:
            self._buckets.setdefault(key, []).append(idx)  # add to each band's bucket
        return False


# ======================================================================
# 5. Page -> sentences
# ======================================================================
def sentences_from_extract(
    extract: str,  # article plain text
    page_title: str,  # page name
    revid: Optional[int],  # revision id
    keep_titles: Sequence[str],  # section headings to keep
    qc: QualityConfig,  # filter thresholds
) -> Tuple[List[CleanSentence], Dict[str, int]]:
    """Turn one article extract into accepted sentences plus a reject histogram."""
    accepted: List[CleanSentence] = []
    rejects: Dict[str, int] = {}  # reason -> count
    for sec_title, body in select_sections(split_into_sections(extract), keep_titles):  # only wanted sections
        for raw_line in body.split("\n"):  # each paragraph/line
            line = raw_line.strip()
            if not line:
                continue
            if qc.reject_leading_bullet and BULLET_RE.search(line):  # list item -> skip whole line
                rejects["leading bullet / list item"] = rejects.get("leading bullet / list item", 0) + 1
                continue
            cleaned = normalize(remove_noise(line))  # clean + normalise
            for sent in split_sentences(cleaned):  # each sentence of the line
                sent = sent.strip()
                if not sent:
                    continue
                toks = tokenize(sent)
                reason = sentence_rejection_reason(sent, toks, qc)  # run quality filters
                if reason:
                    rejects[reason] = rejects.get(reason, 0) + 1  # count the reason
                    continue
                accepted.append(  # sentence passed: store it with provenance
                    CleanSentence(id="", page_title=page_title, revid=revid,
                                  section=sec_title, text=sent, tokens=toks)
                )
    return accepted, rejects
