# -*- coding: utf-8 -*-
"""Phase 2 — build the clean Hindi biography corpus (PLAN §2.2).

Walks the seed categories, fetches each biography's plain-text extract, keeps
the lead and life sections, applies the sentence quality filters, removes
exact and near duplicates, and writes `data/processed/bio_clean.jsonl`.

Run:     python scripts/01_build_corpus.py
Resume:  just run it again — every API response is cached on disk, so a
         re-run costs no network and only re-applies the filters.
Options: --target N       stop once N sentences are collected
         --max-pages N    stop after N biographies (useful for a smoke test)
         --no-cache       ignore the cache (rarely wanted)
"""

from __future__ import annotations  # modern type-hint syntax

import argparse  # parses command-line options like --target 100
import sys  # stdout settings, import path
import time  # timing
from collections import Counter  # counting dictionary
from pathlib import Path  # file paths

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # print Hindi correctly; flush each line immediately
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # make hindi_gec importable

from hindi_gec.config import load_config, set_seed  # noqa: E402  # config + reproducible randomness
from hindi_gec.corpus_builder import (  # noqa: E402
    NearDuplicateFilter,  # MinHash near-duplicate detector
    QualityConfig,  # sentence filter thresholds
    WikiClient,  # Wikipedia API client with cache and rate limit
    sentences_from_extract,  # article text -> clean sentences
)
from hindi_gec.schema import write_jsonl  # noqa: E402  # save records as JSONL


def main() -> None:
    ap = argparse.ArgumentParser()  # command-line option parser
    ap.add_argument("--target", type=int, default=None, help="sentence target")  # stop after N sentences
    ap.add_argument("--max-pages", type=int, default=None, help="cap pages fetched")  # limit pages (quick test)
    ap.add_argument("--no-cache", action="store_true")  # ignore the disk cache
    ap.add_argument("--review-sample", type=int, default=50,
                    help="write N random sentences to reports/ for manual reading")
    ap.add_argument("--out", type=str, default=None)  # alternative output file
    args = ap.parse_args()  # read the options given on the command line

    cfg = load_config()
    set_seed(cfg.seed)
    cc = cfg["corpus"]  # corpus section of the config
    target = args.target or cc["target_sentences"]  # 40 000 unless overridden
    out_path = Path(args.out) if args.out else cfg.path("bio_clean")  # data/processed/bio_clean.jsonl

    qc = QualityConfig(**cc["quality_filters"])  # build filter settings from the config (** unpacks the dict as keyword args)
    client = WikiClient(  # Wikipedia API client
        api_url=cc["api_url"],
        user_agent=cc["user_agent"],
        requests_per_second=cc["max_requests_per_second"],
        cache_dir=None if args.no_cache else cfg.path("cache_wiki"),  # data/raw/wiki_cache
    )
    dedup = NearDuplicateFilter(  # near-duplicate remover
        shingle_size=cc["dedup"]["shingle_size"],
        threshold=cc["dedup"]["near_dup_threshold"],
        seed=cfg.seed,
    )

    print("=" * 72)
    print("PHASE 2 : CLEAN BIOGRAPHY CORPUS")
    print("=" * 72)
    print(f"  API            : {cc['api_url']}")
    print(f"  seed categories: {len(cc['categories'])}  (subcategory depth {cc['subcategory_depth']})")
    print(f"  sentence target: {target}")
    print(f"  output         : {out_path}")

    t0 = time.monotonic()  # start timer
    print("\n--- walking categories ---")
    titles = client.walk_categories(cc["categories"], depth=cc["subcategory_depth"])  # list of biography page titles
    print(f"  unique biography pages discovered : {len(titles)}")
    if args.max_pages:
        titles = titles[: args.max_pages]  # keep only the first N pages
        print(f"  capped to                         : {len(titles)}")

    print("\n--- fetching and filtering ---")
    sentences = []  # accepted sentences
    rejects: Counter = Counter()  # reason -> count
    seen_exact: set = set()  # sentence texts already accepted (exact duplicate check)
    pages_used = 0  # pages that gave at least one sentence
    for i, title in enumerate(titles, 1):  # each biography page
        if len(sentences) >= target:  # reached the target
            break
        try:
            extract, revid = client.page_extract(title)  # download (or read from cache) the article text
        except Exception as exc:            # network hiccup on one page is not fatal
            rejects[f"fetch failed: {type(exc).__name__}"] += 1
            continue
        if not extract:
            rejects["empty extract"] += 1
            continue
        accepted, page_rejects = sentences_from_extract(  # clean, split and filter this page
            extract, title, revid, cc["keep_sections"], qc
        )
        rejects.update(page_rejects)  # add this page's rejection counts
        kept_here = 0
        for s in accepted:
            if s.text in seen_exact:  # exact copy already kept
                rejects["exact duplicate"] += 1
                continue
            if dedup.is_duplicate(s.tokens):  # nearly identical to a kept sentence
                rejects["near duplicate"] += 1
                continue
            seen_exact.add(s.text)
            s.id = f"bio_{len(sentences) + 1:06d}"  # give it an id like bio_000123
            sentences.append(s)
            kept_here += 1
        if kept_here:
            pages_used += 1
        if i % 50 == 0:  # progress line every 50 pages
            print(f"  {i:5d} pages -> {len(sentences):6d} sentences "
                  f"(net {client.stats['network']}, cache {client.stats['cache']})")

    elapsed = time.monotonic() - t0  # total time
    n = write_jsonl(sentences, out_path)  # save the corpus

    print("\n--- rejection histogram ---")
    for reason, count in rejects.most_common():  # why sentences were dropped
        print(f"  {reason:34s}: {count}")

    print("\n--- summary ---")
    print(f"  API calls                     : {client.stats['network'] + client.stats['cache']}"
          f"  (network {client.stats['network']}, cache {client.stats['cache']},"
          f" retries {client.stats['retries']})")
    print(f"  pages contributing sentences  : {pages_used}")
    print(f"  sentences written             : {n}")
    print(f"  yield per contributing page   : {n / pages_used:.1f}" if pages_used else "")  # average sentences per page
    print(f"  distinct source pages         : {len({s.page_title for s in sentences})}")
    print(f"  mean tokens per sentence      : "
          f"{sum(len(s.tokens) for s in sentences) / n:.1f}" if n else "")
    print(f"  wall clock                    : {elapsed / 60:.1f} min")
    print(f"  written to                    : {out_path}")

    if args.review_sample and sentences:  # write a random sample for a human to read
        import random as _random

        sample_path = cfg.path("reports") / "bio_clean_review_sample.txt"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        picks = _random.Random(cfg.seed).sample(  # reproducible random sample
            sentences, min(args.review_sample, len(sentences))
        )
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write("Phase 2 manual check — do these read like clean biography prose?\n")
            f.write("Mark any line that is NOT a well-formed Hindi biography sentence.\n")
            f.write("=" * 72 + "\n\n")
            for i, s in enumerate(picks, 1):
                f.write(f"{i:3d}. {s.text}\n")  # the sentence
                f.write(f"     [{s.page_title} / {s.section or 'lead'}]\n\n")  # where it came from
        print(f"  review sample ({len(picks)})            : {sample_path}")


if __name__ == "__main__":
    main()
