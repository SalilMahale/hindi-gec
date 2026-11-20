# -*- coding: utf-8 -*-
"""Terminal demo — the backup if Streamlit fails on demo day.

    .venv\\Scripts\\python.exe scripts\\demo_cli.py                     # interactive
    .venv\\Scripts\\python.exe scripts\\demo_cli.py "वे भारत लौट आया।"   # one sentence
    .venv\\Scripts\\python.exe scripts\\demo_cli.py --examples          # the 8 example chips (one per error type)

Works offline: weights load from paths.model (models/corrector).
"""

from __future__ import annotations  # modern type-hint syntax

import argparse  # command-line options
import json  # read the example chips
import os  # environment variables
import sys  # stdin/stdout, import path
from pathlib import Path  # file paths

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # print Hindi
sys.stdin.reconfigure(encoding="utf-8")  # read typed Hindi correctly
os.environ["HF_HUB_OFFLINE"] = "1"  # never go online
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # make hindi_gec importable

from hindi_gec.config import load_config  # noqa: E402
from hindi_gec.inference import Corrector  # noqa: E402


def show(corr: Corrector, sentence: str) -> None:  # correct one sentence and print the result
    r = corr.correct(sentence)
    print(f"\n  input     : {r.input}")
    print(f"  corrected : {r.corrected}")
    if r.guarded:  # hallucination guard kicked in
        print(f"  (guard    : model proposed '{r.raw_output}' — too many changes, input kept)")
    if not r.edits:
        print("  edits     : none — sentence looks correct")
    for e in r.edits:  # each correction: wrong -> right [type]
        print(f"  edit      : {e['src'] or '∅':>14s}  →  {e['tgt'] or '∅':<14s} [{e['type']}]  conf {e['confidence']:.2f}")
        print(f"  why       : {e['explanation']['hi']}\n              {e['explanation']['en']}")
    for a in r.alternatives:  # other beams that differ from the correction
        print(f"  other     : {a['text']}  (beam score {a['score']:.2f})")
    print(f"  confidence: {r.confidence:.2f}   latency: {r.latency_ms:.0f} ms ({corr.device})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sentence", nargs="*")  # optional sentence given on the command line
    ap.add_argument("--examples", action="store_true")  # run the example chips
    args = ap.parse_args()
    cfg = load_config()
    print("Loading corrector …")
    corr = Corrector(cfg)  # load the model once
    if args.examples:
        path = cfg.path("app_examples")
        for ex in json.loads(path.read_text(encoding="utf-8")):
            show(corr, ex["source"])
        return
    if args.sentence:  # a sentence was passed as arguments
        show(corr, " ".join(args.sentence))
        return
    print("Type a Hindi sentence (empty line to quit).")
    while True:  # interactive loop
        try:
            line = input("\n> ").strip()  # read a line from the keyboard
        except (EOFError, KeyboardInterrupt):  # Ctrl+Z / Ctrl+C
            break
        if not line:  # empty line -> quit
            break
        show(corr, line)


if __name__ == "__main__":
    main()
