# -*- coding: utf-8 -*-
"""Decode HiWikEd-bio test with the corrector, ONCE, and save the raw outputs.

The test set is read only after every choice (data mix, epochs, thresholds,
example chips) was made on dev_select and synthetic val. This script decodes
the test split a single time with the same beam search as every other
evaluation and saves the raw outputs to
reports/predictions/final_test_corrector.jsonl. Scoring (hallucination guard,
three edit views, GLEU) is done from that file by 19_phase_e.py, so the test
set never has to be decoded again.

To stop the test set from being used twice, the script refuses to run if the
predictions file already exists. It was run on 2026-10-22 (1 823 pairs).

Run:  .venv\\Scripts\\python.exe scripts\\18_final_test.py
"""

from __future__ import annotations  # modern type-hint syntax

import json  # write the predictions
import sys  # stdout, import path
from pathlib import Path  # file paths

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # print Hindi
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # make hindi_gec importable

from hindi_gec.config import load_config  # noqa: E402
from hindi_gec.inference import Corrector  # noqa: E402
from hindi_gec.schema import read_jsonl  # noqa: E402
from hindi_gec.scoring import decode_all  # noqa: E402


def main() -> None:
    cfg = load_config()
    pred = cfg.path("predictions") / "final_test_corrector.jsonl"
    if pred.exists():
        sys.exit(f"{pred} already exists: the test set has been used. Not running again "
                 "(scores: scripts/19_phase_e.py).")
    rows = [r for r in read_jsonl(cfg.path("hiwiked_bio")) if r["split"] == "test"]
    print("=" * 72)
    print(f"FINAL TEST : {cfg.path('model')} on hiwiked_bio test ({len(rows)} pairs), used once")
    print("=" * 72)
    raws = decode_all(Corrector(cfg), rows)
    pred.parent.mkdir(parents=True, exist_ok=True)
    with open(pred, "w", encoding="utf-8") as f:
        for r, o in zip(rows, raws):
            f.write(json.dumps({"id": r["id"], "source": r["source"], "target": r["target"],
                                "raw_output": " ".join(o)}, ensure_ascii=False) + "\n")
    print(f"  raw outputs    : {pred}\n  next           : .venv\\Scripts\\python.exe scripts\\19_phase_e.py")


if __name__ == "__main__":
    main()
