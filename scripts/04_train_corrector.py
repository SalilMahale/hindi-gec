# -*- coding: utf-8 -*-
"""Step 3 — fine-tune IndicBART on bio_synthetic (thin CLI over train_corrector).

Trial (measure speed + VRAM, saves nothing):
    .venv\\Scripts\\python.exe scripts\\04_train_corrector.py --max-steps 200
Full run (Salil starts this in his own PowerShell window):
    .venv\\Scripts\\python.exe scripts\\04_train_corrector.py
Resume after a kill:
    .venv\\Scripts\\python.exe scripts\\04_train_corrector.py --resume

Training data: synthetic train (data/synthetic/bio_synthetic.jsonl) + the real
HiWikEd dev_train pairs (scripts/14_prepare_dev.py) repeated
splits.dev_train_upsample times. Validation loss is on synthetic val.

Output goes to paths.corrector_run (resumable checkpoint, train_log.txt,
train_summary.json, final/). The app loads paths.model (models/corrector); to
use a new run, copy corrector_run/final/* and train_summary.json there.

Everything printed is also appended to <corrector_run>/train_log.txt.
"""

from __future__ import annotations  # modern type-hint syntax

import argparse  # command-line options
import json  # save the training summary
import os  # environment variables
import sys  # stdout, import path
from pathlib import Path  # file paths

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # print Hindi; flush every line
os.environ.setdefault("HF_HUB_OFFLINE", "1")           # weights are cached; never touch the network
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # make hindi_gec importable

from hindi_gec.config import load_config, set_seed  # noqa: E402
from hindi_gec.schema import read_jsonl  # noqa: E402
from hindi_gec.train_corrector import train  # noqa: E402  # the actual training loop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=None, help="trial: stop after N optimiser steps")  # short trial run
    ap.add_argument("--resume", action="store_true")  # continue from the last checkpoint
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)
    out_dir = cfg.path("corrector_run")
    data_path = cfg.path("bio_synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train_log.txt", "a", encoding="utf-8")  # log file (appended to)

    def log(msg: str) -> None:  # print to screen AND write to the log file
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()  # write immediately (so nothing is lost if training is killed)

    log("=" * 72)
    log("STEP 3 : IndicBART CORRECTOR" + (f"  (trial, {args.max_steps} steps)" if args.max_steps else ""))
    log("=" * 72)
    rows = list(read_jsonl(data_path))  # all synthetic pairs
    train_rows = [r for r in rows if r["split"] == "train"]  # 95 % training
    val_rows = [r for r in rows if r["split"] == "val"]  # 5 % validation
    log(f"  data           : {data_path}  train {len(train_rows)}, val {len(val_rows)}")
    split = json.loads(cfg.path("dev_split").read_text(encoding="utf-8"))["split"]  # real pairs from HiWikEd dev_train
    real = [r for r in read_jsonl(cfg.path("hiwiked_bio")) if split.get(r["id"]) == "dev_train"]
    k = cfg["splits"]["dev_train_upsample"]
    train_rows = train_rows + real * k  # each real pair 5x
    log(f"  real dev_train : {len(real)} pairs x {k} = {len(real) * k}  -> train {len(train_rows)}")
    summary = train(cfg, train_rows, val_rows, out_dir, args.max_steps, args.resume, log)  # run training
    log(f"  val loss       : {summary['val_loss']:.4f}")
    log(f"  steps / sec    : {summary['steps_per_sec']:.3f}  ({summary['steps_done']} steps in {summary['seconds']:.0f} s)")
    log(f"  VRAM peak      : allocated {summary['vram_peak_alloc_gib']:.2f} GiB, "
        f"reserved {summary['vram_peak_reserved_gib']:.2f} GiB")
    full_min = summary["total_steps_full_run"] / summary["steps_per_sec"] / 60 if summary["steps_per_sec"] else 0  # projected full-run time in minutes
    log(f"  full run       : {summary['total_steps_full_run']} steps "
        f"({summary['steps_per_epoch']}/epoch) ≈ {full_min:.0f} min at this rate")
    with open(out_dir / ("trial_summary.json" if args.max_steps else "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)  # save the summary


if __name__ == "__main__":
    main()
