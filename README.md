# Hindi Grammar Error Detection and Correction — Biography Domain

NLP mini project. The system finds and fixes grammatical errors in Hindi
biographical prose (जीवनी) with one model, the **IndicBART corrector**. It
uses a fixed error taxonomy: the seven original types plus pronoun forms
(PRON), eight in all. The corrector is trained on a synthetic error generator
grounded in Hindi morphology and on a few hundred real Wikipedia
corrections, and it is compared against a rule + 4-gram language-model
baseline (System A).

Design decisions live in **PROJECT_PLAN.md** and are not re-opened here.

**Final results on real data** (HiWikEd-bio test, 1 823 pairs, scored once;
`reports/results_final.md`):

| system | P | R | F0.5 (all edits) | GLEU |
|---|---|---|---|---|
| IndicBART corrector | 0.627 | 0.325 | **0.529** | 0.7880 |
| System A (rules + 4-gram LM) | 0.392 | 0.155 | 0.300 | 0.7304 |
| do nothing (unchanged input) | — | — | — | 0.7095 |

---

## 1. Setup (Windows 11, Python 3.14, RTX 4060)

Everything is installed into a **virtual environment inside the repo**, never
into the global Python. Two reasons: the global install already carries a
CPU-only `torch` that would shadow the CUDA build, and a venv puts `pytest`
on `PATH` automatically.

```powershell
cd D:\College\Sem7\NLP\NLP_mini_project\hindi_gec

python -m venv .venv
.\.venv\Scripts\Activate.ps1      # PowerShell   (cmd: .venv\Scripts\activate.bat)
python -m pip install --upgrade pip

pip install -r requirements.txt   # pulls torch 2.14.0+cu126 from the CUDA index
pip install -e .
```

If PowerShell blocks the activate script, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**Verify, in this order, from the repo root:**

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import hindi_gec; print(hindi_gec.EXTENDED_TYPES)"
python -m pytest -q
```

The first line must print `2.14.0+cu126 True NVIDIA GeForce RTX 4060`. A bare
`2.14.0 False` means the CPU wheel is still installed — `pip uninstall torch`
and reinstall from `requirements.txt`.

Use `python -m pytest`, not bare `pytest`: it guarantees the tests run under
the same interpreter the package is installed into. On this laptop call the
venv interpreter directly (`.venv\Scripts\python.exe`) rather than relying on
activation.

---

## 2. Configuration

Everything tunable lives in **`configs/config.yaml`** — paths, seed, the
error-type distribution, and every hyperparameter. No script hard-codes a
path or a learning rate. The loader validates on read: the taxonomy must
match `hindi_gec.ERROR_TYPES`, the type distribution must cover the 8 types
and sum to 1, and the split ratios must sum to 1. A typo fails loudly instead
of silently skewing the training data.

```python
from hindi_gec.config import load_config, set_seed
cfg = load_config()
set_seed(cfg.seed)
print(cfg.corrector.lr, cfg.path("model"))    # models/corrector
```

---

## 3. Running the pipeline, in order

Put the HiWikEd files in `data/raw/` first (`hiwiki.extracted.clean.src`,
`hiwiki.extracted.clean.trg`). The figure `app/assets/pipeline.png` shows the
same order.

| # | Script | What it does | Output |
|---|---|---|---|
| 1 | `scripts/01_build_corpus.py` | Scrape + clean Hindi Wikipedia biographies (cached) | `data/processed/bio_clean.jsonl` |
| 2 | `scripts/03_generate_synthetic.py` | Typed synthetic errors, 8 types, quality gate | `data/synthetic/bio_synthetic.jsonl` |
| 3 | `scripts/02_filter_hiwiked.py` | Biography subset of HiWikEd, leaks removed, edits typed | `data/processed/hiwiked_bio.jsonl` |
| 4 | `scripts/14_prepare_dev.py` | dev → dev_train / dev_select; drop near-duplicates of evaluation pairs | `data/processed/hiwiked_dev_split.json` |
| 5 | `scripts/04_train_corrector.py` | Fine-tune IndicBART (≈ 49 min on the GPU) | `models/corrector_run/` → copy `final/` to `models/corrector/` |
| 6 | `scripts/18_final_test.py` | Decode real test **once** (refuses to run twice) | `reports/predictions/final_test_corrector.jsonl` |
| 7 | `scripts/16_baseline_A.py [--final-test]` | System A (rules + 4-gram LM, CPU); test only with the flag | `reports/baseline_A.json`, predictions |
| 8 | `scripts/19_phase_e.py` | All scores from saved outputs: P / R / F0.5, GLEU, confusion, speed, error analysis | `reports/results_final.json` / `.md`, `reports/confusion.png` |
| 9 | `scripts/09_app_stats.py` | Data and training numbers for the app | `reports/app_stats.json` |
| 10 | `scripts/10_app_examples.py` | One example chip per type, each corrected exactly; checks the 5 correct sentences | `reports/app_examples.json` |
| — | `app/streamlit_app.py` | Web demo, offline (backup: `scripts/demo_cli.py`) | — |

Helpers: `00_preprocess_demo.py` (shows each preprocessing step),
`check_tokenizer_roundtrip.py`, `download_indicbart.py`,
`11_build_gazetteer.py` (person-name list for honorific vs plural),
`13_other_patterns.py` (what the real edits outside the taxonomy are),
`20_pipeline_figure.py` (draws the pipeline figure from the saved numbers).

```powershell
.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py     # http://localhost:8501
.venv\Scripts\python.exe scripts\demo_cli.py --examples              # terminal backup
```

### Corpus notes

Every API response is cached under `data/raw/wiki_cache/`, so re-running costs
no network and only re-applies the filters. Two seed categories from the plan
were wrong and have been corrected in `config.yaml`: `भारतीय अभिनेत्रियाँ`
does not exist, and `भारत रत्न` holds 2 pages rather than its 47 recipients
(`भारत रत्न सम्मान प्राप्तकर्ता`).

Section selection matches **substrings**, so `जीवन` also admits
`प्रारम्भिक जीवन`, `व्यक्तिगत जीवन` and `राजनीतिक जीवन`. Hindi Wikipedia has
no house style for these headings and an exact-match list discards most pages.

---

## 4. Repository layout

```
hindi_gec/
  configs/config.yaml        all parameters, validated on load
  data/raw/                  HiWikEd + cached wiki pages   (git-ignored)
  data/processed/            bio_clean, hiwiked_bio, dev split, name gazetteer
  data/synthetic/            bio_synthetic.jsonl
  models/corrector/          THE model (weights, tokenizer, train_summary.json)
  models/baseline_A_4gram.pkl  System A language model
  src/hindi_gec/
      __init__.py            the ⛔ error taxonomy contract (+ PRON)
      config.py              config loading + validation + seeding
      preprocessing.py       noise → NFC → danda split → tokenise
      schema.py              JSONL record types + I/O
      corpus_builder.py      Wikipedia client, filters, near-dup removal
      morphology.py          verb paradigms, postpositions
      error_generator.py     synthetic errors, type quota, quality gate
      edit_extractor.py      (source, output) -> typed edits
      metrics.py, scoring.py edit-level P / R / F0.5, three views
      gleu.py                GLEU (Napoles et al.)
      leakage.py             near-duplicate check
      train_corrector.py     training loop
      inference.py           Corrector: beams, guard, confidence, alternatives
      explain.py, names.py   bilingual explanations, person-name gazetteer
      baseline.py            System A
  scripts/                   thin CLIs (table above)
  app/                       Streamlit demo (4 pages)
  tests/                     pytest, one file per module
  reports/                   results_final.*, app files, predictions/
```

---

## 5. Conventions

* Every module opens with a docstring explaining **the linguistics and the
  why**, not just the what; public functions carry type hints.
* Every entry point begins with `sys.stdout.reconfigure(encoding="utf-8")`
  so the Windows console renders Devanagari instead of raising `cp1252`
  errors.
* All file I/O passes `encoding="utf-8"` explicitly; CSV uses `utf-8-sig`
  so Excel on Windows shows Devanagari rather than mojibake.
* Seeds are fixed at `cfg.seed = 42`.
* Every script prints what it read, what it wrote, and the counts.
* Every reported number traces to a saved script output.

---

## 6. Changes made to the original `hindi_preprocessing.py`

The module was moved to `src/hindi_gec/preprocessing.py` with its behaviour
preserved, except for one deliberate bug fix:

**Fallback tokeniser split the danda wrongly.** The pure-Python fallback used
`[\u0900-\u097F]+`, and the danda `।` is U+0964 — inside that range. So
`"राम बाजार गया।"` tokenised as `['राम', 'बाजार', 'गया।']`, gluing the
sentence-final punctuation onto the verb, whereas
`indic_tokenize.trivial_tokenize` returns `['राम', 'बाजार', 'गया', '।']`.
The range is now `[\u0900-\u0963\u0966-\u097F]`, the same
punctuation-excluding range `remove_noise` was already using. Both paths now
produce identical tokens.

This matters beyond tidiness: edits in the data schema are stored as **token
indices** (PLAN §2.5), so a differently-tokenised sentence shifts every span
after the danda. It only ever affected machines without
`indic-nlp-library` installed.

**Bracket husks left by Latin removal.** Stripping the Latin run from
"विधान सभा सदस्य (MLA)" left "विधान सभा सदस्य ( )". The husk is balanced, so
no bracket check flags it, and it would have reached the training data as a
meaningless token pair. Empty bracket pairs are now removed in the same step.
Measured on a 188-sentence sample: 2 affected before, 0 after.

**Rejection reason independent of the tokenizer.** The Devanagari-ratio check now
runs before the length check. Previously, without `indic-nlp-library`, the fallback
tokenizer dropped Latin/Tamil runs and a transliteration-heavy sentence was reported
as "too short" instead of "not enough Devanagari". The set of accepted sentences is
unchanged — every filter must pass either way.

Also: the console driver moved out of the module into
`scripts/00_preprocess_demo.py` (same output), and `sys.stdout.reconfigure`
moved with it, so importing the library no longer has side effects.
