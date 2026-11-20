# Hindi Grammar Error Detection and Correction — Master Plan (v2, narrowed scope)

Owner: Salil · Course: NLP mini project · Hardware: RTX 4060 (8 GB VRAM), Windows
Status of this document: **all high-level decisions are final.** The implementing model
executes phases in order and does not re-open decisions marked ⛔ without asking.

---

## 0. The one-paragraph version

We narrow the project from "all Hindi" to **Hindi biographical prose (जीवनी)** — sentences
that narrate a person's life, sourced from Hindi Wikipedia biographies of Indian
personalities. We build our own domain corpus (clean sentences), create a typed
**synthetic error generator** grounded in Hindi morphology, keep the biography subset of
HiWikEd as the **real-error test set**, and train a two-model neural system on the 4060:
a **token-level error detector** (MuRIL) and a **seq2seq corrector** (IndicBART), with a
**rule + n-gram baseline** for comparison and explainability. Everything is evaluated with
edit-level Precision / Recall / F0.5 broken down by error type, and demoed in Streamlit.

---

## 1. Why the scope was narrowed and what "narrow" means here ⛔

The teacher's point is correct: a 13k general corpus cannot cover open-domain Hindi.
Narrowing does three things at once:

1. **Vocabulary shrinks** → the language model and the corrector see each word far more
   often → fewer wrong guesses.
2. **Error distribution becomes predictable** → we can enumerate the error types that
   actually occur and target them, instead of chasing a long tail.
3. **We can manufacture unlimited, controlled training data** from clean domain text.

### 1.1 Chosen domain: Hindi biographical prose (जीवनी)

Definition: declarative sentences narrating the life of a real person — birth, family,
education, career, awards, death. Example (clean):

> उनका जन्म 1931 में रामेश्वरम में हुआ था। उनके पिता एक नाविक थे।

Why this domain beats the alternatives considered (sports, news, education, dialogue):

| Property | Biographies | Why it matters |
|---|---|---|
| Past-tense narration dominates | ✔ | Past participles (गया/गई/गए, था/थी/थे) carry **gender + number agreement** — the richest error class in Hindi |
| Honorifics everywhere | ✔ | पिताजी, गुरुजी, महात्मा → **honorific plural agreement** (a uniquely Hindi error) |
| Closed life-event vocabulary | ✔ | जन्म, निधन, विवाह, शिक्षा, पुरस्कार… repeat constantly → sharp LM |
| Abundant clean source | ✔ | Tens of thousands of Hindi Wikipedia biographies, CC BY-SA |
| Real-error test set available | ✔ | HiWikEd can be filtered to its biography subset |
| Low English loanword density | ✔ | Unlike sports/tech, so noise removal doesn't gut sentences |

Rejected alternatives and why: **sports/news** — loanword-heavy, present tense, weak
agreement signal; **school textbooks** — sourcing/licensing friction; **dialogue** —
informal register, HiWikEd has none.

### 1.2 Target error taxonomy ⛔ (this is the contract for every module)

Every module — generator, detector labels, edit classifier, evaluation breakdown, UI colour
legend — uses exactly these codes.

| Code | Name | Example (wrong → right) | How the generator produces it |
|---|---|---|---|
| `AGR-GEN` | Gender agreement | सीता बाजार **गया** → **गई** | Flip gender suffix on verb/adjective whose head noun has known gender |
| `AGR-NUM` | Number agreement | लड़के स्कूल जाता **है** → जाते **हैं** | Flip singular↔plural on verb/aux; must flip consistently within the verb group |
| `AGR-HON` | Honorific agreement | पिताजी सुबह **आया** → **आए** | Subject is honorific noun/ pronoun (आप, पिताजी, जी-suffix, गांधी…) → verb made singular |
| `POSTP` | Postposition | दुनिया **में** घिर जाते हैं → **से** | Substitute within confusion set {में, से, को, पर, का, के, की, तक, ने}; or delete; or insert |
| `ERG` | Ergative ने | **मैं** खाना खाया → **मैंने** | Delete/insert ने on transitive perfective |
| `AUX` | Auxiliary / tense | वह कल **आएगा था** → **आया** | Swap है/था/थे/थी/हैं; रहा/रही/रहे |
| `SPELL` | Orthographic | पुस्त**िक** → पुस्तक, ह**ं**स → ह**ँ**स, ज़/ज | Matra swap (ि↔ी, ु↔ू), anusvara↔chandrabindu, nukta drop, ए/ऐ |
| `NONE` | No error | — | 20 % of generated examples are left clean so the detector learns to say "correct" |

Explicitly **out of scope** (say so in the report): word order, punctuation, semantic
errors, code-mixing.

---

## 2. Data plan

### 2.1 Datasets we will end up with

| Name | Origin | Size target | Role |
|---|---|---|---|
| `bio_clean.jsonl` | Hindi Wikipedia biographies, scraped + preprocessed | 40 000 – 60 000 sentences | Source of truth for synthetic generation and LM |
| `bio_synthetic.jsonl` | Error generator applied to `bio_clean` | 80 000 – 120 000 pairs | Main training data |
| `hiwiked_bio.jsonl` | HiWikEd filtered to biography sentences | ~1 000 – 2 000 pairs | **Real** errors: dev + test (never used to generate) |
| `hiwiked_bio_gold200.jsonl` | 200 pairs of the above, manually verified by Salil | 200 | Gold test set; report headline numbers on this |

### 2.2 Building the clean domain corpus (`corpus_builder.py`)

- Source: Hindi Wikipedia via the MediaWiki API (`https://hi.wikipedia.org/w/api.php`),
  polite user-agent, ≤ 5 requests/s, cache raw pages to disk so re-runs are free.
- Seed categories (walk 1 level of subcategories):
  भारतीय स्वतंत्रता सेनानी, भारतीय वैज्ञानिक, भारतीय राजनीतिज्ञ, हिन्दी साहित्यकार,
  भारतीय अभिनेता, भारतीय अभिनेत्रियाँ, भारतीय क्रिकेट खिलाड़ी, भारत रत्न, पद्म भूषण.
- Keep only the **lead section + "जीवन"/"प्रारंभिक जीवन"/"शिक्षा"/"करियर" sections**.
  Drop infobox, tables, lists, references.
- Run through the existing `hindi_preprocessing.gec_pipeline` (noise removal, NFC,
  danda segmentation, tokenisation). Then apply **sentence quality filters**:
  6 ≤ tokens ≤ 40, ≥ 80 % Devanagari characters, ends in danda, no unbalanced brackets,
  no citation markers, no leading bullet.
- **Deduplicate at sentence level** (exact + near-dup via shingle hashing on stems).
- License: CC BY-SA 4.0 — cite Wikipedia in the report; store `page_title` and `revid`
  per sentence.

### 2.3 Filtering HiWikEd to biographies (`domain_filter.py`)

A sentence pair is "biography" if the **target** sentence scores ≥ 2 on:
- +1 for each life-event keyword: जन्म, जन्मे, जन्मी, निधन, मृत्यु, पुत्र, पुत्री, पिता,
  माता, पत्नी, पति, विवाह, शिक्षा, स्नातक, स्नातकोत्तर, पुरस्कार, सम्मानित, सम्मान,
  प्रोफेसर, कार्यकाल, नियुक्त, प्रसिद्ध, ज्ञात, सदस्य, संस्थापक
- +1 if a 4-digit year (1800–2026) is present
- +1 if a honorific/pronoun-of-respect appears: उन्होंने, उनका, उनकी, उनके, इन्होंने
Tune the threshold so the subset is ≥ 1 000 pairs; report precision by hand-checking 50.

### 2.4 Splits ⛔

- `hiwiked_bio` → **dev 40 % / test 60 %** (real errors are precious; we want the test set
  large). It is **never** used for training and never used as a synthetic-generation
  source.
- `bio_synthetic` → train 95 % / val 5 %, split **by source page** so sentences from one
  biography never leak across the split.
- Salil hand-verifies **200 test pairs** (fix any bad HiWikEd annotations, tag error
  type) → `gold200`. This is the one manual task in the project and it is worth it.

### 2.5 Data schema ⛔ (JSONL, one object per line, UTF-8)

```json
{
  "id": "syn_000123",
  "origin": "synthetic | hiwiked | gold",
  "page_title": "ए॰ पी॰ जे॰ अब्दुल कलाम",
  "source": "उनका जन्म 1931 में रामेश्वरम में हुआ था ।",
  "target": "उनका जन्म 1931 में रामेश्वरम में हुआ था ।",
  "source_tokens": ["उनका", "जन्म", "..."],
  "target_tokens": ["उनका", "जन्म", "..."],
  "edits": [
    {"src_span": [5, 6], "tgt_span": [5, 6], "src": "हुई", "tgt": "हुआ", "type": "AGR-GEN"}
  ],
  "split": "train | val | dev | test"
}
```

Spans are **token indices, half-open**. Edits are stored on every record (empty list for
`NONE`) so the detector's BIO labels can be derived deterministically.

---

## 3. Synthetic error generator (`error_generator.py`) — the intelligent part

### 3.1 Linguistic resources to build (`morphology.py`)

1. **Verb paradigm table.** For the closed set of ~60 high-frequency biography verbs
   (होना, जाना, करना, रहना, बनना, मिलना, लेना, देना, पढ़ना, लिखना, जन्मना, मरना…),
   encode the full agreement grid:
   `{root: {("M","sg"): "गया", ("F","sg"): "गई", ("M","pl"): "गए", ("F","pl"): "गईं"}}`
   for perfective, imperfective (-ता/-ती/-ते), and progressive (रहा/रही/रहे).
   Also a **generic suffix-rule fallback** for verbs outside the table:
   -ा↔-ी↔-े, -ता↔-ती↔-ते.
2. **Auxiliary grid.** है/हैं/हूँ/हो, था/थी/थे/थीं.
3. **Adjective grid** for -ा adjectives (अच्छा/अच्छी/अच्छे, बड़ा/बड़ी/बड़े). Invariant
   adjectives (सुंदर, प्रसिद्ध) are excluded.
4. **Noun gender lexicon.** Seed with ~300 biography nouns tagged M/F (जन्म M, मृत्यु F,
   शिक्षा F, पुरस्कार M, पुस्तक F, विवाह M…). Extend automatically from `bio_clean`
   using Stanza's `Gender=` feature on nouns (majority vote per lemma).
5. **Honorific trigger list.** आप, उन्होंने (as subject), any token ending in जी,
   महात्मा, पंडित, डॉ॰, श्री, श्रीमती, गुरु, स्वामी, and proper names followed by जी.
6. **Postposition confusion sets** with realistic substitution probabilities
   (में↔पर↔से most confusable; का/के/की amongst themselves; को↔ने).
7. **Spelling confusion table**: (ि,ी), (ु,ू), (ं,ँ), (ज़,ज), (फ़,फ), (ए,ऐ), (ो,ौ),
   applied only inside content words, never to postpositions.

### 3.2 Generation algorithm

```
for each clean sentence s (tokens, POS+feats from Stanza hi):
    with p = 0.20 → emit (s, s, NONE); continue
    k = 1 with p 0.75, 2 with p 0.25          # errors per sentence
    candidates = enumerate_applicable_corruptions(s)   # each: (type, position, corrupted_token)
    weight candidates by the target type distribution (see 3.3)
    sample k non-overlapping candidates; apply; record edits with types
    reject if corrupted sentence == clean sentence (no-op)
    reject if corrupted sentence already exists in output (duplicate)
```

Agreement corruptions must be **grammatically coherent as an error**: flipping the verb
also flips its auxiliary (जाता है → जाती है, never जाती है with a stray mismatch), because
that is how humans err.

### 3.3 Target type distribution (matches what real HiWikEd-bio shows) ⛔

AGR-NUM 25 %, AGR-GEN 20 %, POSTP 20 %, SPELL 12 %, AGR-HON 10 %, AUX 8 %, ERG 5 %.
First measure the real distribution on `hiwiked_bio` with the edit classifier (§5.1);
if it differs by > 10 points on any type, adopt the measured distribution and say so.

### 3.4 Quality gate (must pass before training)

- Auto: 0 no-op pairs, 0 duplicates, every edit re-classifies to its declared type.
- Manual: Salil reads 100 random synthetic pairs; ≥ 90 must look like plausible human
  mistakes. If not, fix the generator before spending GPU time.

---

## 4. Models ⛔

Three systems, all evaluated on the same test set. The baseline exists for the viva as
much as for the numbers.

### 4.1 System A — Rule + n-gram baseline (`rules_baseline.py`, `lm.py`)

- **Detection**: Stanza parse → for each verb/adjective find its subject (nsubj) → look up
  subject gender/number/honorific → check agreement with the verb's feats. Flag mismatch.
  Postpositions: flag if the LM probability of the observed postposition is < 0.1 × the
  best alternative's.
- **Correction**: for each flagged token, generate candidates from the paradigm grids /
  confusion sets, re-score the full sentence with a **5-gram Kneser-Ney LM** trained on
  `bio_clean` (nltk `KneserNeyInterpolated`; if too slow, an interpolated stupid-backoff
  implemented directly in Python), pick the argmax.
- Fully CPU, fully explainable: every correction comes with "rule fired: subject पिताजी
  is honorific, verb आया is singular masculine".

### 4.2 System B — Neural detector (`train_detector.py`)

- Model: `google/muril-base-cased` (Hindi-strong BERT, 238M). Token classification.
- Labels: BIO over the 7 error types → 15 tags + O. Predict on the **source** sentence.
- Data: `bio_synthetic` train (+ optionally `hiwiked_bio` dev for early stopping).
- Hyperparameters (fit 8 GB): max_len 64, batch 32, fp16, lr 3e-5, 3 epochs, warmup 6 %,
  weight decay 0.01, class-weighted CE (O is ~90 % of tokens). ~10–15 min on the 4060.
- Output used by: the UI (highlighting) and as a **gating signal** for the corrector
  (sentences with no detected error are returned unchanged → kills false positives).

### 4.3 System C — Neural corrector (`train_corrector.py`)

- Model: `ai4bharat/IndicBART` (244M, mBART-style, Devanagari-native). **Read the model
  card for the exact language-tag convention** (input `... </s> <2hi>`, decoder start
  `<2hi>`) and assert it with a round-trip test before training.
- Task: source sentence → target sentence.
- Hyperparameters: max_len 64/64, batch 16, grad-accum 2, fp16, lr 5e-5, 4 epochs,
  label smoothing 0.1, beam 5 at inference. ~20–30 min on the 4060.
- Two training runs (ablation for the report): **synthetic only** vs
  **synthetic + hiwiked_bio dev** (dev then can't be used for early stopping — use synthetic val).
- **Hallucination guard** at inference: reject any output whose token-level edit distance
  from the source exceeds 3 (or 25 % of length); fall back to source. Seq2seq GEC models
  "rewrite" when unsure; this guard is what makes precision acceptable.

### 4.4 Combined pipeline (`inference.py`)

`detector → if no error: return unchanged → corrector → hallucination guard →
edit extractor → typed edits for display`. Expose `correct(sentence) -> Result` with
`Result = {corrected, edits:[{span, src, tgt, type, confidence}], system:"A|B+C"}`.

---

## 5. Evaluation ⛔

### 5.1 Edit extraction and classification (`edit_extractor.py`)

Align `source_tokens` / `target_tokens` with `difflib.SequenceMatcher` (autojunk off) →
replace/insert/delete spans → merge adjacent edits. Classify each edit by rules:
- both tokens in postposition set → `POSTP`; ने inserted/deleted → `ERG`
- both in auxiliary grid → `AUX`
- same stem, suffix differs, subject honorific → `AGR-HON`; else gender-suffix diff →
  `AGR-GEN`; number-suffix diff → `AGR-NUM`
- character edit distance ≤ 2 and not above → `SPELL`
- otherwise `OTHER` (reported but not trained on)
This same classifier is used to measure the real HiWikEd-bio error distribution (§3.3).

### 5.2 Metrics (`metrics.py`)

- **Detection**: token-level Precision / Recall / F1 per type + macro.
- **Correction**: edit-level **Precision / Recall / F0.5** (the GEC standard; precision
  weighted because a wrong "fix" is worse than a missed one), **GLEU** (Napoles et al.
  2015 — implement directly, do *not* use `nltk.gleu_score`, which is the MT variant),
  **exact match**.
- Report every number on: synthetic val, hiwiked_bio test, gold200. Headline = gold200.
- Required tables in the report: (i) A vs B+C vs B+C(+real) on gold200; (ii) F0.5 per
  error type; (iii) confusion matrix of predicted vs true type; (iv) 10 qualitative
  examples incl. 3 failures.

### 5.3 Success criteria

Not an absolute number — Hindi GEC literature reports F0.5 ≈ 0.3–0.5 on real errors.
Criteria: **B+C beats A by ≥ 10 F0.5 points on gold200**, precision ≥ 0.6, and the
per-type table shows the agreement classes (the domain's whole point) above 0.6.

---

## 6. Demo app (`app/streamlit_app.py`)

- Text box (pre-filled with a biography sentence containing an error), "Check" button.
- Output: source with error spans highlighted, colour-coded by type with a legend;
  corrected sentence; table of edits (span, wrong, right, type, confidence);
  toggle to show System A's rule explanation alongside System B+C.
- Loads models once (`st.cache_resource`); runs on GPU if available else CPU.

---

## 7. Repository layout ⛔

```
hindi_gec/
  README.md                     how to run everything, in order
  requirements.txt              pinned
  configs/config.yaml           paths, seeds, hyperparameters, type distribution
  data/raw/                     hiwiki .src/.trg, cached wiki pages   (git-ignored)
  data/processed/               bio_clean, hiwiked_bio, gold200 (jsonl)
  data/synthetic/               bio_synthetic (jsonl)
  models/                       saved checkpoints                     (git-ignored)
  src/hindi_gec/
      __init__.py
      preprocessing.py          (existing hindi_preprocessing.py, importable)
      corpus_builder.py         §2.2
      domain_filter.py          §2.3
      morphology.py             §3.1
      error_generator.py        §3.2
      edit_extractor.py         §5.1
      metrics.py                §5.2
      lm.py  rules_baseline.py  §4.1
      train_detector.py         §4.2
      train_corrector.py        §4.3
      inference.py              §4.4
  scripts/
      01_build_corpus.py … 08_evaluate.py     thin CLIs, one per phase
      run_all.ps1                            runs 01–08 in order on Windows
  app/streamlit_app.py
  tests/                        pytest; every module has one
  reports/                      college-format docx per experiment + final report
```

Conventions: every module starts with a docstring explaining **the linguistics and the
why**, not just the what; every public function has a docstring + type hints; every
script prints what it read, what it wrote, and counts; `sys.stdout.reconfigure("utf-8")`
at the top of every entry point; seeds fixed (`config.seed = 42`); no hard-coded paths.

---

## 8. Phase plan with definition of done

| # | Phase | Deliverable | Done when |
|---|---|---|---|
| 1 | Scaffold | repo layout, config, requirements, `preprocessing.py` moved in, tests pass | `pytest` green; `python -c "import hindi_gec"` works |
| 2 | Domain corpus | `bio_clean.jsonl` | ≥ 40 k sentences; 50 random ones read by Salil look like clean biography prose |
| 3 | Real test set | `hiwiked_bio.jsonl`, real-error type distribution table | ≥ 1 000 pairs; Salil verifies 200 → `gold200` |
| 4 | Morphology + generator | `morphology.py`, `error_generator.py`, `bio_synthetic.jsonl` | Quality gate §3.4 passed |
| 5 | Edit extractor + metrics | `edit_extractor.py`, `metrics.py` | Unit tests with hand-computed expected values pass; classifier agrees with gold200 tags ≥ 85 % |
| 6 | Baseline A | `rules_baseline.py`, `lm.py`, eval table | Numbers on all three test sets recorded |
| 7 | Detector B | checkpoint + eval | Token F1 on synthetic val ≥ 0.85 |
| 8 | Corrector C | two checkpoints + ablation table | Success criteria §5.3 checked (pass or documented miss) |
| 9 | Pipeline + app | `inference.py`, Streamlit | Demo runs end to end on a fresh sentence in < 3 s on GPU |
| 10 | Reports | experiment docx files + final report | College format; every number traceable to a script output |

Rough wall-clock on the 4060: corpus build 20–40 min (network), generation 10 min,
baseline 10 min, detector 15 min, corrector 2 × 30 min, eval 5 min.

---

## 9. Risk register

| Risk | Mitigation |
|---|---|
| Wikipedia category walk yields too few sentences | Add categories; drop to lead-section-only of *all* hi-wiki pages with an infobox of type "व्यक्ति" |
| HiWikEd biography subset < 1 000 | Lower threshold to ≥ 1; report filter precision honestly |
| Stanza Hindi POS/feats noisy | Generator uses the paradigm table first, Stanza only as fallback; agreement rules require nsubj confidence |
| OOM on 8 GB | batch 8 + grad-accum 4; gradient checkpointing; max_len 48 |
| IndicBART tokenizer/language-tag misuse | Round-trip test: encode→decode a clean sentence must be identity before training |
| Corrector hallucinates rewrites | Hallucination guard §4.3; detector gating §4.4 |
| Windows cp1252 console | `reconfigure("utf-8")` everywhere; write files with `encoding="utf-8"`; CSV with `utf-8-sig` |
| Synthetic ≠ real error distribution | §3.3 measurement step; ablation with real dev data |

---

## 10. What goes in the final report (college format)

Abstract · Introduction & motivation for narrowing · Related work (HiWikEd paper,
Hi-GEC/COLING 2025, GECToR, IndicBART) · Domain and error taxonomy · Dataset construction
(corpus, filter, gold200, synthetic generator with algorithm) · System design (both
diagrams, updated for the two-model design) · Experiments & results (tables §5.2) ·
Error analysis · Limitations (out-of-scope error types, Wikipedia register only) ·
Conclusion & future work · References.

Formatting: Times New Roman 12 body / 14 headings, justified, black only, no italics, no
headers/footers, bold keywords, single line spacing, code and output in bordered tables.
