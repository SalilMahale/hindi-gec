# Demo day

Everything runs **offline** on this laptop. The demo uses the one model, the
**IndicBART corrector** in `models\corrector` (tokenizer included); no internet
is needed.

## 1. Launch (PowerShell, from the repo root)

```powershell
cd D:\College\Sem7\NLP\NLP_mini_project\hindi_gec
.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Then open **http://localhost:8501** in the browser. The first *Check · जाँचें*
loads the model (a few seconds); after that one sentence takes about 0.14 s on
the GPU (mean 138 ms, p95 199 ms; `reports\results_final.md`).

Use `.venv\Scripts\python.exe` exactly as written — do not rely on `Activate.ps1`.

Pages: **Checker** (one sentence, inline coloured fixes, explanation and
confidence per edit, other possible corrections), **Paragraph mode**,
**Model insights** (all scores from saved result files), **About / How it works**
(pipeline figure, taxonomy, data, limitations).

## 2. Backup if Streamlit fails — terminal demo

```powershell
.venv\Scripts\python.exe scripts\demo_cli.py --examples         # the 8 example chips
.venv\Scripts\python.exe scripts\demo_cli.py                    # type your own, empty line quits
.venv\Scripts\python.exe scripts\demo_cli.py "उसकी माँ एक शिक्षिका था।"
```

## 3. The eight example chips (each corrected exactly; `scripts\10_app_examples.py`)

| chip | input (wrong) | corrected | type |
|---|---|---|---|
| लिंग | उनके बचपन में ही उनका परिवार कोलकाता **चली आई** । | … **चला आया** । | AGR-GEN |
| वचन | सिंह ने लंदन स्कूल ऑफ़ इकोनॉमिक्स में अध्ययन **किए** । | … **किया** । | AGR-NUM |
| आदर | वे भारतीय संस्कृति की अन्तरराष्ट्रीय अकादमी के भी निदेशक **है** । | … **हैं** । | AGR-HON |
| परसर्ग | अपने जीवन काल **पर** उन्होंने दो पुस्तकें भी लिखीं । | … काल **में** … | POSTP |
| ने | **वे** 9 जून 1964 को भारत के प्रधानमंत्री का पद भार ग्रहण किया । | **उन्होंने** … | ERG |
| काल | वे वर्ष 1984 में 8वीं लोकसभा में उत्तरी गोवा से निर्वाचित हुए **हैं** । | … **थे** । | AUX |
| वर्तनी | इन्होंने उपन्यास , निबंध तथा **हीन्दी** साहित्य का अनुशीलन आदि ग्रंथ भी लिखे । | … **हिन्दी** … | SPELL |
| सर्वनाम | वे प्रोफेसर सत्येंद्रनाथ बोस को **अपनी** पूज्य गुरु मानते थे । | … **अपना** … | PRON |

The 5 correct sentences in `reports\demo_sentence_bank.txt` are all left unchanged.

Fresh sentences (not in any dataset), re-tested with the model:

| input | output |
|---|---|
| मेरी दादी का जन्म 1940 में एक छोटे गाँव में हुआ था। | unchanged (correct) |
| महात्मा गांधी ने दक्षिण अफ्रीका में वकालत की और वे 1915 में भारत लौट **आया**। | … लौट **आए** । (AGR-HON) |
| **उसकी** माँ एक शिक्षिका **था**। | **उनकी** माँ एक शिक्षिका **थीं** । (PRON + AGR-GEN) |

The last one shows the honorific-pronoun bias (section 4).

## 4. What to say

- **Scope:** Hindi *biographical* prose, 8 error types — gender / number / honorific
  agreement, postpositions (में से को पर तक, का/की/के), ergative ने,
  auxiliary/tense, spelling, pronouns (honorific उसने → उन्होंने, possessive
  endings). Word order, word choice, punctuation, meaning errors and code-mixing
  are out of scope.
- **Training data:** 76 571 synthetic (wrong, right) pairs from 34 692 clean
  biography sentences, plus 856 real Wikipedia corrections (HiWikEd dev_train,
  each used 5×) = 80 851 pairs; 55 real pairs nearly identical to an evaluation
  pair were removed first.
- **If a sentence isn't corrected:** it is probably outside the 8 types or
  outside biography style. The model is deliberately conservative — a wrong
  "correction" is worse than a missed error, which is why the metric is F0.5
  (precision weighted).
- **Honorific-pronoun bias:** pronouns are trained only in the direction real
  editors correct (singular → honorific), so the model tends to make any
  singular pronoun honorific, which is wrong for a child, an animal or an
  organisation.
- **If the yellow guard message appears:** the model proposed rewriting too much,
  so the hallucination guard kept the input unchanged (rejected if more than
  max(3, 25 % of the words) would change).

## 5. Numbers (`reports\results_final.md`, `scripts\19_phase_e.py`, from saved outputs)

| set | system | P | R | F0.5 | GLEU |
|---|---|---|---|---|---|
| real HiWikEd-bio **test** (1 823 pairs, scored once) | IndicBART corrector | 0.627 | 0.325 | **0.529** | 0.7880 |
| | System A (rules + 4-gram LM) | 0.392 | 0.155 | 0.300 | 0.7304 |
| | do nothing | — | — | — | 0.7095 |
| real dev_select (304 pairs) | IndicBART corrector | 0.644 | 0.346 | 0.549 | 0.7863 |
| | System A | 0.354 | 0.157 | 0.283 | 0.7216 |
| synthetic val (4 178 pairs) | IndicBART corrector | 0.850 | 0.798 | 0.839 | 0.9428 |
| | System A | 0.677 | 0.532 | 0.642 | 0.8652 |

On real test, counting only the original 7 types: F0.5 0.505; the 8 types: 0.576.

Honest caveats: real-data precision is under-estimated because each HiWikEd
reference fixes only one Wikipedia edit and often leaves other errors in place.
Synthetic scores are high because the test errors come from the same generator
as the training errors.

## 6. Pipeline in one line

clean Wikipedia biographies (34 692 sentences) → rule-based error generator
(80 749 pairs, labels correct by construction) + 856 real corrections →
IndicBART fine-tuned (4 epochs, 10 108 steps, bf16, 49 min on the RTX 4060) →
beam search → hallucination guard → edit extractor + rule-based type classifier
→ coloured edits with explanations.
