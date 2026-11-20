# System A baseline — rules + 4-gram LM

Produced by `.venv\Scripts\python.exe scripts\16_baseline_A.py`. Margin 5.0 (tuned on dev_select, all-edits F0.5). Guard not applicable.

| set | pairs | view | TP | FP | FN | P | R | F0.5 |
|---|---|---|---|---|---|---|---|---|
| dev_select | 304 | all edits | 51 | 93 | 273 | 0.354 | 0.157 | 0.283 |
| dev_select | 304 | original 7 types | 36 | 93 | 147 | 0.279 | 0.197 | 0.258 |
| dev_select | 304 | extended (+PRON) | 51 | 93 | 176 | 0.354 | 0.225 | 0.318 |
| synthetic_val | 4178 | all edits | 2249 | 1071 | 1982 | 0.677 | 0.532 | 0.642 |
| synthetic_val | 4178 | original 7 types | 1770 | 1071 | 1710 | 0.623 | 0.509 | 0.596 |
| synthetic_val | 4178 | extended (+PRON) | 2249 | 1071 | 1980 | 0.677 | 0.532 | 0.642 |
| hiwiked_test | 1823 | all edits | 302 | 468 | 1644 | 0.392 | 0.155 | 0.300 |
| hiwiked_test | 1823 | original 7 types | 201 | 468 | 902 | 0.300 | 0.182 | 0.266 |
| hiwiked_test | 1823 | extended (+PRON) | 302 | 468 | 1095 | 0.392 | 0.216 | 0.337 |
