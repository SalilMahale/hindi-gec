# -*- coding: utf-8 -*-
"""scoring: guard and the three views on two hand-built Hindi pairs (counts worked out by hand)."""

from __future__ import annotations  # modern type-hint syntax

from hindi_gec.scoring import exact_match, guard_outputs, views  # functions under test


def row(src: str, tgt: str) -> dict:
    return {"source_tokens": src.split(), "target_tokens": tgt.split()}


ROWS = [row("उसने कई पुस्तकें लिखीं ।", "उन्होंने कई पुस्तकें लिखीं ।"),       # reference: one pronoun edit
        row("उनका जन्म दिल्ली में हुई था ।", "उनका जन्म दिल्ली में हुआ था ।")]  # reference: one AGR-GEN edit


def test_three_views_count_the_pronoun_edit_differently() -> None:
    finals = [r["target_tokens"] for r in ROWS]  # a perfect system
    v = views(ROWS, finals, 0.5)
    assert (v["all edits"]["tp"], v["all edits"]["fn"]) == (2, 0)
    # 7-type classifier: the pronoun edit is OTHER, so it is left out of this view -> only AGR-GEN remains
    assert (v["original 7 types"]["tp"], v["original 7 types"]["fp"], v["original 7 types"]["fn"]) == (1, 0, 0)
    # 8-type classifier: the pronoun edit is PRON and counts
    assert v["extended (+PRON)"]["tp"] == 2 and "PRON" in v["extended (+PRON)"]["by_type"]
    assert exact_match(ROWS, finals) == 1.0


def test_guard_replaces_a_rewrite_with_the_source() -> None:
    rewrite = "यह वाक्य पूरी तरह बदल गया है ।".split()  # 7 of 5 words changed > max(3, 25 %)
    finals, n = guard_outputs(ROWS[:1], [rewrite], 3, 0.25)
    assert n == 1 and finals == [ROWS[0]["source_tokens"]]
    v = views(ROWS[:1], finals, 0.5)
    assert (v["all edits"]["tp"], v["all edits"]["fp"], v["all edits"]["fn"]) == (0, 0, 1)
