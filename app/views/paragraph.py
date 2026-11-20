# -*- coding: utf-8 -*-
"""Paragraph mode: many sentences in, each corrected, all in one GPU batch.

The paragraph is split after every । ? ! (inference.split_sentences) because the
model was trained on single sentences. The summary bar counts sentences and
edits per type; each sentence then gets its own accordion row.
"""

from __future__ import annotations  # modern type-hint syntax

from collections import Counter  # edits per type

import streamlit as st  # web-page library

import ui  # shared colours, CSS, model, renderers
from hindi_gec import EXTENDED_TYPES, OTHER  # type order for the summary (8 types + OTHER)
from hindi_gec.inference import split_sentences  # paragraph -> sentences

KEY = "para_text"


def default_text() -> str:  # the example sentences joined into one paragraph
    return " ".join(e["source"] for e in ui.examples()[:4])


def summary(results: list) -> str:
    counts = Counter(e["type"] for r in results for e in r.edits)
    n_err = sum(1 for r in results if r.edits)
    total = sum(counts.values())
    tiles = (ui.tile(str(len(results)), "sentences checked") + ui.tile(str(n_err), "sentences with errors")
             + ui.tile(str(total), "edits") + ui.tile(f"{results[0].latency_ms:.0f} ms" if results else "—",
                                                      "per sentence (one batch)"))
    order = [t for t in list(EXTENDED_TYPES) + [OTHER] if counts[t]]
    stack = "".join(f'<span class="t-{t}" style="flex:{counts[t]}" title="{t}: {counts[t]}"></span>' for t in order)
    chips = " ".join(ui.tchip(t, text=f"{t} · {counts[t]}") for t in order)
    bar = f'<div class="stack">{stack}</div><div class="legend">{chips}</div>' if total else ""
    return f'<div class="sumrow">{tiles}</div>{bar}'


def render() -> None:
    st.title("Paragraph mode")
    st.caption("Paste several sentences. They are split after । ? ! and corrected together in one GPU batch.")
    threshold = ui.cfg()["app"]["suggestion_below"]
    if KEY not in st.session_state:
        st.session_state[KEY] = default_text()
    st.text_area("अनुच्छेद · Paragraph", key=KEY, height=200)
    if st.button("Check paragraph · जाँचें", type="primary") and st.session_state[KEY].strip():
        sents = split_sentences(st.session_state[KEY])
        corr = ui.get_corrector()
        with st.spinner(f"Correcting {len(sents)} sentences …"):
            st.session_state["para_results"] = corr.correct_batch(sents, ui.cfg()["app"]["paragraph_batch_size"])
    results = st.session_state.get("para_results")
    if not results:
        return
    st.html(summary(results))
    for k, r in enumerate(results, 1):
        n = len(r.edits)
        status = f"{n} edit{'s' if n != 1 else ''}" if n else "no error found"
        if r.guarded:
            status += " · guard kept input"
        with st.expander(f"{k}.  {r.input}   —   {status}", expanded=bool(n)):
            if n:
                st.html(ui.diff_html(r.input.split(), r.edits, threshold, small=True))
                st.html(ui.edits_table(r.edits, threshold))
            else:
                st.html(ui.plain_html(r.corrected, small=True))
            if r.guarded:
                st.html(ui.guard_box(r.raw_output))
    st.subheader("Corrected paragraph")
    st.html('<div class="card-label">Copy-ready text</div>')
    st.code(" ".join(r.corrected for r in results), language=None, wrap_lines=True)
