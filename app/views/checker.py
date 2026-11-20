# -*- coding: utf-8 -*-
"""Checker (home page): one sentence in, corrected sentence with an inline diff out.

Layout: header band with three stat chips (read from reports/app_stats.json),
then input on the left (text area, one example chip per error type, Check
button) and the result card on the right, then the edits table.

The last result is kept in st.session_state, so it survives Streamlit's
re-run on every click (e.g. when the copy box is used).
"""

from __future__ import annotations  # modern type-hint syntax

import streamlit as st  # web-page library

import ui  # shared colours, CSS, model, renderers

SAMPLE_KEY = "chk_text"  # session-state key of the text area


def header() -> None:
    s = ui.stats()
    h = s["headline"]
    chips = (f'<span class="stat"><b>{s["training"]["pairs"]:,}</b>training pairs</span>'
             f'<span class="stat"><b>{len(s["error_types"])}</b>error types</span>'
             f'<span class="stat"><b>{h["test_f05"]:.3f}</b>F0.5 on real Wikipedia edits '
             f'(precision {h["test_precision"]:.2f})</span>')
    st.html('<div class="band"><h1>Hindi Grammar Correction</h1>'
            '<div class="hi">हिंदी व्याकरण सुधार · जीवनी (biographical prose)</div>'
            '<div class="sub">Finds and fixes agreement, postposition, ergative, tense, spelling and pronoun '
            'errors in Hindi biography sentences, and says what kind of error each fix is.</div>'
            f'<div class="chips">{chips}</div></div>')


def pick_example() -> None:  # chip clicked -> put its sentence in the editor and check it
    label = st.session_state.get("chk_chip")
    for ex in ui.examples():
        if chip_label(ex) == label:
            st.session_state[SAMPLE_KEY] = ex["source"]
            st.session_state["chk_run"] = True


def chip_label(ex: dict) -> str:
    return f"{ex['label']} · {ex['type']}"


def run_check(text: str) -> None:
    corr = ui.get_corrector()
    st.session_state["chk_result"] = corr.correct(text)
    st.session_state["chk_device"] = corr.device


def render() -> None:
    header()
    threshold = ui.cfg()["app"]["suggestion_below"]
    exs = ui.examples()
    if SAMPLE_KEY not in st.session_state:
        st.session_state[SAMPLE_KEY] = exs[0]["source"] if exs else ""

    left, right = st.columns([1, 1.05], gap="large")
    with left:
        st.text_area("वाक्य लिखें · Enter a Hindi sentence", key=SAMPLE_KEY, height=150)
        st.pills("Examples, one per error type", [chip_label(e) for e in exs], key="chk_chip",
                 on_change=pick_example, selection_mode="single")
        clicked = st.button("Check · जाँचें", type="primary", use_container_width=True)
    text = st.session_state[SAMPLE_KEY]
    if (clicked or st.session_state.pop("chk_run", False)) and text.strip():
        with right:
            run_check(text)

    r = st.session_state.get("chk_result")
    with right:
        if r is None:
            st.html(ui.card("Result", '<div class="placeholder">Type or pick a sentence and press Check. '
                                      'Corrections appear here, coloured by error type; hover one to see why.</div>'))
        else:
            src = r.input.split()
            if r.edits:
                body = ui.diff_html(src, r.edits, threshold)
                n = len(r.edits)
                st.html(ui.card(f"Corrected · {n} edit{'s' if n > 1 else ''}", body))
            else:
                st.html(ui.card("Result", ui.plain_html(r.corrected) + ui.no_error_box()))
            if r.guarded:
                st.html(ui.guard_box(r.raw_output))
            st.html('<div class="card-label">Copy-ready text</div>')
            st.code(r.corrected, language=None, wrap_lines=True)  # Streamlit's copy icon sits on this box
            st.html(f'<div class="meta">Copy with the icon on the box above · '
                    f'{r.latency_ms:.0f} ms on {st.session_state.get("chk_device", "")} · '
                    f'sentence confidence {r.confidence:.2f}</div>')

    if r is not None and r.edits:
        st.html(ui.card("Edits", ui.edits_table(r.edits, threshold)))
        st.html(f'<div class="note">Confidence is the mean probability the model gave to the words of each ' 
                f'correction. Below {threshold:.2f} an edit is shown dashed, as a suggestion rather than a firm ' 
                'correction.</div>')
    if r is not None and r.alternatives:
        rows = "".join(
            f'<div class="alt">{ui.diff_html(r.input.split(), a["edits"], 0.0, small=True) if a["edits"] else ui.plain_html(a["text"], small=True)}'
            f'<div class="meta">{"" if a["edits"] else "leave unchanged · "}beam score {a["score"]:.2f}</div></div>'
            for a in r.alternatives)
        st.html(ui.card("Other possible corrections", rows))
    with st.expander("Error types (colour legend)"):
        st.html(ui.legend())
