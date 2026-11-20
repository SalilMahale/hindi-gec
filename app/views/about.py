# -*- coding: utf-8 -*-
"""About / How it works: the viva cheat-sheet on screen.

Pipeline figure (app/assets/pipeline.png, drawn by scripts/20_pipeline_figure.py),
the error taxonomy with examples, dataset numbers from reports/app_stats.json,
limitations and references.
"""

from __future__ import annotations  # modern type-hint syntax

import streamlit as st  # web-page library

import ui  # shared colours, CSS, data
from hindi_gec import EXTENDED_TYPES  # taxonomy order (7 types + PRON)


def taxonomy_table() -> str:
    rows = "".join(
        f'<tr><td>{ui.tchip(t)}</td><td class="deva">{ui.esc(ui.TYPES[t]["hi"])}</td>'
        f'<td>{ui.esc(ui.TYPES[t]["en"])}</td><td>{ui.esc(ui.TYPES[t]["why_en"])}</td>'
        f'<td class="deva">{ui.esc(ui.TYPES[t]["ex"])}</td></tr>' for t in EXTENDED_TYPES)
    return ('<table class="grid"><thead><tr><th>code</th><th>हिंदी</th><th>error</th><th>rule</th>'
            f'<th>example (wrong → right)</th></tr></thead><tbody>{rows}</tbody></table>')


def render() -> None:
    s = ui.stats()
    st.title("How it works")
    st.caption("Hindi grammar error detection and correction for biographical prose (जीवनी).")

    st.subheader("Pipeline")
    st.image(str(ui.cfg().path("pipeline_figure")), use_container_width=True)
    st.html('<div class="card"><div class="label">At correction time</div>'
            '<ol style="margin:0;padding-left:20px;line-height:1.8">'
            '<li><b>Clean</b> the sentence exactly as the training data was cleaned (Unicode NFC, nukta, '
            'tokenisation; a missing danda is added).</li>'
            '<li><b>Correct</b> it with IndicBART, an encoder–decoder pre-trained on Indian languages and fine-tuned '
            'here on synthetic (wrong, right) pairs plus real Wikipedia corrections; beam search with 5 beams.</li>'
            '<li><b>Guard</b>: if the output changes more than max(3 words, 25 % of the sentence), it is treated as '
            'a hallucination and the input is kept.</li>'
            '<li><b>Explain</b>: input and output are aligned word by word; every changed span becomes an edit, and '
            'rules over Hindi verb paradigms, postpositions and spelling give it a type.</li></ol></div>')

    st.subheader("Error taxonomy")
    st.html(taxonomy_table())

    st.subheader("Data")
    c, y, r, t = s["corpus"], s["synthetic"], s["real"], s["training"]
    st.html('<div class="kpis">'
            + ui.tile(f"{c['sentences']:,}", f"clean biography sentences, {c['pages']:,} people (Hindi Wikipedia)")
            + ui.tile(f"{y['pairs']:,}", f"synthetic pairs: train {y['train']:,}, val {y['val']:,}, "
                                         f"{y['clean_share']:.0%} left error-free")
            + ui.tile(f"{r['dev_train']:,}", f"real dev_train pairs, each used {t['real_upsample']}× in training "
                                             f"({r['dev_train_dropped']} near-duplicates of evaluation pairs removed)")
            + ui.tile(f"{r['dev_select']:,}", "real dev_select pairs (held out, for choosing settings)")
            + ui.tile(f"{r['test']:,}", "real test pairs (HiWikEd-bio), scored once")
            + ui.tile(f"{t['pairs']:,}", "training pairs in all")
            + "</div>")
    st.html('<div class="note">Synthetic errors are made by breaking correct sentences on purpose (changing a '
            'verb\'s gender or number, swapping a postposition or a pronoun, dropping ने, changing tense, '
            'misspelling a word), so the right answer is always known. Of the real human-edited pairs, dev_train is '
            'added to training; any dev_train pair that is nearly identical (token-set Jaccard ≥ 0.8) to a '
            'dev_select or test pair was removed first, and any real pair whose sentence also appears in the '
            'training corpus was removed from the real data.</div>')

    st.subheader("Limitations")
    share = ui.out_of_scope_share()
    st.html('<ul class="lim">'
            f'<li>Only the 8 types are corrected. {share:.0%} of real Wikipedia edits are outside them (word choice, '
            'noun plurals and oblique forms, adjective agreement, punctuation).</li>'
            '<li>Trained mostly on synthetic errors (plus a few hundred real ones): real errors are subtler, so '
            'real-data recall stays modest; precision is kept higher on purpose (F0.5 weights precision about '
            'twice as much as recall).</li>'
            '<li>Biography domain: past-tense narration about people. Other genres (news, dialogue) are untested.</li>'
            '<li>One sentence at a time: agreement across sentences (a pronoun referring to an earlier name) '
            'is not seen.</li>'
            '<li>Honorific vs plural is decided from the sentence alone; without context, '
            '"वे … थे" can be either.</li>'
            '<li>Honorific-pronoun bias: the pronoun type (PRON) is trained only in the '
            'direction real Wikipedia editors correct, singular → honorific (उसने → उन्होंने, 88 % of real dev '
            'pronoun edits). The model therefore tends to make any singular pronoun honorific, which is wrong '
            'when the referent is not a respected person (a child, an animal, an organisation).</li></ul>')

    st.subheader("References")
    st.html('<ul class="lim">'
            '<li>Dabre et al. (2022). IndicBART: A Pre-trained Model for Indic Natural Language Generation. '
            'Findings of ACL 2022.</li>'
            '<li>Sonawane et al. (2020). Generating Inflectional Errors for Grammatical Error Correction in Hindi. '
            'AACL-IJCNLP 2020 Student Research Workshop (HiWikEd).</li>'
            '<li>Dahlmeier and Ng (2012). Better Evaluation for Grammatical Error Correction. NAACL 2012 '
            '(edit-level P / R / F0.5).</li>'
            '<li>Bryant, Felice and Briscoe (2017). Automatic Annotation and Evaluation of Error Types for '
            'Grammatical Error Correction. ACL 2017 (typed edits, ERRANT).</li>'
            '<li>Kunchukuttan (2020). The IndicNLP Library (normalisation and tokenisation).</li></ul>')
    with st.expander("Error types (colour legend)"):
        st.html(ui.legend())
