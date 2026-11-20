# -*- coding: utf-8 -*-
"""Model insights: dashboard built only from saved result files.

Nothing here runs the model. Every number traces to a script:
reports/results_final.json (scripts/19_phase_e.py, computed from the saved
system outputs in reports/predictions/) holds the scores, confusion matrix and
speed; reports/app_stats.json (scripts/09_app_stats.py) holds the data and
training numbers.

Charts are Altair (bundled with Streamlit, renders offline). The corrector is
the accent indigo and System A (the rule + 4-gram baseline) a neutral grey,
so neither borrows an error-type colour; the per-type charts use the type
colours, and the confusion heatmap one blue ramp (light = small share).
"""

from __future__ import annotations  # modern type-hint syntax

from typing import Optional  # type hints

import altair as alt  # charts
import pandas as pd  # tables for the charts
import streamlit as st  # web-page library

import ui  # shared colours, CSS, data
from hindi_gec import EXTENDED_TYPES, OTHER  # type order

CORR, BASE = "IndicBART corrector", "System A"  # keys in results_final.json
SYSTEM_COLOURS = {CORR: "#3730A3", BASE: "#A3A8B6"}  # accent indigo, neutral grey
SEQ = ["#FFFFFF", "#CDE2FB", "#6DA7EC", "#2A78D6", "#104281"]  # one-hue blue ramp for the confusion heatmap
FONT = "Segoe UI"
VIEWS = ("all edits", "original 7 types", "extended (+PRON)")
SET_NAMES = {"hiwiked_test": "Real test", "dev_select": "Real dev_select", "synthetic_val": "Synthetic val"}


def style(chart: alt.Chart) -> alt.Chart:  # recessive axes and grid, system font
    return (chart.configure_view(stroke=None)
            .configure_axis(labelFont=FONT, titleFont=FONT, labelColor="#4B5163", titleColor="#4B5163",
                            gridColor="#ECEEF3", domainColor="#D5D8E0", tickColor="#D5D8E0", labelFontSize=12,
                            titleFontSize=12, titleFontWeight="normal")
            .configure_legend(labelFont=FONT, titleFont=FONT, labelColor="#1F2433", titleColor="#4B5163",
                              orient="top", direction="horizontal", titleOrient="left")
            .properties(background="#FFFFFF", padding={"left": 8, "right": 12, "top": 8, "bottom": 4}))


def final_results() -> Optional[dict]:  # reports/results_final.json (19_phase_e.py)
    return ui.read_json(str(ui.cfg().path("results_final")))


def results_table(rf: dict) -> str:
    """Every set: corrector and System A (P / R, F0.5 in three views, GLEU, exact match) + the do-nothing GLEU."""
    rows = []
    for key, r in rf["sets"].items():
        n = 3
        for k, s in enumerate((CORR, BASE)):
            x = r[s]
            a = x["all edits"]
            best = lambda v: x[v]["f05"] >= max(r[o][v]["f05"] for o in (CORR, BASE))  # noqa: E731
            cells = "".join(f'<td class="num{" best" if best(v) else ""}">{x[v]["f05"]:.3f}</td>' for v in VIEWS)
            g_best = x["gleu"] >= max(r[CORR]["gleu"], r[BASE]["gleu"], r["gleu_source"])
            head = f'<td rowspan="{n}"><b>{SET_NAMES[key]}</b><br><span class="note">{r["pairs"]:,} pairs</span></td>' if k == 0 else ""
            rows.append(f'<tr>{head}<td>{s}</td><td class="num">{a["p"]:.3f}</td><td class="num">{a["r"]:.3f}</td>'
                        f'{cells}<td class="num{" best" if g_best else ""}">{x["gleu"]:.4f}</td>'
                        f'<td class="num">{x["exact_match"]:.1%}</td></tr>')
        rows.append(f'<tr><td class="note">do nothing (unchanged input)</td><td class="num">—</td><td class="num">—</td>'
                    f'<td class="num">—</td><td class="num">—</td><td class="num">—</td>'
                    f'<td class="num">{r["gleu_source"]:.4f}</td><td class="num">—</td></tr>')
    return ('<table class="grid"><thead><tr><th>set</th><th>system</th><th class="num">P</th><th class="num">R</th>'
            '<th class="num">F0.5 all edits</th><th class="num">F0.5 7 types</th><th class="num">F0.5 8 types</th>'
            '<th class="num">GLEU</th><th class="num">exact match</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")


def systems_chart(rf: dict) -> tuple[alt.Chart, pd.DataFrame]:
    """F0.5 (all edits) per set, corrector vs System A."""
    df = pd.DataFrame([{"set": SET_NAMES[k], "system": s, "F05": r[s]["all edits"]["f05"],
                        "P": r[s]["all edits"]["p"], "R": r[s]["all edits"]["r"]}
                       for k, r in rf["sets"].items() for s in (CORR, BASE)])
    order = [SET_NAMES[k] for k in rf["sets"]]
    ch = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("set:N", sort=order, title=None, scale=alt.Scale(paddingInner=0.35), axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("system:N", sort=[CORR, BASE], scale=alt.Scale(paddingInner=0.1)),
        y=alt.Y("F05:Q", scale=alt.Scale(domain=[0, 1]), title="F0.5 (all edits)"),
        color=alt.Color("system:N", scale=alt.Scale(domain=list(SYSTEM_COLOURS), range=list(SYSTEM_COLOURS.values())),
                        title="system"),
        tooltip=["set", "system", alt.Tooltip("F05:Q", format=".3f", title="F0.5"),
                 alt.Tooltip("P:Q", format=".3f"), alt.Tooltip("R:Q", format=".3f")],
    ).properties(height=280)
    return style(ch), df


def per_type_chart(rf: dict) -> tuple[alt.Chart, pd.DataFrame]:
    # Field names must not contain "." : Vega-Lite reads "F0.5" as field "F0", sub-field "5".
    pt = rf["per_type_test"]
    order = [t for t in EXTENDED_TYPES if t in pt]
    df = pd.DataFrame([{"type": t, "name": ui.TYPES[t]["en"], "F05": pt[t]["f05"], "TP": pt[t]["tp"],
                        "FP": pt[t]["fp"], "FN": pt[t]["fn"]} for t in order])
    ch = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("type:N", sort=order, title=None, scale=alt.Scale(paddingInner=0.3), axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("F05:Q", scale=alt.Scale(domain=[0, 1]), title="F0.5 (real test, 8-type classifier)"),
        color=alt.Color("type:N", scale=alt.Scale(domain=order, range=[ui.TYPES[t]["c"] for t in order]), legend=None),
        tooltip=["type", "name", alt.Tooltip("F05:Q", format=".3f", title="F0.5"), "TP", "FP", "FN"],
    ).properties(height=300)
    return style(ch), df


def real_types_chart(s: dict) -> tuple[alt.Chart, pd.DataFrame]:
    r = s["real"]
    order = list(EXTENDED_TYPES) + [OTHER]
    df = pd.DataFrame([{"type": t, "edits": r["edit_types"][t], "share": r["edit_types"][t] / r["edits"],
                        "name": ui.TYPES[t]["en"]} for t in order])
    base = alt.Chart(df).encode(y=alt.Y("type:N", sort=order, title=None))
    bars = base.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height={"band": 0.7}).encode(
        x=alt.X("edits:Q", title="real edits in HiWikEd-bio (dev + test)"),
        color=alt.Color("type:N", scale=alt.Scale(domain=order, range=[ui.TYPES[t]["c"] for t in order]), legend=None),
        tooltip=["type", "name", "edits", alt.Tooltip("share:Q", format=".1%")])
    labels = base.mark_text(align="left", dx=6, font=FONT, fontSize=12, color="#4B5163").encode(
        x="edits:Q", text=alt.Text("share:Q", format=".1%"))
    return style((bars + labels).properties(height=300)), df


def loss_chart(s: dict) -> tuple[alt.Chart, pd.DataFrame]:
    df = pd.DataFrame([{"step": st_, "loss": lo} for st_, lo in s["training"]["losses"]])
    hover = alt.selection_point(fields=["step"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(df).encode(x=alt.X("step:Q", title="training step",
                                        scale=alt.Scale(domain=[0, int(df["step"].max())], nice=False)))
    line = base.mark_line(strokeWidth=2, color=SYSTEM_COLOURS[CORR]).encode(
        y=alt.Y("loss:Q", scale=alt.Scale(zero=False), title="training loss (label-smoothed)"))
    points = base.mark_point(size=60, filled=True, color=SYSTEM_COLOURS[CORR]).encode(
        y="loss:Q", opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=["step", alt.Tooltip("loss:Q", format=".4f")]).add_params(hover)
    rule = base.mark_rule(color="#C9CDD8").encode(opacity=alt.condition(hover, alt.value(1), alt.value(0))
                                                  ).transform_filter(hover)
    return style((line + rule + points).properties(height=280)), df


def confusion_chart(c: dict) -> tuple[alt.Chart, pd.DataFrame]:
    """Heatmap: colour = share of the row (true type), count printed in each non-empty cell."""
    rows = []
    for i, t in enumerate(c["true"]):
        total = sum(c["matrix"][i]) or 1
        for j, p in enumerate(c["predicted"]):
            rows.append({"true": t, "predicted": p, "edits": c["matrix"][i][j], "share": c["matrix"][i][j] / total})
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        x=alt.X("predicted:N", sort=c["predicted"], title="type of the system edit at the same place",
                axis=alt.Axis(labelAngle=-30, orient="top")),
        y=alt.Y("true:N", sort=c["true"], title="true type (generator)"))
    rect = base.mark_rect(cornerRadius=2, stroke="#FFFFFF", strokeWidth=2).encode(
        color=alt.Color("share:Q", scale=alt.Scale(domain=[0, 1], range=SEQ), title="share of row",
                        legend=alt.Legend(format=".0%")),
        tooltip=["true", "predicted", "edits", alt.Tooltip("share:Q", format=".1%")])
    text = base.mark_text(font=FONT, fontSize=11).encode(
        text=alt.condition("datum.edits > 0", "edits:Q", alt.value("")),
        color=alt.condition("datum.share > 0.5", alt.value("#FFFFFF"), alt.value("#1F2433")))
    return style((rect + text).properties(height=34 * len(c["true"]))), df


def render() -> None:
    s = ui.stats()
    rf = final_results()
    st.title("Model insights")
    st.caption("From saved result files only (reports/results_final.json, reports/app_stats.json). "
               "Edit-level scores, hallucination guard on, β = 0.5.")
    if rf is None:
        st.error(r"reports\results_final.json is missing. Run: .venv\Scripts\python.exe scripts\19_phase_e.py")
        st.stop()
    t = rf["sets"]["hiwiked_test"]
    c, a = t[CORR], t[BASE]
    st.html('<div class="kpis">'
            + ui.tile(f"{c['all edits']['f05']:.3f}", f"F0.5, real test, all edits (System A {a['all edits']['f05']:.3f})")
            + ui.tile(f"{c['extended (+PRON)']['f05']:.3f}", "F0.5, real test, the 8 types")
            + ui.tile(f"{c['all edits']['p']:.3f}", "precision, real test (all edits)")
            + ui.tile(f"{c['gleu']:.3f}", f"GLEU, real test (do nothing {t['gleu_source']:.3f})")
            + "</div>")

    st.subheader("Results: IndicBART corrector vs System A")
    st.html(results_table(rf))
    st.html('<div class="note">Real test was decoded once per system, after every choice was made on dev_select '
            'and synthetic val. "7 types" leaves out real edits outside the original taxonomy (OTHER) while still '
            'counting every wrong change as a false positive; "8 types" also counts pronoun edits, which the '
            'corrector is trained on. GLEU compares whole sentences with the reference; the unchanged input is the '
            '"do nothing" score a system must beat. System A is the rule + 4-gram language-model baseline.</div>')
    ch, df = systems_chart(rf)
    st.altair_chart(ch, use_container_width=True, theme=None)
    with st.expander("Data table"):
        st.dataframe(df, hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("F0.5 per error type, real test")
        ch, df = per_type_chart(rf)
        st.altair_chart(ch, use_container_width=True, theme=None)
        with st.expander("Data table"):
            st.dataframe(df, hide_index=True, use_container_width=True)
        erg = rf["per_type_test"].get("ERG", {"fp": 0})
        st.html(f'<div class="note">ERG scores 0 because HiWikEd-bio has {s["real"]["edit_types"]["ERG"]} real ने '
                f'edits, so each of the {erg["fp"]} ने changes on test is a false positive. Many real spelling '
                'edits are variant spellings (गये → गए) that the generator deliberately does not teach.</div>')
    with c2:
        st.subheader("What real editors fix")
        ch, df = real_types_chart(s)
        st.altair_chart(ch, use_container_width=True, theme=None)
        with st.expander("Data table"):
            st.dataframe(df, hide_index=True, use_container_width=True)
        st.html(f'<div class="note">Edit types in {s["real"]["pairs"]:,} real HiWikEd-bio pairs, typed by the same '
                '8-type edit classifier used for scoring. OTHER (word choice, noun forms, punctuation …) is outside '
                'what the corrector is trained on.</div>')

    st.subheader("Training loss")
    ch, df = loss_chart(s)
    st.altair_chart(ch, use_container_width=True, theme=None)
    tr = s["training"]
    st.html(f'<div class="note">{tr["pairs"]:,} training pairs ({tr["synthetic_pairs"]:,} synthetic + '
            f'{tr["real_pairs"]:,} real dev_train pairs × {tr["real_upsample"]}), {tr["epochs"]} epochs, batch 32, '
            f'bf16: {tr["steps"]:,} steps in {tr["minutes"]:.0f} min, final validation loss {tr["val_loss"]:.4f}. '
            'Training loss includes label smoothing 0.1, so it levels off well above zero.</div>')
    with st.expander("Data table"):
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.subheader("Type confusion matrix")
    cm = rf["confusion"]
    ch, df = confusion_chart(cm)
    st.altair_chart(ch, use_container_width=True, theme=None)
    st.html('<div class="note">Synthetic val. Row = true type (the generator\'s label, right by construction); '
            'column = type of the corrector\'s edit at the same place; colour = share of the row. When it found an '
            f'error, it gave it the right type {cm["type_accuracy_when_found"]:.1%} of the time '
            f'({cm["found"]:,} found of {cm["reference_edits"]:,} reference edits). Most AGR-HON → AGR-NUM cells '
            'are the edit classifier, not the model: without a page title it cannot always tell an honorific '
            'subject from a plural one.</div>')
    with st.expander("Data table"):
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.subheader("Speed")
    lat = rf["latency"]
    st.html('<div class="kpis">'
            + ui.tile(f"{lat['batch_1']['mean_ms']:.0f} ms", f"one sentence, mean (p95 {lat['batch_1']['p95_ms']:.0f} ms)")
            + ui.tile(f"{lat['batched']['mean_ms_per_sentence']:.0f} ms",
                      f"per sentence in a batch of {lat['batch']} (p95 {lat['batched']['p95_ms_per_sentence']:.0f} ms)")
            + "</div>"
            f'<div class="note">Full app pipeline on the {lat["device"]} (beam search, per-edit confidence, '
            f'explanations, alternatives), {lat["sentences"]} real dev sentences, each setting timed '
            f'{lat["repeats"]} times after 5 warm-up calls.</div>')
