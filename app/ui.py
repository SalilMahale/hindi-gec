# -*- coding: utf-8 -*-
"""Shared pieces of the web app: error-type colours, CSS, cached model and data, HTML renderers.

Every page imports from here, so an error type has one colour and one name
everywhere (legend, inline diff, chips, charts). Every number shown comes from
a file under reports/ (app_stats.json is written by scripts/09_app_stats.py).

HTML is built as one-line strings and rendered with st.html, which keeps
classes, inline styles and data-* attributes but runs no scripts.
"""

from __future__ import annotations  # modern type-hint syntax

import html  # html.escape() makes user text safe inside HTML
import json  # read the report files
from pathlib import Path  # file paths
from typing import Any, Dict, List, Optional, Sequence  # type hints

import streamlit as st  # web-page library

from hindi_gec import OTHER  # code for edits outside the 8 types
from hindi_gec.config import load_config  # configs/config.yaml

APP_DIR = Path(__file__).resolve().parent  # .../hindi_gec/app

# One colour per type, used everywhere. The 7 original types take the first 7 slots of the
# dataviz reference categorical palette in taxonomy order (an order that keeps
# neighbouring colours distinguishable for colour-blind viewers); OTHER is grey.
# AUX is dark brown instead of slot 6 green: green sat 15.6 OKLab dE from AGR-HON and 3.2 from
# AGR-NUM under protanopia; brown is >= 25.8 from every type normally and >= 17.4 under all CVD types.
# PRON is berry #b00080: slot 8 red sat 7.1 from AGR-NUM. A grid search (chroma >= 0.10, OKLab L
# 0.40-0.75, >= 3:1 on white) gave #b00080: >= 21.2 from every type and the text ink normally, >= 13.0 under CVD.
TYPES: Dict[str, Dict[str, str]] = {
    "AGR-GEN": {"c": "#2a78d6", "hi": "लिंग सहमति", "en": "gender agreement",
                "why_hi": "क्रिया का लिंग कर्ता से मेल खाना चाहिए।",
                "why_en": "The verb must match the gender of its subject.",
                "ex": "सीता बाजार गया → गई"},
    "AGR-NUM": {"c": "#eb6834", "hi": "वचन सहमति", "en": "number agreement",
                "why_hi": "क्रिया का वचन कर्ता (ने के बाद कर्म) से मेल खाना चाहिए।",
                "why_en": "The verb must match the number of its subject (of the object after ने).",
                "ex": "लड़के जाता है → जाते हैं"},
    "AGR-HON": {"c": "#1baf7a", "hi": "आदरसूचक सहमति", "en": "honorific agreement",
                "why_hi": "आदरणीय व्यक्ति के लिए क्रिया बहुवचन में आती है।",
                "why_en": "A respected person takes a plural (honorific) verb.",
                "ex": "पिताजी आया → आए"},
    "POSTP":   {"c": "#eda100", "hi": "परसर्ग", "en": "postposition",
                "why_hi": "यहाँ दूसरा परसर्ग आता है।",
                "why_en": "A different postposition (में, पर, से, को, तक, का/की/के) belongs here.",
                "ex": "ननिहाल पर रहते हुए → में"},
    "ERG":     {"c": "#e87ba4", "hi": "ने (कर्ता)", "en": "ergative ने",
                "why_hi": "भूतकाल की सकर्मक क्रिया के साथ कर्ता 'ने' लेता है।",
                "why_en": "With a past-tense transitive verb the subject takes ने.",
                "ex": "वे पद ग्रहण किया → उन्होंने"},
    "AUX":     {"c": "#5c4033", "hi": "सहायक क्रिया / काल", "en": "auxiliary / tense",
                "why_hi": "सहायक क्रिया का काल वाक्य के काल से मेल खाना चाहिए।",
                "why_en": "The auxiliary (है / था) must match the tense of the sentence.",
                "ex": "1984 में निर्वाचित हुए हैं → थे"},
    "SPELL":   {"c": "#6250d6", "hi": "वर्तनी", "en": "spelling",
                "why_hi": "मानक वर्तनी: मात्रा, अनुस्वार/चंद्रबिंदु या नुक्ता।",
                "why_en": "Standard spelling: vowel length, nasal sign or nukta.",
                "ex": "शीक्षा → शिक्षा"},
    "PRON":    {"c": "#b00080", "hi": "सर्वनाम", "en": "pronoun",
                "why_hi": "आदरणीय व्यक्ति के लिए बहुवचन सर्वनाम; संबंध-सूचक सर्वनाम अगले शब्द से मेल खाता है।",
                "why_en": "One respected person takes the plural pronoun; a possessive pronoun agrees with the next noun.",
                "ex": "उसने कई पुस्तकें लिखीं → उन्होंने"},
    OTHER:     {"c": "#8a8f9c", "hi": "अन्य", "en": "other",
                "why_hi": "यह बदलाव आठ समर्थित प्रकारों में नहीं आता।",
                "why_en": "A change outside the eight supported types.",
                "ex": "कैरियर → करियर"},
}


def esc(s: Any) -> str:  # shorthand for HTML escaping
    return html.escape(str(s), quote=True)


# ----------------------------------------------------------------------
# Setup, cached data and model
# ----------------------------------------------------------------------
def inject_css() -> None:
    """Page-wide stylesheet + one CSS variable per type (--c = that type's colour)."""
    css = (APP_DIR / "style.css").read_text(encoding="utf-8")
    css += "".join(f".t-{t} {{ --c: {v['c']}; }}" for t, v in TYPES.items())  # .t-AGR-GEN { --c: #2a78d6; } ...
    st.html(f"<style>{css}</style>")


@st.cache_resource(show_spinner=False)  # one config object for the whole server
def cfg() -> Any:
    return load_config()


@st.cache_resource(show_spinner=False)  # the model loads once and is shared by every page and click
def corrector() -> Any:
    from hindi_gec.inference import Corrector  # imported here: torch is slow to import
    return Corrector(cfg())  # paths.model


def model_loaded() -> bool:  # has the model been loaded yet in this session?
    return st.session_state.get("_model_ready", False)


def get_corrector() -> Any:
    """The cached corrector; shows a spinner the first time (loading takes a few seconds)."""
    if not model_loaded():
        with st.spinner(f"Loading the {model_label()} …"):
            c = corrector()
        st.session_state["_model_ready"] = True
        return c
    return corrector()


def model_label() -> str:  # "IndicBART corrector": the one name the app uses for the model
    return cfg()["app"]["model_label"]


@st.cache_data(show_spinner=False)
def read_json(path: str) -> Optional[Any]:  # None if the file does not exist (yet)
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def stats() -> Dict[str, Any]:
    """reports/app_stats.json; stops the page with instructions if it is missing."""
    s = read_json(str(cfg().path("app_stats")))
    if s is None:
        st.error(r"reports\app_stats.json is missing. Run: .venv\Scripts\python.exe scripts\09_app_stats.py")
        st.stop()
    return s


def examples() -> List[Dict[str, str]]:  # one example chip per error type (scripts/10_app_examples.py)
    return read_json(str(cfg().path("app_examples"))) or []


def out_of_scope_share() -> float:  # share of real HiWikEd edits outside the 8 types (OTHER)
    r = stats()["real"]
    return r["edit_types"][OTHER] / r["edits"]


# ----------------------------------------------------------------------
# HTML renderers
# ----------------------------------------------------------------------
def tchip(t: str, suggestion: bool = False, text: Optional[str] = None) -> str:
    """Small rounded chip: coloured dot + type code."""
    return f'<span class="tchip t-{t}{" sugg" if suggestion else ""}">{esc(text or t)}</span>'


def legend() -> str:
    return '<div class="legend">' + "".join(
        tchip(t, text=f"{t} · {v['hi']}") for t, v in TYPES.items()) + "</div>"


def explanation(e: Dict[str, Any]) -> tuple[str, str]:
    """(Hindi, English) reason for an edit: its own explanation if it has one, else the type's rule."""
    ex = e.get("explanation")
    if isinstance(ex, dict):
        return ex.get("hi", ""), ex.get("en", "")
    info = TYPES.get(e["type"], TYPES[OTHER])
    return info["why_hi"], info["why_en"]


def is_suggestion(e: Dict[str, Any], threshold: float) -> bool:  # low confidence -> shown as a suggestion
    return e.get("confidence", 1.0) < threshold


def diff_html(source_tokens: Sequence[str], edits: Sequence[Dict[str, Any]], threshold: float,
              small: bool = False) -> str:
    """The corrected sentence with each edit shown inline: wrong (struck) → right.

    Unchanged words are the same on both sides, so they are copied from the
    source; each edit's span says which source words it replaces.
    """
    out, i = [], 0  # HTML pieces; current source position
    for e in sorted(edits, key=lambda e: e["span"][0]):
        a, b = e["span"]  # source words a..b-1 are replaced by e["tgt"]
        out += [esc(w) for w in source_tokens[i:a]]  # unchanged words before the edit
        hi, en = explanation(e)
        info = TYPES.get(e["type"], TYPES[OTHER])
        tip = f"{e['type']} · {info['hi']} · {info['en']}\n{hi}\n{en}"
        if "confidence" in e:
            tip += f"\nconfidence {e['confidence']:.2f}"
        wrong = f"<del>{esc(e['src'])}</del>" if e["src"] else ""
        right = f"<ins>{esc(e['tgt'])}</ins>" if e["tgt"] else ""
        arrow = '<span class="arrow">→</span>' if wrong and right else ""
        sugg = " sugg" if is_suggestion(e, threshold) else ""
        out.append(f'<span class="edit t-{e["type"]}{sugg}" data-tip="{esc(tip)}">{wrong}{arrow}{right}</span>')
        i = b
    out += [esc(w) for w in source_tokens[i:]]  # unchanged words after the last edit
    return f'<div class="sent{" small" if small else ""}">{" ".join(out)}</div>'


def plain_html(text: str, small: bool = False) -> str:
    return f'<div class="sent{" small" if small else ""}">{esc(text)}</div>'


def edits_table(edits: Sequence[Dict[str, Any]], threshold: float) -> str:
    """One row per edit: wrong → correct, type chip, confidence bar, explanation."""
    rows = []
    for e in edits:
        hi, en = explanation(e)
        sugg = is_suggestion(e, threshold)
        conf = e.get("confidence")
        bar = ("—" if conf is None else
               f'<div class="conf{" low" if sugg else ""}"><div class="bar"><span style="width:{conf * 100:.0f}%"></span></div>'
               f'<span class="v">{conf:.2f}</span></div>')
        kind = tchip(e["type"], sugg) + (' <span class="note">suggestion</span>' if sugg else "")
        rows.append(f'<tr class="t-{e["type"]}"><td class="w">{esc(e["src"] or "∅")}</td>'
                    f'<td class="r">{esc(e["tgt"] or "∅")}</td><td>{kind}</td><td>{bar}</td>'
                    f'<td class="why"><span class="hi">{esc(hi)}</span><br>{esc(en)}</td></tr>')
    return ('<table class="edits"><thead><tr><th>Wrong</th><th>Correct</th><th>Type</th>'
            '<th>Confidence</th><th>Why</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")


def no_error_box() -> str:
    """Calm green message that also says what the checker does not look at."""
    share = out_of_scope_share()
    return ('<div class="okbox"><div class="t">No errors found in the 8 supported types</div>'
            '<div class="hi">इन आठ प्रकारों में कोई त्रुटि नहीं मिली।</div>'
            '<div class="scope">Checked: gender, number and honorific agreement, postpositions, ergative ने, '
            'auxiliary / tense, spelling, pronouns (honorific, possessive endings). Not checked: word choice, '
            'noun plural and oblique forms, adjective agreement, punctuation and style. Corrections outside the '
            f'8 types are {share:.0%} of the real Wikipedia edits in HiWikEd-bio.</div></div>')


def guard_box(raw_output: str) -> str:
    return ('<div class="warnbox">The model proposed a large rewrite, so the hallucination guard kept the input '
            f'unchanged. Proposed:<div class="sent">{esc(raw_output)}</div></div>')


def tile(n: str, k: str) -> str:  # big number + small caption
    return f'<div class="tile"><div class="n">{esc(n)}</div><div class="k">{esc(k)}</div></div>'


def card(label: str, body: str, extra_class: str = "") -> str:
    return f'<div class="card {extra_class}"><div class="label">{esc(label)}</div>{body}</div>'
