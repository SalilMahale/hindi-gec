# -*- coding: utf-8 -*-
"""Hindi GEC web app (multi-page): Checker, Paragraph mode, Model insights, About.

Run (offline-safe):
    .venv\\Scripts\\python.exe -m streamlit run app\\streamlit_app.py

This file only sets up the page and the navigation; each page is a render()
function in app/views/. Shared colours, CSS and the cached model are in app/ui.py.
Nothing is fetched from the internet: system fonts, Streamlit's bundled charts
and local model files. Pages have no icons on purpose: Streamlit loads Material
page icons from fonts.gstatic.com, which fails offline.
"""

from __future__ import annotations  # modern type-hint syntax

import os  # environment variables
import sys  # module search path
from pathlib import Path  # file paths

os.environ["HF_HUB_OFFLINE"] = "1"  # Hugging Face: never go to the internet
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # transformers: local model files only
APP = Path(__file__).resolve().parent  # .../hindi_gec/app
sys.path.insert(0, str(APP.parent / "src"))  # "import hindi_gec"
sys.path.insert(0, str(APP))  # "import ui", "from views import ..."

import streamlit as st  # noqa: E402

import ui  # noqa: E402
from views import about, checker, insights, paragraph  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="Hindi GEC · जीवनी", layout="wide", initial_sidebar_state="expanded")
    ui.inject_css()  # one stylesheet for every page
    nav = st.navigation([
        st.Page(checker.render, title="Checker", default=True),
        st.Page(paragraph.render, title="Paragraph mode", url_path="paragraph"),
        st.Page(insights.render, title="Model insights", url_path="insights"),
        st.Page(about.render, title="About / How it works", url_path="about"),
    ])
    with st.sidebar:
        st.caption(f"Model: {ui.model_label()} · runs offline on this laptop")
    nav.run()  # draw the selected page


if __name__ == "__main__":
    main()
