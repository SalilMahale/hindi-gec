# -*- coding: utf-8 -*-
"""Make `src/` importable so pytest works before `pip install -e .`."""
# pytest automatically runs conftest.py before any test file.

import sys  # sys.path = list of folders Python searches for imports
from pathlib import Path  # file paths

SRC = Path(__file__).resolve().parents[1] / "src"  # .../hindi_gec/src
if str(SRC) not in sys.path:  # not already on the search path
    sys.path.insert(0, str(SRC))  # add it at the front so "import hindi_gec" works
