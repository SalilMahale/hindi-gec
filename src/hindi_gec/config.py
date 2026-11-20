# -*- coding: utf-8 -*-
"""Single source of truth for paths, seeds and hyperparameters.

Every script reads `configs/config.yaml` through this module instead of
hard-coding a path or a learning rate. Two reasons this matters for a GEC
project specifically:

1. **Reproducibility of the numbers in the report.** A result table is only
   defensible if the exact hyperparameters that produced it can be pointed at.
   One YAML file, versioned alongside the code, is that pointer.
2. **The type distribution is a linguistic claim, not a magic number.** PLAN
   §3.3 fixes the share of each error type the generator must emit; this
   loader validates that the distribution covers exactly the seven taxonomy
   codes plus PRON and sums to 1, so a typo cannot silently skew the training data.

Paths in the YAML are relative to the repository root and are resolved to
absolute paths here, so scripts behave the same whether they are launched
from the root or from `scripts/`.
"""

from __future__ import annotations  # allows modern type-hint syntax such as "str | Path"

import random  # Python's built-in random number generator (we seed it for reproducibility)
from pathlib import Path  # file-path objects
from typing import Any, Dict, Optional  # type hints: Any = any type, Dict = dictionary, Optional = may be None

import yaml  # PyYAML: reads .yaml config files into Python dicts/lists

from . import ERROR_TYPES, EXTENDED_TYPES  # the 7 error types (+ PRON) from hindi_gec/__init__.py

#: Repository root = two levels above this file (src/hindi_gec/config.py).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]  # .../hindi_gec (the project folder)
DEFAULT_CONFIG_PATH: Path = REPO_ROOT / "configs" / "config.yaml"  # where the config file lives


class Config(dict):  # a normal dictionary with one extra trick (attribute access)
    """A dict with attribute access, so `cfg.detector.lr` reads naturally."""

    def __getattr__(self, name: str) -> Any:  # called when you write cfg.something and "something" isn't a normal attribute
        try:
            value = self[name]  # look the name up as a dictionary key
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc  # missing key -> behave like a missing attribute
        return Config(value) if isinstance(value, dict) else value  # wrap nested dicts too, so cfg.a.b.c works

    def path(self, key: str) -> Path:
        """Return `paths.<key>` as an absolute Path, creating parents on demand."""
        raw = self["paths"][key] if "paths" in self else self[key]  # read the relative path string from the "paths" section
        resolved = (REPO_ROOT / raw).resolve()  # join it with the project folder to get a full absolute path
        return resolved


def load_config(path: Optional[str | Path] = None) -> Config:
    """Load, validate and return the project configuration."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH  # use the given file, else configs/config.yaml
    with open(cfg_path, encoding="utf-8") as f:  # open as UTF-8 because the file contains Hindi text
        raw: Dict[str, Any] = yaml.safe_load(f)  # parse YAML -> nested Python dict (safe_load can't run code)
    cfg = Config(raw)  # wrap in our Config class for cfg.x.y access
    validate_config(cfg)  # check it is correct before anyone uses it
    return cfg


def validate_config(cfg: Config) -> None:
    """Fail loudly and early on a malformed configuration."""
    assert "seed" in cfg, "config.seed is required (PLAN §7: seeds fixed)"  # a random seed must be present

    declared = tuple(cfg["error_types"])  # error types written in the YAML
    assert declared == ERROR_TYPES, (  # must be exactly the 7 types, in the same order
        f"config.error_types must match the ⛔ taxonomy {ERROR_TYPES}, got {declared}"
    )

    dist = cfg["generator"]["type_distribution"]  # what share of synthetic errors each type should get
    assert set(dist) == set(EXTENDED_TYPES), (  # every type (7 + PRON) must be listed, and nothing else
        f"type_distribution must cover exactly {EXTENDED_TYPES}, got {sorted(dist)}"
    )
    total = sum(dist.values())  # add up all the shares
    assert abs(total - 1.0) < 1e-9, f"type_distribution must sum to 1.0, got {total}"  # shares are probabilities -> must total 1 (tiny tolerance for float rounding)

    splits = cfg["splits"]["hiwiked_bio"]  # real-data split ratios
    assert abs(splits["dev"] + splits["test"] - 1.0) < 1e-9, "dev + test must be 1.0"  # dev + test must cover all data
    syn = cfg["splits"]["synthetic"]  # synthetic-data split ratios
    assert abs(syn["train"] + syn["val"] - 1.0) < 1e-9, "train + val must be 1.0"  # train + validation must cover all data


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch if they are installed.

    Imported lazily so that the CPU-only phases of this project (corpus
    building, generation, evaluation) never pay the cost of importing torch.
    """
    random.seed(seed)  # same seed -> same "random" numbers every run -> reproducible results
    try:
        import numpy as np  # imported here (not at top) so it's optional

        np.random.seed(seed)  # seed NumPy's random generator too
    except ImportError:  # pragma: no cover
        pass  # NumPy not installed -> nothing to seed
    try:
        import torch  # PyTorch, the deep-learning library

        torch.manual_seed(seed)  # seed PyTorch on the CPU (weight init, dropout, shuffling)
        if torch.cuda.is_available():  # if an NVIDIA GPU is usable
            torch.cuda.manual_seed_all(seed)  # seed every GPU as well
    except ImportError:  # pragma: no cover
        pass  # torch not installed -> skip
