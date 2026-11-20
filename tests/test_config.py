# -*- coding: utf-8 -*-
"""Tests that the configuration honours the contracts marked ⛔ in the plan.

The point of these tests is to make a silent mistake impossible: a mistyped
error code, a type distribution that no longer sums to 1, or a split ratio
that quietly shrinks the precious real-error test set would all corrupt the
headline numbers without raising an exception anywhere else.
"""
# How pytest works: every function named test_* is run; it passes if no "assert" fails.

import pytest  # testing framework
import yaml  # to read the raw YAML and deliberately break it

import hindi_gec  # our package
from hindi_gec import ERROR_TYPES, NO_ERROR, OTHER
from hindi_gec.config import DEFAULT_CONFIG_PATH, Config, load_config, set_seed, validate_config


def test_package_imports_and_exposes_taxonomy():
    assert hindi_gec.__version__  # version string exists
    assert len(ERROR_TYPES) == 7  # exactly 7 error types
    assert ERROR_TYPES[0] == "AGR-GEN"  # order is fixed
    assert NO_ERROR == "NONE" and OTHER == "OTHER"
    assert NO_ERROR not in ERROR_TYPES and OTHER not in ERROR_TYPES  # these are not real error types


def test_config_file_exists_and_loads():
    assert DEFAULT_CONFIG_PATH.exists()  # configs/config.yaml is there
    cfg = load_config()  # loads and validates without error
    assert cfg.seed == 42  # seed value


def test_error_types_match_the_taxonomy():
    cfg = load_config()
    assert tuple(cfg["error_types"]) == ERROR_TYPES  # YAML list == code tuple


def test_type_distribution_sums_to_one_and_matches_decision():
    cfg = load_config()
    dist = cfg["generator"]["type_distribution"]
    assert abs(sum(dist.values()) - 1.0) < 1e-9  # shares add up to 1
    # PLAN §3.3 escape hatch (2026-10-08, option a) for the 7 types, then PRON added at its real dev
    # share 0.1784 with ERG 0.05 and SPELL 0.1785 kept and the other five scaled (Salil, 2026-10-22).
    assert dist == {"AGR-NUM": 0.1776, "AGR-GEN": 0.0639, "POSTP": 0.1654, "SPELL": 0.1785,
                    "AGR-HON": 0.0964, "AUX": 0.0898, "ERG": 0.05, "PRON": 0.1784}  # exact agreed values


def test_real_error_split_keeps_test_larger_than_dev():
    cfg = load_config()
    split = cfg["splits"]["hiwiked_bio"]
    assert split["dev"] == 0.40 and split["test"] == 0.60  # 40 / 60 split


def test_paths_resolve_to_absolute_inside_repo():
    cfg = load_config()
    p = cfg.path("bio_clean")
    assert p.is_absolute()  # full path, not relative
    assert p.name == "bio_clean.jsonl"
    assert "data" in p.parts and "processed" in p.parts  # .../data/processed/bio_clean.jsonl


def test_attribute_access_reaches_nested_values():
    cfg = load_config()
    assert cfg.detector.model_name == "google/muril-base-cased"  # cfg.a.b style access works
    assert cfg.corrector.model_name == "ai4bharat/IndicBART"
    assert cfg.corrector.lang_tag == "<2hi>"


def test_validate_rejects_a_broken_distribution(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))  # raw config dict
    raw["generator"]["type_distribution"]["SPELL"] = 0.50   # now sums to 1.38
    with pytest.raises(AssertionError, match="sum to 1.0"):  # validation must fail with this message
        validate_config(Config(raw))


def test_validate_rejects_an_unknown_error_code():
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["error_types"][0] = "AGR-GENDER"  # misspelled error code
    with pytest.raises(AssertionError, match="taxonomy"):  # must be rejected
        validate_config(Config(raw))


def test_set_seed_is_deterministic():
    import random

    set_seed(42)
    a = [random.random() for _ in range(3)]  # three random numbers
    set_seed(42)  # same seed again
    b = [random.random() for _ in range(3)]
    assert a == b  # same seed -> same numbers
