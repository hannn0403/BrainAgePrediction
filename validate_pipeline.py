#!/usr/bin/env python3
"""
Brain Age Prediction Pipeline Validator

Validates dataset integrity, config consistency, and code health
for the Brain Age Prediction project (HCP, Cam-CAN, IXI).

Usage:
    python validate_pipeline.py
"""

import ast
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
SHAP_DIR = PROJECT_ROOT / "SHAP"
CONFIG_PATH = SHAP_DIR / "config.yaml"

EXPECTED_N_FEATURES = 153
REQUIRED_META_COLS = ["Age", "Subject"]

# Expected training-set sizes (hardcoded in SHAP scripts)
EXPECTED_TRAIN_COUNTS = {
    "hcp_train.csv": 890,
    "cc_train.csv": 500,
    "ixi_train.csv": 453,
}

DATASET_FILES = [
    "hcp_train.csv",
    "hcp_test.csv",
    "cc_train.csv",
    "cc_test.csv",
    "ixi_train.csv",
    "ixi_test.csv",
]

TRAIN_TEST_PAIRS = [
    ("hcp_train.csv", "hcp_test.csv"),
    ("cc_train.csv", "cc_test.csv"),
    ("ixi_train.csv", "ixi_test.csv"),
]

AGE_MIN = 0
AGE_MAX = 120

# Required config keys expressed as dot-paths
REQUIRED_CONFIG_KEYS = [
    "paths.datapath",
    "paths.results",
    "paths.genpath",
    "data.parcellation",
    "preproc.combat",
    "preproc.regress",
    "preproc.pca",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ValidationResult:
    """Accumulates pass / warn / fail results."""

    def __init__(self):
        self.results: list[tuple[str, str, str]] = []  # (status, category, message)

    def ok(self, category: str, msg: str):
        self.results.append(("PASS", category, msg))

    def warn(self, category: str, msg: str):
        self.results.append(("WARN", category, msg))

    def fail(self, category: str, msg: str):
        self.results.append(("FAIL", category, msg))

    def summary(self):
        passes = sum(1 for s, _, _ in self.results if s == "PASS")
        warns = sum(1 for s, _, _ in self.results if s == "WARN")
        fails = sum(1 for s, _, _ in self.results if s == "FAIL")
        return passes, warns, fails

    def print_report(self):
        print("\n" + "=" * 70)
        print("  BRAIN AGE PREDICTION -- PIPELINE VALIDATION REPORT")
        print("=" * 70)

        current_cat = None
        for status, cat, msg in self.results:
            if cat != current_cat:
                print(f"\n--- {cat} ---")
                current_cat = cat
            tag = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[status]
            print(f"  {tag} {msg}")

        passes, warns, fails = self.summary()
        print("\n" + "-" * 70)
        print(f"  Total: {passes} passed, {warns} warnings, {fails} failed")
        if fails == 0 and warns == 0:
            print("  All checks passed.")
        elif fails == 0:
            print("  No failures, but there are warnings to review.")
        else:
            print("  Some checks FAILED -- review the output above.")
        print("-" * 70 + "\n")


def resolve_nested_key(d: dict, dotpath: str):
    """Traverse a dict with a dot-separated key path. Returns (found, value)."""
    keys = dotpath.split(".")
    current = d
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            return False, None
        current = current[k]
    return True, current


# ---------------------------------------------------------------------------
# 1. Dataset Integrity
# ---------------------------------------------------------------------------

def validate_datasets(vr: ValidationResult):
    cat = "Dataset Integrity"

    # 1a. File existence
    for fname in DATASET_FILES:
        fpath = DATASET_DIR / fname
        if fpath.is_file():
            vr.ok(cat, f"{fname} exists")
        else:
            vr.fail(cat, f"{fname} is MISSING from {DATASET_DIR}")

    # Load dataframes for further checks (skip files that don't exist or are
    # not valid CSVs)
    dfs: dict[str, pd.DataFrame] = {}
    for fname in DATASET_FILES:
        fpath = DATASET_DIR / fname
        if not fpath.is_file():
            continue
        try:
            df = pd.read_csv(fpath)
            if df.empty or len(df.columns) < 2:
                vr.warn(cat, f"{fname} appears to be a placeholder / sample file (too few columns or rows)")
                continue
            dfs[fname] = df
        except Exception as exc:
            vr.fail(cat, f"{fname} could not be parsed as CSV: {exc}")

    if not dfs:
        vr.warn(cat, "No valid CSV datasets could be loaded; skipping remaining dataset checks")
        return

    # 1b. Required columns
    for fname, df in dfs.items():
        missing_meta = [c for c in REQUIRED_META_COLS if c not in df.columns]
        if missing_meta:
            vr.fail(cat, f"{fname} missing required columns: {missing_meta}")
        else:
            vr.ok(cat, f"{fname} has required meta columns (Age, Subject)")

        feature_cols = [c for c in df.columns if c not in REQUIRED_META_COLS]
        n_feat = len(feature_cols)
        if n_feat == EXPECTED_N_FEATURES:
            vr.ok(cat, f"{fname} has {n_feat} feature columns (expected {EXPECTED_N_FEATURES})")
        else:
            vr.warn(cat, f"{fname} has {n_feat} feature columns (expected {EXPECTED_N_FEATURES})")

    # 1c. Missing values in key columns
    for fname, df in dfs.items():
        for col in REQUIRED_META_COLS:
            if col not in df.columns:
                continue
            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                vr.ok(cat, f"{fname}: no missing values in '{col}'")
            else:
                vr.fail(cat, f"{fname}: {n_missing} missing values in '{col}'")

    # 1d. Age range
    for fname, df in dfs.items():
        if "Age" not in df.columns:
            continue
        ages = pd.to_numeric(df["Age"], errors="coerce")
        age_min = ages.min()
        age_max = ages.max()
        if np.isnan(age_min) or np.isnan(age_max):
            vr.warn(cat, f"{fname}: could not determine numeric Age range")
        elif age_min < AGE_MIN:
            vr.fail(cat, f"{fname}: minimum Age={age_min} is below {AGE_MIN}")
        elif age_max > AGE_MAX:
            vr.fail(cat, f"{fname}: maximum Age={age_max} exceeds {AGE_MAX}")
        else:
            vr.ok(cat, f"{fname}: Age range [{age_min:.1f}, {age_max:.1f}] is within bounds")

    # 1e. Train/test subject overlap
    for train_f, test_f in TRAIN_TEST_PAIRS:
        if train_f not in dfs or test_f not in dfs:
            continue
        if "Subject" not in dfs[train_f].columns or "Subject" not in dfs[test_f].columns:
            continue
        train_ids = set(dfs[train_f]["Subject"])
        test_ids = set(dfs[test_f]["Subject"])
        overlap = train_ids & test_ids
        if overlap:
            vr.fail(cat, f"{train_f} / {test_f}: {len(overlap)} overlapping Subject IDs detected")
        else:
            vr.ok(cat, f"{train_f} / {test_f}: no Subject ID overlap between train and test")

    # 1f. Row counts vs expected
    for fname, expected in EXPECTED_TRAIN_COUNTS.items():
        if fname not in dfs:
            continue
        actual = len(dfs[fname])
        if actual == expected:
            vr.ok(cat, f"{fname}: row count {actual} matches expected {expected}")
        else:
            vr.warn(cat, f"{fname}: row count {actual} differs from expected {expected}")


# ---------------------------------------------------------------------------
# 2. Config Validation
# ---------------------------------------------------------------------------

def validate_config(vr: ValidationResult):
    cat = "Config Validation"

    if not CONFIG_PATH.is_file():
        vr.fail(cat, f"config.yaml not found at {CONFIG_PATH}")
        return

    vr.ok(cat, "config.yaml exists")

    # Parse YAML
    try:
        with open(CONFIG_PATH, "r") as fh:
            cfg = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        vr.fail(cat, f"config.yaml is not valid YAML: {exc}")
        return

    if not isinstance(cfg, dict):
        vr.fail(cat, "config.yaml did not parse into a dictionary")
        return

    vr.ok(cat, "config.yaml is valid YAML")

    # Required keys
    for dotpath in REQUIRED_CONFIG_KEYS:
        found, value = resolve_nested_key(cfg, dotpath)
        if found:
            vr.ok(cat, f"Key '{dotpath}' present (value: {value})")
        else:
            vr.fail(cat, f"Key '{dotpath}' is MISSING")

    # Check if configured directory paths exist on this machine
    path_keys = ["paths.datapath", "paths.results", "paths.genpath"]
    for dotpath in path_keys:
        found, value = resolve_nested_key(cfg, dotpath)
        if not found or value is None:
            continue
        p = Path(str(value))
        if p.is_dir():
            vr.ok(cat, f"Path '{dotpath}' -> {value} exists on disk")
        else:
            vr.warn(cat, f"Path '{dotpath}' -> {value} does NOT exist on this machine")


# ---------------------------------------------------------------------------
# 3. Code Consistency
# ---------------------------------------------------------------------------

def validate_code(vr: ValidationResult):
    cat = "Code Consistency"

    # 3a. Syntax-check all .py files in SHAP/
    if not SHAP_DIR.is_dir():
        vr.fail(cat, f"SHAP directory not found at {SHAP_DIR}")
        return

    py_files = sorted(SHAP_DIR.glob("*.py"))
    if not py_files:
        vr.warn(cat, "No .py files found in SHAP/")
    else:
        for pyf in py_files:
            try:
                source = pyf.read_text(encoding="utf-8", errors="replace")
                ast.parse(source, filename=str(pyf))
                vr.ok(cat, f"{pyf.name} -- syntax OK")
            except SyntaxError as exc:
                vr.fail(cat, f"{pyf.name} -- syntax error: {exc}")

    # 3b. functions/ directory inside SHAP/
    funcs_dir = SHAP_DIR / "functions"
    if funcs_dir.is_dir():
        vr.ok(cat, "SHAP/functions/ directory exists")
    else:
        vr.warn(cat, "SHAP/functions/ directory is MISSING")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    vr = ValidationResult()

    print("Running pipeline validation ...\n")

    validate_datasets(vr)
    validate_config(vr)
    validate_code(vr)

    vr.print_report()

    _, _, fails = vr.summary()
    sys.exit(1 if fails > 0 else 0)


if __name__ == "__main__":
    main()
