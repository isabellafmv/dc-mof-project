"""
Categorical encoding for topology and metal_node features.

Two methods, call with the same interface:

  method="label"   → integers; use for XGBoost and RF
  method="target"  → mean CO2 uptake per category; use for MLP

Usage
-----
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src.encoding import encode_categoricals

train_enc, val_enc, test_enc = encode_categoricals(
    df_train, df_val, df_test, method="label"
)
# → adds topology_encoded and metal_node_encoded columns (int for label, float for target)

# For RF, pass unknown_int=n_classes so unseen categories don't become negative:
train_enc, val_enc, test_enc = encode_categoricals(
    df_train, df_val, df_test, method="label", unknown_int="n_classes"
)

Notes
-----
- Fitting always uses df_train only; val/test are only transformed.
- Unseen categories in val/test (common in topology- and metal-splits) are mapped to
  unknown_int for label encoding, or the global training mean for target encoding.
- Pass None for val or test if you only need a subset.
- Original DataFrames are never modified; copies are returned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CAT_COLS = ["topology", "metal_node"]
ENC_COLS = ["topology_encoded", "metal_node_encoded"]

TARGET_COLS = [
    "co2_mol_kg_0.01bar",
    "co2_mol_kg_0.05bar",
    "co2_mol_kg_0.1bar",
    "co2_mol_kg_0.5bar",
    "co2_mol_kg_2.5bar",
]


def encode_categoricals(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame | None = None,
    df_test: pd.DataFrame | None = None,
    *,
    method: str = "label",
    unknown_int: int | str = -1,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Add topology_encoded and metal_node_encoded to each split.

    Parameters
    ----------
    df_train, df_val, df_test:
        DataFrames from data/splits/. Pass None to skip val or test.
    method:
        "label"  — label-encoded integers (XGBoost, RF)
        "target" — mean CO2 uptake per category (MLP)
    unknown_int:
        Only used for method="label". Integer assigned to categories not seen
        in training. Use -1 for XGBoost (handles missing natively). Use the
        string "n_classes" to assign n_unique_train_classes per column, which
        keeps values non-negative (safer for RF and some sklearn pipelines).

    Returns
    -------
    (df_train_enc, df_val_enc, df_test_enc) — 3-tuple, Nones preserved.
    """
    if method == "label":
        return _label_encode(df_train, df_val, df_test, unknown_int)
    if method == "target":
        return _target_encode(df_train, df_val, df_test)
    raise ValueError(f"method must be 'label' or 'target', got {method!r}")


# ── label encoding ────────────────────────────────────────────────────────────

def _label_encode(df_train, df_val, df_test, unknown_int):
    # Build str→int mappings from training data (sorted for determinism)
    mappings: dict[str, dict[str, int]] = {}
    for cat in CAT_COLS:
        classes = sorted(df_train[cat].dropna().unique())
        mappings[cat] = {v: i for i, v in enumerate(classes)}

    results = []
    for df in (df_train, df_val, df_test):
        if df is None:
            results.append(None)
            continue
        out = df.copy()
        for cat, enc in zip(CAT_COLS, ENC_COLS):
            unk = (
                len(mappings[cat]) if unknown_int == "n_classes" else unknown_int
            )
            out[enc] = out[cat].map(mappings[cat]).fillna(unk).astype(int)
        results.append(out)
    return tuple(results)


# ── target encoding ───────────────────────────────────────────────────────────

def _target_encode(df_train, df_val, df_test):
    # Scalar per category: mean of per-sample means across all 5 pressure targets.
    # Fitted on training set only; unseen categories → global training mean.
    train_target_mean = df_train[TARGET_COLS].mean(axis=1)
    global_mean = float(train_target_mean.mean())

    mappings: dict[str, pd.Series] = {}
    for cat in CAT_COLS:
        mappings[cat] = train_target_mean.groupby(df_train[cat]).mean()

    results = []
    for df in (df_train, df_val, df_test):
        if df is None:
            results.append(None)
            continue
        out = df.copy()
        for cat, enc in zip(CAT_COLS, ENC_COLS):
            out[enc] = out[cat].map(mappings[cat]).fillna(global_mean)
        results.append(out)
    return tuple(results)
