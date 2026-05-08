"""
Shared data-loading utility for all three team members.

Usage
-----
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("..").resolve()))   # from notebooks/
from src.data_utils import load_experiment

X_train, X_test, y_train, y_test = load_experiment(
    split_strategy="random",       # "random" | "topology" | "metal"
    feature_set="combined_decorr", # any of the 7 named sets (see below)
)

Feature sets (defined in data/merged/feature_selection.json)
--------------------------------------------------------------
  baseline        2  topology_encoded + metal_node_encoded (categorical)
  geo_decorr      5  geometric features, post variance + correlation filter
  geo_all         6  geometric features, post variance filter only (+ lcd)
  rac_decorr     45  RAC features, post variance + correlation filter
  rac_all        81  RAC features, post variance filter only
  combined_decorr 50 geo_decorr + rac_decorr
  combined_all   87  geo_all + rac_all

Returns
-------
X_train, X_test : float64 arrays of shape (n, n_features)  — NOT standardised
y_train, y_test : float64 arrays of shape (n, 5)
                  columns → co2_mol_kg at 0.01 / 0.05 / 0.1 / 0.5 / 2.5 bar

There is no separate validation set — tune hyperparameters via cross-validation
on the training set, then evaluate the final model on X_test / y_test.

Standardise inside your model script, not here, because tree models don't
need it and MLP does.  Example:

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

Baseline encoding
-----------------
Default encoding is label encoding (integers, works for XGBoost and RF).
For MLP, pass encoding="target":

    load_experiment("random", "baseline", encoding="target")

See src/encoding.py for details on how unknown categories are handled.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.encoding import encode_categoricals

# ── paths ─────────────────────────────────────────────────────────────────────

# Resolve project root as two levels up from this file (src/data_utils.py)
_ROOT = Path(__file__).resolve().parent.parent
_SPLITS_DIR = _ROOT / "data" / "splits"
_CONFIG_PATH = _ROOT / "data" / "merged" / "feature_selection.json"

# hmof_merged.parquet holds ALL features (102) + metadata + targets.
# The split_*_merged.parquet files only carry the 50 decorrelated features,
# so we use them only for their name lists and always pull feature values
# from the master file.  This way geo_all / rac_all / combined_all work too.
_MASTER_PATH = _ROOT / "data" / "merged" / "hmof_merged.parquet"

# ── constants ─────────────────────────────────────────────────────────────────

TARGET_COLS = [
    "co2_mol_kg_0.01bar",
    "co2_mol_kg_0.05bar",
    "co2_mol_kg_0.1bar",
    "co2_mol_kg_0.5bar",
    "co2_mol_kg_2.5bar",
]

VALID_SPLITS = {"random", "topology", "metal"}


# ── public API ────────────────────────────────────────────────────────────────

def load_experiment(
    split_strategy: str,
    feature_set: str,
    *,
    encoding: str = "label",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one (split × feature-set) experiment as four numpy arrays.

    Parameters
    ----------
    split_strategy : "random" | "topology" | "metal"
    feature_set    : one of the 7 named sets in feature_selection.json
    encoding       : "label" (default, XGBoost / RF) or "target" (MLP)
                     Only used when feature_set == "baseline".

    Returns
    -------
    X_train, X_test : float64 (n, n_features)
    y_train, y_test : float64 (n, 5)  — all five CO2 pressures
    """
    if split_strategy not in VALID_SPLITS:
        raise ValueError(
            f"split_strategy must be one of {sorted(VALID_SPLITS)}, "
            f"got {split_strategy!r}"
        )

    config = json.loads(_CONFIG_PATH.read_text())
    feature_sets = config["feature_sets"]

    if feature_set not in feature_sets:
        raise ValueError(
            f"feature_set must be one of {sorted(feature_sets)}, "
            f"got {feature_set!r}"
        )

    is_baseline = feature_set == "baseline"
    feat_cols = feature_sets[feature_set]

    # Columns to load from the master parquet.
    # Baseline: need the raw categorical strings for encoding, not the
    # encoded names (those don't exist in the file yet).
    if is_baseline:
        master_cols = ["name", "topology", "metal_node"] + TARGET_COLS
    else:
        master_cols = ["name"] + feat_cols + TARGET_COLS

    master = pd.read_parquet(_MASTER_PATH, columns=master_cols).set_index("name")

    # Load each partition
    dfs: dict[str, pd.DataFrame] = {}
    for partition in ("train", "test"):
        split_path = _SPLITS_DIR / f"split_{split_strategy}_{partition}_merged.parquet"
        names = pd.read_parquet(split_path, columns=["name"])["name"]
        dfs[partition] = master.loc[names].reset_index(drop=True)

    df_train, df_test = dfs["train"], dfs["test"]

    if is_baseline:
        # encode_categoricals always returns a 3-tuple; pass None for the
        # (removed) val slot and discard it.
        df_train, _, df_test = encode_categoricals(
            df_train, None, df_test, method=encoding
        )
        feat_cols = ["topology_encoded", "metal_node_encoded"]

    X_train = df_train[feat_cols].to_numpy(dtype=np.float64)
    X_test  = df_test[feat_cols].to_numpy(dtype=np.float64)

    y_train = df_train[TARGET_COLS].to_numpy(dtype=np.float64)
    y_test  = df_test[TARGET_COLS].to_numpy(dtype=np.float64)

    return X_train, X_test, y_train, y_test
