#!/usr/bin/env python3
"""
Features vs performance on NIST (one 80/20 fold).

Runs stages 1–3 (Lasso ranking), then for each k in [25,50,...,400]
computes RidgeCV 5-fold CV MAE on top-k. Returns/saves DataFrame with
k, cv_mae_mean, cv_mae_std, optimal_k, parsimony_k.

Uses: feature_selection from proton-affinity-paper-clean/code.
Data: processed/featurization/features_1185.parquet (NIST-i).
Saves: processed/analysis/nist_features_vs_perf.csv.
If featurization data is missing, returns empty DataFrame and skips without error.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer

# Repo root (parent of code/analysis)
ROOT = Path(__file__).resolve().parents[2]
_clean_code = ROOT.parent / "proton-affinity-paper-clean" / "code"
if _clean_code.exists():
    sys.path.insert(0, str(_clean_code))
else:
    _clean_code = ROOT / "proton-affinity-paper-clean" / "code"
    if _clean_code.exists():
        sys.path.insert(0, str(_clean_code))
from feature_selection import (
    _stage1_variance,
    _stage2_correlation,
    _stage3_rank,
    RANDOM_STATE,
    VARIANCE_THRESHOLD,
    CORR_THRESHOLD_DEFAULT,
    CORR_THRESHOLD_GROUP_PROTECT,
)

warnings.filterwarnings("ignore")

FEAT_DIR = ROOT / "processed" / "featurization"
OUT_DIR = ROOT / "processed" / "analysis"
SEED = 42
K_CANDIDATES = [25, 50, 75, 100, 150, 200, 250, 300, 350, 400]
SOURCES_NO_DFT = {"PM7", "RDKit", "Morgan", "MACCS"}


def load_nist_X_y_delta():
    """Load NIST feature matrix, y_delta, feature_names, feature_sources (NIST-i). Returns None if data missing."""
    parquet_path = FEAT_DIR / "features_1185.parquet"
    meta_path = FEAT_DIR / "metadata_1185.json"
    if not parquet_path.exists() or not meta_path.exists():
        return None
    df = pd.read_parquet(parquet_path)
    with open(meta_path) as f:
        meta = json.load(f)
    names = meta["feature_names"]
    sources = meta["feature_sources"]
    keep = [i for i, s in enumerate(sources) if s in SOURCES_NO_DFT]
    fnames = [names[i] for i in keep]
    fsources = [sources[i] for i in keep]
    X = df[fnames].values.astype(np.float64)
    X = np.nan_to_num(X)
    target = df["exp_pa_kcal"].values.astype(float)
    pm7 = df["pm7_pa_kcal"].values.astype(float)
    y_delta = target - pm7
    return X, y_delta, fnames, fsources


def run_one_fold_analysis(
    X: np.ndarray,
    y_delta: np.ndarray,
    feature_names: list,
    feature_sources: list,
    seed: int = SEED,
    k_candidates: list[int] | None = None,
) -> pd.DataFrame:
    """
    Single 80/20 split: stages 1–3, then for each k RidgeCV 5-fold CV MAE.
    Returns DataFrame with k, cv_mae_mean, cv_mae_std, optimal_k, parsimony_k.
    """
    if k_candidates is None:
        k_candidates = K_CANDIDATES

    n = X.shape[0]
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(0.8 * n)
    train_idx, _test_idx = idx[:split], idx[split:]
    X_train = X[train_idx]
    y_train = y_delta[train_idx]

    # Stage 1
    X1, names1, sources1 = _stage1_variance(
        X_train, feature_names, feature_sources, threshold=VARIANCE_THRESHOLD
    )
    if X1.shape[1] == 0:
        return pd.DataFrame(columns=["k", "cv_mae_mean", "cv_mae_std", "optimal_k", "parsimony_k"])

    # Stage 2
    X2, names2, sources2, keep_idx = _stage2_correlation(
        X1, y_train, names1, sources1,
        corr_thresh=CORR_THRESHOLD_DEFAULT,
        group_protect_thresh=CORR_THRESHOLD_GROUP_PROTECT,
    )
    if X2.shape[1] == 0:
        return pd.DataFrame(columns=["k", "cv_mae_mean", "cv_mae_std", "optimal_k", "parsimony_k"])

    # Stage 3
    X3_scaled, _n3, _s3, sorted_idx, _scaler = _stage3_rank(
        X2, y_train, names2, sources2, random_state=RANDOM_STATE
    )
    n_avail = len(sorted_idx)
    k_list = [k for k in k_candidates if 1 <= k <= n_avail]
    if not k_list:
        k_list = [min(50, n_avail)]

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    rows = []
    for k in k_list:
        top_k = sorted_idx[:k]
        X_k = X3_scaled[:, top_k]
        ridge = RidgeCV(cv=5, scoring=make_scorer(mean_absolute_error, greater_is_better=False))
        scores = cross_val_score(ridge, X_k, y_train, cv=kf, scoring=mae_scorer, n_jobs=-1)
        mae_mean = float(-scores.mean())
        mae_std = float(scores.std() if not np.isnan(scores.std()) else 0.0)
        rows.append({"k": k, "cv_mae_mean": mae_mean, "cv_mae_std": mae_std})

    df = pd.DataFrame(rows)
    if df.empty:
        df["optimal_k"] = np.nan
        df["parsimony_k"] = np.nan
        return df

    best = df.loc[df["cv_mae_mean"].idxmin()]
    optimal_k = int(best["k"])
    min_mae = best["cv_mae_mean"]
    min_std = best["cv_mae_std"]
    threshold = min_mae + min_std
    parsimony_row = df[df["cv_mae_mean"] <= threshold].sort_values("k").iloc[0]
    parsimony_k = int(parsimony_row["k"])
    df["optimal_k"] = optimal_k
    df["parsimony_k"] = parsimony_k
    return df


def main() -> pd.DataFrame:
    """Run analysis and save CSV. If featurization data missing, skip without error and return empty DataFrame."""
    data = load_nist_X_y_delta()
    if data is None:
        return pd.DataFrame(columns=["k", "cv_mae_mean", "cv_mae_std", "optimal_k", "parsimony_k"])
    X, y_delta, fnames, fsources = data
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = run_one_fold_analysis(X, y_delta, fnames, fsources)
    if df.empty:
        return df
    out_path = OUT_DIR / "nist_features_vs_perf.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df


if __name__ == "__main__":
    main()
