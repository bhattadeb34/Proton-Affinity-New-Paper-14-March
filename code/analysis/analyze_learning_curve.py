#!/usr/bin/env python3
"""
Learning curve on NIST.

For train_frac in [0.1, 0.2, ..., 1.0]: 5-fold CV with subsampled train
(full pipeline: select_features_fold + BayesianRidge). Returns/saves DataFrame
with train_frac, train_size, cv_mae_mean, cv_mae_std.

Uses: feature_selection from proton-affinity-paper-clean/code.
Data: processed/featurization/features_1185.parquet (NIST-i).
Saves: processed/analysis/nist_learning_curve.csv.
If featurization data is missing, returns empty DataFrame and skips without error.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# Repo root (parent of code/analysis)
ROOT = Path(__file__).resolve().parents[2]
_clean_code = ROOT.parent / "proton-affinity-paper-clean" / "code"
if _clean_code.exists():
    sys.path.insert(0, str(_clean_code))
else:
    _clean_code = ROOT / "proton-affinity-paper-clean" / "code"
    if _clean_code.exists():
        sys.path.insert(0, str(_clean_code))
from feature_selection import select_features_fold, K_CANDIDATES_NIST, RANDOM_STATE

warnings.filterwarnings("ignore")

FEAT_DIR = ROOT / "processed" / "featurization"
OUT_DIR = ROOT / "processed" / "analysis"
SEED = 42
TRAIN_FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_FOLDS = 5
SOURCES_NO_DFT = {"PM7", "RDKit", "Morgan", "MACCS"}


def load_nist_X_y_delta():
    """Load NIST X, y_delta, feature_names, feature_sources (NIST-i). Returns None if data missing."""
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


def run_learning_curve(
    X: np.ndarray,
    y_delta: np.ndarray,
    feature_names: list,
    feature_sources: list,
    train_fracs: list[float] | None = None,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    For each train_frac, 5-fold CV: subsample train to train_size = int(train_frac * N),
    select_features_fold + BayesianRidge, predict test, MAE.
    Returns DataFrame: train_frac, train_size, cv_mae_mean, cv_mae_std.
    """
    if train_fracs is None:
        train_fracs = TRAIN_FRACS
    n = X.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for frac in train_fracs:
        train_size = max(1, int(frac * n))
        maes = []
        fold_idx = 0
        for train_idx, test_idx in kf.split(np.arange(n)):
            rng = np.random.RandomState(seed + fold_idx)
            perm = rng.permutation(len(train_idx))
            take = min(train_size, len(train_idx))
            sub_train_idx = train_idx[perm[:take]]
            fold_idx += 1
            X_train = X[sub_train_idx]
            y_train = y_delta[sub_train_idx]
            X_test = X[test_idx]
            y_test = y_delta[test_idx]
            if len(sub_train_idx) < 10:
                maes.append(np.nan)
                continue
            mask, _indices, scaler, _report = select_features_fold(
                X_train,
                y_train,
                feature_names,
                feature_sources,
                k_candidates=K_CANDIDATES_NIST,
                random_state=RANDOM_STATE,
                verbose=False,
            )
            X_train_sel = scaler.transform(X_train[:, mask])
            X_test_sel = scaler.transform(X_test[:, mask])
            model = BayesianRidge()
            model.fit(X_train_sel, y_train)
            pred = model.predict(X_test_sel)
            mae = mean_absolute_error(y_test, pred)
            maes.append(mae)
        maes = np.array(maes)
        valid = ~np.isnan(maes)
        cv_mean = float(np.mean(maes[valid])) if valid.any() else np.nan
        cv_std = float(np.std(maes[valid])) if valid.sum() > 1 else 0.0
        rows.append({
            "train_frac": frac,
            "train_size": train_size,
            "cv_mae_mean": cv_mean,
            "cv_mae_std": cv_std,
        })
    return pd.DataFrame(rows)


def main() -> pd.DataFrame:
    """Run analysis and save CSV. If featurization data missing, skip without error and return empty DataFrame."""
    data = load_nist_X_y_delta()
    if data is None:
        return pd.DataFrame(columns=["train_frac", "train_size", "cv_mae_mean", "cv_mae_std"])
    X, y_delta, fnames, fsources = data
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = run_learning_curve(X, y_delta, fnames, fsources)
    if df.empty:
        return df
    out_path = OUT_DIR / "nist_learning_curve.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df


if __name__ == "__main__":
    main()
