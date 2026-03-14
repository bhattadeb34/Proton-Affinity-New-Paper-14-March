#!/usr/bin/env python3
"""
Delta-learning cross-validation for NIST and Site-specific datasets.

Four variants:
  NIST-i:   Δ_exp, features = PM7/RDKit/Morgan/MACCS only
  NIST-ii:  Δ_exp, features = all (incl. Mordred, 3D)
  Site-i:   Δ_DFT, features = PM7/RDKit/Morgan/MACCS/Site only
  Site-ii:  Δ_DFT, features = all

Pipeline: y_delta = target_PA - PM7_PA; outer 5-fold CV with per-fold feature
selection; train 9 models; final_PA = PM7_PA + pred_delta; metric = MAE(final_PA, target_PA).

Saves to: processed/model_outputs/
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

# Feature selection from proton-affinity-paper-clean (sibling repo)
ROOT = Path(__file__).resolve().parents[0]
_clean_code = ROOT.parent / "proton-affinity-paper-clean" / "code"
if _clean_code.exists():
    sys.path.insert(0, str(_clean_code))
else:
    _clean_code = ROOT / "proton-affinity-paper-clean" / "code"
    if _clean_code.exists():
        sys.path.insert(0, str(_clean_code))
from feature_selection import (
    K_CANDIDATES_NIST,
    K_CANDIDATES_SITE,
    select_features_fold,
)

try:
    import xgboost
    XGBRegressor = xgboost.XGBRegressor
except ImportError:
    XGBRegressor = None

warnings.filterwarnings("ignore", category=UserWarning)

# Paths (ROOT set above)
FEAT_DIR = ROOT / "processed" / "featurization"
OUT_DIR = ROOT / "processed" / "model_outputs"
RANDOM_STATE = 42
N_FOLDS = 5

# Feature sources: no-DFT variants use only these
SOURCES_NO_DFT = {"PM7", "RDKit", "Morgan", "MACCS", "Site"}


def _get_feature_subset(meta: dict, with_dft: bool):
    """Return (feature_names, feature_sources) for the variant."""
    names = meta["feature_names"]
    sources = meta["feature_sources"]
    if with_dft:
        return names, sources
    keep = [i for i, s in enumerate(sources) if s in SOURCES_NO_DFT]
    return [names[i] for i in keep], [sources[i] for i in keep]


def _build_models():
    """Return list of (model_name, model_instance)."""
    models = [
        ("RidgeCV", RidgeCV(cv=5, scoring="neg_mean_absolute_error")),
        ("LassoCV", LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=5000)),
        ("ElasticNetCV", ElasticNetCV(cv=5, random_state=RANDOM_STATE, max_iter=5000)),
        ("BayesianRidge", BayesianRidge()),
        ("DecisionTree", DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE)),
        ("RandomForest", RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)),
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)),
    ]
    if XGBRegressor is not None:
        models.append(("XGBoost", XGBRegressor(n_estimators=100, random_state=RANDOM_STATE)))
    models.append(
        (
            "MLPRegressor",
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                max_iter=1000,
                early_stopping=True,
                random_state=RANDOM_STATE,
                validation_fraction=0.1,
            ),
        ),
    )
    return models


def run_one_variant(
    variant_id: str,
    X: np.ndarray,
    feature_names: list,
    feature_sources: list,
    target_pa: np.ndarray,
    pm7_pa: np.ndarray,
    k_candidates: list[int] | None,
    verbose: bool = True,
):
    """
    Run outer 5-fold CV with delta learning. Returns dict of results.
    """
    y_delta = target_pa - pm7_pa
    n = X.shape[0]
    if X.shape[1] == 0:
        raise ValueError(f"{variant_id}: no features after filtering")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    model_list = _build_models()
    # Collect: per-fold MAE per model, and fold indices for reproducibility
    results = {
        "variant": variant_id,
        "n_samples": n,
        "n_features": X.shape[1],
        "fold_mae": {name: [] for name, _ in model_list},
        "fold_indices": [],
        "per_fold_reports": [],
    }

    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n))):
        if verbose:
            print(f"  Fold {fold + 1}/{N_FOLDS}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_delta_train = y_delta[train_idx]
        pm7_test = pm7_pa[test_idx]
        target_test = target_pa[test_idx]

        mask, _indices, scaler, report = select_features_fold(
            X_train,
            y_delta_train,
            feature_names,
            feature_sources,
            k_candidates=k_candidates,
            random_state=RANDOM_STATE,
            verbose=False,
        )
        results["per_fold_reports"].append(report)

        X_train_sel = scaler.transform(X_train[:, mask])
        X_test_sel = scaler.transform(X_test[:, mask])
        n_sel = X_train_sel.shape[1]
        results["fold_indices"].append({"train": train_idx.tolist(), "test": test_idx.tolist()})

        for name, model in model_list:
            clf = __clone_or_fit(model, X_train_sel, y_delta_train, name)
            pred_delta = clf.predict(X_test_sel)
            final_pa = pm7_test + pred_delta
            mae = mean_absolute_error(final_pa, target_test)
            results["fold_mae"][name].append(mae)

    # Aggregate (use Python floats for JSON)
    for name in results["fold_mae"]:
        maes = results["fold_mae"][name]
        mean_, std_ = np.mean(maes), np.std(maes)
        results["fold_mae"][name] = {
            "maes": [float(m) for m in maes],
            "mean": float(mean_),
            "std": float(std_) if not np.isnan(np.std(maes)) else 0.0,
        }
    return results


def __clone_or_fit(model, X_train_sel, y_delta_train, name):
    """Clone sklearn-style model and fit; handle MLP early_stopping."""
    from sklearn.base import clone
    m = clone(model)
    if name == "MLPRegressor":
        m.fit(X_train_sel, y_delta_train)
    else:
        m.fit(X_train_sel, y_delta_train)
    return m


def load_nist():
    """Load 1185-molecule (NIST) data from processed/featurization."""
    parquet_path = FEAT_DIR / "features_1185.parquet"
    meta_path = FEAT_DIR / "metadata_1185.json"
    if not parquet_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Run run_featurization.py first. Missing: {parquet_path} or {meta_path}")
    df = pd.read_parquet(parquet_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return df, meta


def load_site():
    """Load 251/site (821 sites) data from processed/featurization."""
    parquet_path = FEAT_DIR / "features_251.parquet"
    meta_path = FEAT_DIR / "metadata_251.json"
    if not parquet_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Run run_featurization.py first. Missing: {parquet_path} or {meta_path}")
    df = pd.read_parquet(parquet_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return df, meta


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Delta-learning CV: NIST + Site, 4 variants, 9 models")
    print("Output:", OUT_DIR)

    # ---- NIST ----
    print("\n" + "=" * 60)
    print("NIST (1185 molecules)")
    print("=" * 60)
    df_nist, meta_nist = load_nist()
    target_nist = df_nist["exp_pa_kcal"].values.astype(float)
    pm7_nist = df_nist["pm7_pa_kcal"].values.astype(float)

    for variant, with_dft in [("NIST-i", False), ("NIST-ii", True)]:
        fnames, fsources = _get_feature_subset(meta_nist, with_dft)
        X_nist = df_nist[fnames].values.astype(np.float64)
        X_nist = np.nan_to_num(X_nist)
        print(f"\n{variant} (with_dft={with_dft}, n_features={X_nist.shape[1]})")
        res = run_one_variant(
            variant,
            X_nist,
            fnames,
            fsources,
            target_nist,
            pm7_nist,
            k_candidates=K_CANDIDATES_NIST,
            verbose=True,
        )
        out_path = OUT_DIR / f"delta_cv_{variant.replace('-', '_')}.json"
        def _json_default(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            raise TypeError(type(obj))
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2, default=_json_default)
        print(f"  Saved: {out_path}")
        for name, v in res["fold_mae"].items():
            print(f"    {name}: MAE = {v['mean']:.4f} ± {v['std']:.4f} kcal/mol")

    # ---- Site ----
    print("\n" + "=" * 60)
    print("Site-specific (821 sites)")
    print("=" * 60)
    df_site, meta_site = load_site()
    target_site = df_site["dft_pa_kcal"].values.astype(float)
    pm7_site = df_site["pm7_pa_kcal"].values.astype(float)

    for variant, with_dft in [("Site-i", False), ("Site-ii", True)]:
        fnames, fsources = _get_feature_subset(meta_site, with_dft)
        X_site = df_site[fnames].values.astype(np.float64)
        X_site = np.nan_to_num(X_site)
        print(f"\n{variant} (with_dft={with_dft}, n_features={X_site.shape[1]})")
        res = run_one_variant(
            variant,
            X_site,
            fnames,
            fsources,
            target_site,
            pm7_site,
            k_candidates=K_CANDIDATES_SITE,
            verbose=True,
        )
        out_path = OUT_DIR / f"delta_cv_{variant.replace('-', '_')}.json"
        def _json_default(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            raise TypeError(type(obj))
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2, default=_json_default)
        print(f"  Saved: {out_path}")
        for name, v in res["fold_mae"].items():
            print(f"    {name}: MAE = {v['mean']:.4f} ± {v['std']:.4f} kcal/mol")

    # Summary table
    summary_rows = []
    for fname in sorted(OUT_DIR.glob("delta_cv_*.json")):
        with open(fname) as f:
            data = json.load(f)
        for model_name, v in data["fold_mae"].items():
            summary_rows.append({
                "variant": data["variant"],
                "model": model_name,
                "MAE_mean": v["mean"],
                "MAE_std": v["std"],
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "delta_cv_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
