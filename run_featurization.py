#!/usr/bin/env python3
"""
run_featurization.py
====================
Compute and save features for both datasets.

Outputs (in processed/featurization/):
  - features_1185.parquet  (1155 molecules × ~1942 features)
  - features_251.parquet   (~800 sites × ~1887 features)
  - metadata_1185.json     (feature names, sources, counts, target info)
  - metadata_251.json
"""
import sys, os, json, time, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

import numpy as np
import pandas as pd
from rdkit import Chem

from code.data_loading import (
    B3LYP1185Loader, B3LYP251Loader,
    PM7_1185Loader, PM7_251Loader,
    KJMOL_TO_KCALMOL,
)
from code.featurization import FeatureEngine

OUT_DIR = "processed/featurization"
os.makedirs(OUT_DIR, exist_ok=True)


def canon(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else str(smiles)
    except Exception:
        return str(smiles)


# ============================================================
# 1185-MOLECULE DATASET
# ============================================================
def featurize_1185():
    print("\n" + "=" * 60)
    print("FEATURIZING 1185-MOLECULE DATASET")
    print("=" * 60)
    t0 = time.time()

    # ---- Load data ----
    pm7_loader = PM7_1185Loader("data/1185_molecules")
    df_dataset = pm7_loader.load()  # 1185 rows, has smiles + EXP_PA + Mordred + FP + PM7
    print(f"Dataset.csv: {len(df_dataset)} molecules")

    # Load B3LYP DFT data and merge
    b3lyp_loader = B3LYP1185Loader("data/1185_molecules")
    df_b3lyp = b3lyp_loader.load()
    df_b3lyp_ok = df_b3lyp[df_b3lyp["status"] == "OK"].copy()
    print(f"B3LYP OK: {len(df_b3lyp_ok)} molecules")

    # Canonicalize for merge
    df_dataset["can"] = df_dataset["smiles"].apply(canon)
    df_b3lyp_ok["can"] = df_b3lyp_ok["smiles"].apply(canon)

    # Merge to get protonated SMILES from B3LYP
    df = df_dataset.merge(
        df_b3lyp_ok[["can", "prot_smiles", "dft_pa_kcal", "exp_pa_kcal"]],
        on="can", how="inner",
    )
    print(f"Merged: {len(df)} molecules with both data sources")

    # Load PM7 neutral/protonated files for PM7 quantum features
    pm7_neu_path = "data/1185_molecules/pm7/FINAL_PM7_ALL_neutral_cleaned.csv"
    pm7_prot_path = "data/1185_molecules/pm7/FINAL_PM7_ALL_protonated_cleaned.csv"

    if os.path.exists(pm7_neu_path) and os.path.exists(pm7_prot_path):
        d_neu = pd.read_csv(pm7_neu_path)
        d_prot = pd.read_csv(pm7_prot_path)
        d_neu["can"] = d_neu["smiles"].apply(canon)
        d_prot["can"] = d_prot["neutral_smiles"].apply(canon)
        # Best protonated site per molecule
        d_prot_best = d_prot.sort_values(["can", "heat_of_formation"]).drop_duplicates("can")
        # Merge neutral + protonated PM7
        pm7_merged = pd.merge(d_neu, d_prot_best, on="can", suffixes=("_n", "_p"))
        # Merge with main df
        df = df.merge(pm7_merged[["can"] + [c for c in pm7_merged.columns if c != "can"]],
                       on="can", how="inner")
        print(f"After PM7 merge: {len(df)} molecules")
        pm7_df = df
        pm7_n_suffix = "_n"
        pm7_p_suffix = "_p"
        # Protonated SMILES for RDKit delta
        smiles_prot = df["smiles_p"].tolist() if "smiles_p" in df.columns else df["can"].tolist()
    else:
        pm7_df = None
        pm7_n_suffix = "_n"
        pm7_p_suffix = "_p"
        smiles_prot = df["can"].tolist()

    smiles_neutral = df["can"].tolist()

    # Mordred features from Dataset.csv
    exclude_patterns = ["fp_", "smiles", "reg_num", "EXP_PA", "Ehomo", "Elumo",
                        "chemical potential", "hardness", "MK_charge", "Dipole moment",
                        "CM5_charge", "can", "prot_smiles", "dft_pa_kcal", "exp_pa_kcal"]
    mordred_cols = [c for c in df.columns
                    if not any(p in c for p in exclude_patterns)
                    and df[c].dtype in ["float64", "int64"]
                    and not c.endswith("_n") and not c.endswith("_p")]
    # Filter to only original Dataset.csv columns
    orig_dataset_cols = set(df_dataset.columns)
    mordred_cols = [c for c in mordred_cols if c in orig_dataset_cols]
    mordred_df = df[mordred_cols]

    # ---- Build features ----
    engine = FeatureEngine()
    X, names, sources = engine.build_features(
        smiles_neutral=smiles_neutral,
        smiles_protonated=smiles_prot,
        pm7_df=pm7_df,
        pm7_neutral_suffix=pm7_n_suffix,
        pm7_prot_suffix=pm7_p_suffix,
        mordred_df=mordred_df,
        compute_3d=True,
    )

    # ---- Build output DataFrame ----
    df_features = pd.DataFrame(X, columns=names)

    # Add identifiers and targets
    df_features.insert(0, "smiles", smiles_neutral)
    df_features.insert(1, "exp_pa_kcal", df["EXP_PA"].values * KJMOL_TO_KCALMOL
                       if df["EXP_PA"].mean() > 500 else df["EXP_PA"].values)
    df_features.insert(2, "dft_pa_kcal", df["dft_pa_kcal"].values)

    # PM7 PA
    if "heat_of_formation_n" in df.columns and "heat_of_formation_p" in df.columns:
        pm7_pa = df["heat_of_formation_n"].values + 365.7 - df["heat_of_formation_p"].values
        df_features.insert(3, "pm7_pa_kcal", pm7_pa)

    # ---- Save ----
    out_path = os.path.join(OUT_DIR, "features_1185.parquet")
    df_features.to_parquet(out_path, index=False, engine="fastparquet")
    print(f"\nSaved: {out_path}")
    print(f"  Shape: {df_features.shape}")

    # Metadata
    meta = {
        "dataset": "1185_molecules",
        "n_molecules": len(df_features),
        "n_features": len(names),
        "feature_names": names,
        "feature_sources": sources,
        "feature_counts": engine.feature_counts_,
        "target_columns": ["exp_pa_kcal", "dft_pa_kcal", "pm7_pa_kcal"],
        "identifier_columns": ["smiles"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = os.path.join(OUT_DIR, "metadata_1185.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {meta_path}")

    elapsed = time.time() - t0
    print(f"1185 featurization complete in {elapsed:.1f}s")
    return df_features


# ============================================================
# 251-MOLECULE DATASET
# ============================================================
def featurize_251():
    print("\n" + "=" * 60)
    print("FEATURIZING 251-MOLECULE DATASET")
    print("=" * 60)
    t0 = time.time()

    # ---- Load & merge data ----
    # B3LYP site-level data
    b3lyp_loader = B3LYP251Loader("data/251_molecules")
    df_b3lyp = b3lyp_loader.load_sites()
    print(f"B3LYP sites: {len(df_b3lyp)} from {df_b3lyp['mol_id'].nunique()} molecules")

    # PM7 data
    pm7_loader = PM7_251Loader("data/251_molecules")
    df_pm7 = pm7_loader.load()
    print(f"PM7 sites: {len(df_pm7)}")

    # Canonicalize SMILES for merging
    df_b3lyp["neutral_smiles_canon"] = df_b3lyp["neutral_smiles"].apply(canon)
    df_b3lyp["protonated_smiles_canon"] = df_b3lyp["protonated_smiles"].apply(canon)
    df_pm7["neutral_smiles_canon"] = df_pm7["neutral_smiles"].apply(canon)
    df_pm7["protonated_smiles_canon"] = df_pm7["protonated_smiles"].apply(canon)

    # Merge on both neutral and protonated SMILES
    df = df_b3lyp.merge(
        df_pm7,
        on=["neutral_smiles_canon", "protonated_smiles_canon"],
        how="inner",
        suffixes=("_b3lyp", "_pm7"),
    )
    print(f"Merged: {len(df)} sites with both B3LYP and PM7 data")

    # Resolve site columns
    if "site_index_b3lyp" in df.columns:
        df["site_index"] = df["site_index_b3lyp"]
    if "protonation_site_element_protonated" in df.columns:
        df["site_element"] = df["protonation_site_element_protonated"]

    # Targets
    y_dft = df["dft_pa_kcal"].values.astype(float)
    y_pm7 = df["pm7_pa_kcal"].values.astype(float)

    smiles_neutral = df["neutral_smiles_canon"].tolist()
    smiles_prot = df["protonated_smiles_canon"].tolist()

    # ---- Build features ----
    engine = FeatureEngine()
    X, names, sources = engine.build_features(
        smiles_neutral=smiles_neutral,
        smiles_protonated=smiles_prot,
        pm7_df=df,
        pm7_neutral_suffix="_neutral",
        pm7_prot_suffix="_protonated",
        site_elements=df.get("site_element"),
        site_indices=df.get("site_index").astype(float) if "site_index" in df.columns else None,
        mol_ids=df.get("neutral_smiles_canon") if "neutral_smiles_canon" in df.columns else None,
    )

    # ---- Build output DataFrame ----
    df_features = pd.DataFrame(X, columns=names)

    # Add identifiers and targets
    df_features.insert(0, "neutral_smiles", smiles_neutral)
    df_features.insert(1, "protonated_smiles", smiles_prot)
    df_features.insert(2, "mol_id", df["mol_id"].values if "mol_id" in df.columns else range(len(df)))
    df_features.insert(3, "site_index", df["site_index"].values if "site_index" in df.columns else 0)
    df_features.insert(4, "site_element", df["site_element"].values if "site_element" in df.columns else "")
    df_features.insert(5, "dft_pa_kcal", y_dft)
    df_features.insert(6, "pm7_pa_kcal", y_pm7)

    # ---- Save ----
    out_path = os.path.join(OUT_DIR, "features_251.parquet")
    df_features.to_parquet(out_path, index=False, engine="fastparquet")
    print(f"\nSaved: {out_path}")
    print(f"  Shape: {df_features.shape}")

    # Metadata
    meta = {
        "dataset": "251_molecules",
        "n_sites": len(df_features),
        "n_molecules": df["mol_id"].nunique() if "mol_id" in df.columns else len(df),
        "n_features": len(names),
        "feature_names": names,
        "feature_sources": sources,
        "feature_counts": engine.feature_counts_,
        "target_columns": ["dft_pa_kcal", "pm7_pa_kcal"],
        "identifier_columns": ["neutral_smiles", "protonated_smiles", "mol_id", "site_index", "site_element"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = os.path.join(OUT_DIR, "metadata_251.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {meta_path}")

    elapsed = time.time() - t0
    print(f"251 featurization complete in {elapsed:.1f}s")
    return df_features


# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PROTON AFFINITY FEATURIZATION PIPELINE")
    print("=" * 60)

    df_1185 = featurize_1185()
    df_251 = featurize_251()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"1185-molecule: {df_1185.shape[0]} molecules × {df_1185.shape[1]} columns")
    print(f"251-molecule:  {df_251.shape[0]} sites × {df_251.shape[1]} columns")
    print(f"\nSaved to: {OUT_DIR}/")
    print("  features_1185.parquet + metadata_1185.json")
    print("  features_251.parquet  + metadata_251.json")
    print("\nDONE!")
