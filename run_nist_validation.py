#!/usr/bin/env python3
"""
Master script for NIST validation analysis.

Runs in order (without re-running steps that already have outputs):
  1. Features vs performance (optimal k, parsimony k) → nist_features_vs_perf.csv
  2. Learning curve (MAE vs train size) → nist_learning_curve.csv
  3. Plots → nist_validation_plots.pdf

Uses featurization data: processed/featurization/features_1185.parquet.
Uses CV data only for plot (median selected features): processed/model_outputs/delta_cv_NIST_i.json.
If featurization or CV data is missing, skips the affected steps without error.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
FEAT_DIR = ROOT / "processed" / "featurization"
ANALYSIS_DIR = ROOT / "processed" / "analysis"
MODEL_OUTPUTS_DIR = ROOT / "processed" / "model_outputs"

PERF_CSV = ANALYSIS_DIR / "nist_features_vs_perf.csv"
LC_CSV = ANALYSIS_DIR / "nist_learning_curve.csv"
CV_JSON = MODEL_OUTPUTS_DIR / "delta_cv_NIST_i.json"


def main() -> None:
    # Featurization data required for analysis 1 and 2
    if not (FEAT_DIR / "features_1185.parquet").exists() or not (FEAT_DIR / "metadata_1185.json").exists():
        print("Skipping NIST validation: featurization data not found (run run_featurization.py first).")
        return

    sys.path.insert(0, str(ROOT))

    # 1. Features vs performance — run only if output missing
    if not PERF_CSV.exists():
        from code.analysis.analyze_features_vs_performance import main as run_perf
        run_perf()
    else:
        print(f"Using existing: {PERF_CSV}")

    # 2. Learning curve — run only if output missing
    if not LC_CSV.exists():
        from code.analysis.analyze_learning_curve import main as run_lc
        run_lc()
    else:
        print(f"Using existing: {LC_CSV}")

    # 3. Plots — only if both CSVs exist; CV data optional (median line omitted if missing)
    if not CV_JSON.exists():
        print("CV results not found; plot will omit median selected features line.")
    if PERF_CSV.exists() and LC_CSV.exists():
        from code.plotting.plot_nist_validation import main as run_plot
        if run_plot():
            print("NIST validation done.")
        else:
            print("Plot skipped (missing or empty analysis CSVs).")
    else:
        print("Skipping plots: analysis CSVs not available.")


if __name__ == "__main__":
    main()
