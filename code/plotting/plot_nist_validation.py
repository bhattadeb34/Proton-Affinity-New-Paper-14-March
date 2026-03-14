#!/usr/bin/env python3
"""
Plot NIST validation (features vs performance + learning curve).

Loads:
  - processed/analysis/nist_features_vs_perf.csv
  - processed/analysis/nist_learning_curve.csv
  - processed/model_outputs/delta_cv_NIST_i.json (optional; for median_selected_features)

Fig 1 (left): k vs cv_mae_mean, errorbars=cv_mae_std; vlines at optimal_k (blue ★),
  median_selected_features (red ♦) if CV data available.
Fig 2 (right): train_size vs cv_mae_mean, errorbars=cv_mae_std.

If either CSV is missing, skips without error. If CV JSON missing, omits median line only.
Fonts: axis labels 24, legend/ticks 18, min 18; spine linewidth 1.5.
Saves: processed/analysis/nist_validation_plots.pdf (dpi=300, bbox_inches='tight').
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Repo root (parent of code/plotting)
ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "processed" / "analysis"
MODEL_OUTPUTS_DIR = ROOT / "processed" / "model_outputs"

FONT_LABELS = 24
FONT_LEGEND_TICKS = 18
FONT_MIN = 18
SPINE_LW = 1.5

matplotlib.rcParams["axes.titlesize"] = FONT_LABELS
matplotlib.rcParams["axes.labelsize"] = FONT_LABELS
matplotlib.rcParams["xtick.labelsize"] = FONT_LEGEND_TICKS
matplotlib.rcParams["ytick.labelsize"] = FONT_LEGEND_TICKS
matplotlib.rcParams["legend.fontsize"] = FONT_LEGEND_TICKS
matplotlib.rcParams["font.size"] = max(FONT_LEGEND_TICKS, FONT_MIN)


def load_median_selected_features() -> float | None:
    """From delta_cv_NIST_i.json per_fold_reports, median of n_final. Returns None if missing (skip without error)."""
    path = MODEL_OUTPUTS_DIR / "delta_cv_NIST_i.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None
    reports = data.get("per_fold_reports", [])
    if not reports:
        return None
    n_finals = [r.get("n_final") for r in reports if "n_final" in r]
    if not n_finals:
        return None
    return float(np.median(n_finals))


def plot_features_vs_performance(
    df: pd.DataFrame,
    median_selected: float | None,
    ax: plt.Axes,
) -> None:
    """Fig 1: k vs cv_mae_mean with errorbars; vlines optimal_k (blue ★), median_selected (red ♦) if not None."""
    ax.errorbar(
        df["k"],
        df["cv_mae_mean"],
        yerr=df["cv_mae_std"],
        capsize=3,
        marker="o",
        linestyle="-",
        color="black",
        markersize=6,
    )
    optimal_k = df["optimal_k"].iloc[0] if "optimal_k" in df.columns and len(df) else None
    if optimal_k is not None and not np.isnan(optimal_k):
        ax.axvline(optimal_k, color="blue", linestyle="--", linewidth=1.5, label=f"optimal k = {int(optimal_k)}")
        row = df[df["k"] == optimal_k]
        if not row.empty:
            ax.scatter([optimal_k], [row["cv_mae_mean"].values[0]], marker="*", s=400, color="blue", zorder=5)
    if median_selected is not None:
        ax.axvline(median_selected, color="red", linestyle="-.", linewidth=1.5, label=f"median selected = {int(median_selected)}")
        y_med = np.interp(median_selected, df["k"].values, df["cv_mae_mean"].values)
        ax.scatter([median_selected], [y_med], marker="D", s=200, color="red", zorder=5)
    ax.set_xlabel("Number of features (k)", fontsize=FONT_LABELS)
    ax.set_ylabel("CV MAE (kcal/mol)", fontsize=FONT_LABELS)
    ax.tick_params(axis="both", labelsize=FONT_LEGEND_TICKS)
    ax.legend(fontsize=FONT_LEGEND_TICKS)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)
    ax.yaxis.get_offset_text().set_fontsize(FONT_MIN)


def plot_learning_curve(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Fig 2: train_size vs cv_mae_mean with errorbars."""
    ax.errorbar(
        df["train_size"],
        df["cv_mae_mean"],
        yerr=df["cv_mae_std"],
        capsize=3,
        marker="o",
        linestyle="-",
        color="black",
        markersize=6,
    )
    ax.set_xlabel("Training set size", fontsize=FONT_LABELS)
    ax.set_ylabel("CV MAE (kcal/mol)", fontsize=FONT_LABELS)
    ax.tick_params(axis="both", labelsize=FONT_LEGEND_TICKS)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)
    ax.yaxis.get_offset_text().set_fontsize(FONT_MIN)


def main() -> bool:
    """
    Load CSVs and plot. Does not run analysis scripts.
    Returns True if PDF was saved, False if skipped (missing data).
    """
    perf_path = ANALYSIS_DIR / "nist_features_vs_perf.csv"
    lc_path = ANALYSIS_DIR / "nist_learning_curve.csv"
    if not perf_path.exists() or not lc_path.exists():
        return False
    df_perf = pd.read_csv(perf_path)
    df_lc = pd.read_csv(lc_path)
    if df_perf.empty or df_lc.empty:
        return False
    median_selected = load_median_selected_features()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_features_vs_performance(df_perf, median_selected, ax1)
    plot_learning_curve(df_lc, ax2)
    plt.tight_layout()
    out_path = ANALYSIS_DIR / "nist_validation_plots.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return True


if __name__ == "__main__":
    main()
