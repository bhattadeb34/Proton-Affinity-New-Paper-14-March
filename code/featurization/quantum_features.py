"""
PM7 semi-empirical quantum chemical feature extraction.

Extracts 13 electronic structure properties × 3 states
(neutral, protonated, delta) = 39 features total.
"""
from __future__ import annotations

import math
import logging
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The 13 PM7 properties described in the paper
PM7_PROPERTIES = [
    "heat_of_formation",
    "dipole_x",
    "dipole_y",
    "dipole_z",
    "dipole_moment",
    "homo_ev",
    "lumo_ev",
    "gap_ev",
    "ionization_potential",
    "cosmo_area",
    "cosmo_volume",
    "total_energy",
    "num_atoms",
]

# Alternate column name patterns for properties that differ across datasets
# Maps base name → list of alternate base names to try
_PROPERTY_ALIASES = {
    "total_energy": ["total_energy_ev", "total_energy_kcal_mol"],
}


def _safe_float(x, default: float = 0.0) -> float:
    """Convert value to float, returning default on failure or NaN."""
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def compute_pm7_features(
    df: pd.DataFrame,
    neutral_suffix: str = "_neutral",
    prot_suffix: str = "_protonated",
    properties: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract PM7 quantum features for neutral, protonated, and delta states.

    The function looks for columns named '{property}{suffix}' in the DataFrame.
    It handles different column naming conventions across the two datasets
    via the suffix parameters and property aliases:
        - 1185 dataset (merged neutral+protonated files): suffix='_n' / '_p'
        - 251 dataset (FINAL_PM7_DFT_master.csv): suffix='_neutral' / '_protonated'

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing PM7 property columns.
    neutral_suffix : str, default '_neutral'
        Suffix for neutral property columns.
    prot_suffix : str, default '_protonated'
        Suffix for protonated property columns.
    properties : list of str, optional
        PM7 properties to extract. Defaults to all 13 standard properties.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_properties * 3)
        Feature matrix [neutral, protonated, delta] for each property.
    names : list of str
        Feature names with _Neu, _Prot, _Delta suffixes.
    """
    if properties is None:
        properties = PM7_PROPERTIES

    logger.info(f"Extracting PM7 quantum features ({len(properties)} props × 3 states)...")

    # Resolve actual column names for each property
    # Returns list of (prop_name, col_neutral, col_protonated) tuples
    resolved = []
    for prop in properties:
        col_pair = _resolve_columns(df, prop, neutral_suffix, prot_suffix)
        if col_pair is not None:
            resolved.append((prop, col_pair[0], col_pair[1]))
        else:
            logger.debug(f"  PM7 property '{prop}' not found in DataFrame")

    if not resolved:
        logger.warning("No PM7 properties found in DataFrame!")
        return np.zeros((len(df), 0)), []

    # Extract feature values
    rows = []
    for _, row in df.iterrows():
        feat = []
        for prop, col_n, col_p in resolved:
            val_n = _safe_float(row.get(col_n, 0.0))
            val_p = _safe_float(row.get(col_p, 0.0))
            feat.extend([val_n, val_p, val_p - val_n])
        rows.append(feat)

    X = np.nan_to_num(np.array(rows, dtype=float))

    names = []
    for prop, _, _ in resolved:
        names.extend([
            f"PM7_{prop}_Neu",
            f"PM7_{prop}_Prot",
            f"PM7_{prop}_Delta",
        ])

    logger.info(f"  Generated {len(names)} PM7 features ({len(resolved)} props × 3)")
    return X, names


def _resolve_columns(
    df: pd.DataFrame,
    prop: str,
    neutral_suffix: str,
    prot_suffix: str,
) -> Optional[Tuple[str, str]]:
    """
    Resolve the actual column names for a PM7 property.

    Tries the primary suffixes first, then alternate suffixes,
    then property aliases with all suffix combinations.

    Returns
    -------
    tuple of (col_neutral, col_protonated) or None if not found.
    """
    all_suffixes = [
        (neutral_suffix, prot_suffix),
        ("_n", "_p"),
        ("_neutral", "_protonated"),
    ]

    # All base names to try: original + aliases
    base_names = [prop] + _PROPERTY_ALIASES.get(prop, [])

    for base in base_names:
        for n_suf, p_suf in all_suffixes:
            col_n = f"{base}{n_suf}"
            col_p = f"{base}{p_suf}"
            if col_n in df.columns and col_p in df.columns:
                return (col_n, col_p)

    return None
