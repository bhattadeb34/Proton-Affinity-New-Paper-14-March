"""
Site-level features for protonation site characterization.

For the 251-molecule dataset, each protonation site is described by:
- One-hot encoded element type (N, O, S)
- Site index within the molecule
- Normalized site index (0-1)
- Total number of protonation sites per molecule
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard protonation site elements
SITE_ELEMENTS = ["N", "O", "S"]


def compute_site_features(
    site_elements: pd.Series,
    site_indices: pd.Series,
    mol_ids: pd.Series,
    element_categories: List[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute site-specific features encoding protonation site characteristics.

    Parameters
    ----------
    site_elements : pd.Series
        Element type at each protonation site (e.g., 'N', 'O', 'S').
    site_indices : pd.Series
        Index of the protonation site within the molecule.
    mol_ids : pd.Series
        Molecule ID for grouping (to compute n_sites and normalized index).
    element_categories : list of str, optional
        Elements for one-hot encoding. Defaults to ['N', 'O', 'S'].

    Returns
    -------
    X : np.ndarray of shape (n_sites, n_features)
        Site feature matrix (typically 6 features).
    names : list of str
        Feature names.
    """
    if element_categories is None:
        element_categories = SITE_ELEMENTS

    logger.info(f"Computing site features for {len(site_elements)} sites...")

    elements = site_elements.fillna("Unknown").values
    indices = site_indices.values.astype(float)

    # 1. One-hot encode element type
    ohe_features = []
    ohe_names = []
    for elem in element_categories:
        ohe_features.append((elements == elem).astype(int))
        ohe_names.append(f"Site_elem_{elem}")

    # 2. Raw site index
    site_idx = indices

    # 3. Number of sites per molecule
    mol_site_counts = mol_ids.groupby(mol_ids).transform("count").values.astype(float)

    # 4. Normalized site index (0-1 range within each molecule)
    site_idx_norm = indices / np.maximum(mol_site_counts, 1.0)

    # Stack all features
    X = np.column_stack([
        *ohe_features,
        site_idx,
        site_idx_norm,
        mol_site_counts,
    ])

    names = ohe_names + ["Site_index", "Site_index_norm", "Mol_n_sites"]

    logger.info(
        f"  Generated {len(names)} site features "
        f"({len(element_categories)} elements + 3 index features)"
    )
    return X, names
