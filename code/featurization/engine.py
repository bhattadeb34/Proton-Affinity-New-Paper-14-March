"""
Unified feature engineering engine.

Assembles all feature categories into a single matrix,
with a consistent interface for both datasets.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from .fingerprints import compute_maccs, compute_morgan
from .descriptors import compute_rdkit_delta, compute_3d_descriptors
from .quantum_features import compute_pm7_features
from .site_features import compute_site_features

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Unified feature engineering for proton affinity prediction.

    Computes all feature categories described in the paper and
    assembles them into a single feature matrix. Works for both
    the 1185-molecule and 251-molecule datasets.

    Feature categories:
        1. MACCS keys (167 bits)
        2. Morgan fingerprints (1024 bits, radius 2)
        3. RDKit 2D descriptors (210 × 3 states = 630)
        4. PM7 quantum properties (13 × 3 states = 39)
        5. Mordred descriptors (optional, from pre-computed CSV)
        6. 3D shape descriptors (optional, 10 features)
        7. Site features (optional, ~6 features, for 251-dataset)

    Usage
    -----
    >>> engine = FeatureEngine()
    >>> X, names, sources = engine.build_features(
    ...     smiles_neutral=neutral_smiles,
    ...     smiles_protonated=prot_smiles,
    ...     pm7_df=pm7_data
    ... )
    """

    def __init__(self, morgan_radius: int = 2, morgan_bits: int = 1024):
        """
        Parameters
        ----------
        morgan_radius : int, default 2
            Morgan fingerprint radius.
        morgan_bits : int, default 1024
            Number of Morgan fingerprint bits.
        """
        self.morgan_radius = morgan_radius
        self.morgan_bits = morgan_bits

        # Populated after build_features()
        self.feature_names_: List[str] = []
        self.feature_sources_: List[str] = []
        self.feature_counts_: dict = {}

    def build_features(
        self,
        smiles_neutral: List[str],
        smiles_protonated: List[str] = None,
        pm7_df: pd.DataFrame = None,
        pm7_neutral_suffix: str = "_neutral",
        pm7_prot_suffix: str = "_protonated",
        mordred_df: pd.DataFrame = None,
        mordred_prefix: str = "Mordred_",
        compute_3d: bool = False,
        site_elements: pd.Series = None,
        site_indices: pd.Series = None,
        mol_ids: pd.Series = None,
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Build the complete feature matrix from all requested categories.

        Parameters
        ----------
        smiles_neutral : list of str
            Neutral molecule SMILES (required).
        smiles_protonated : list of str, optional
            Protonated molecule SMILES. If provided, computes delta
            features for RDKit descriptors (3 states).
        pm7_df : pd.DataFrame, optional
            DataFrame with PM7 property columns for quantum features.
        pm7_neutral_suffix : str, default '_neutral'
            Column suffix for neutral PM7 properties.
        pm7_prot_suffix : str, default '_protonated'
            Column suffix for protonated PM7 properties.
        mordred_df : pd.DataFrame, optional
            Pre-computed Mordred descriptors DataFrame (columns are features).
        mordred_prefix : str, default 'Mordred_'
            Prefix to add to Mordred column names.
        compute_3d : bool, default False
            Whether to compute 3D shape descriptors (slow).
        site_elements : pd.Series, optional
            Protonation site elements (e.g., 'N', 'O', 'S').
        site_indices : pd.Series, optional
            Integer index of each protonation site.
        mol_ids : pd.Series, optional
            Molecule IDs for grouping site features.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            Complete feature matrix.
        names : list of str
            Feature names.
        sources : list of str
            Feature source category for each feature.
        """
        n_samples = len(smiles_neutral)
        all_X = []
        all_names = []
        all_sources = []

        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING")
        logger.info("=" * 60)

        # ---- 1. MACCS fingerprints (167) ----
        X_maccs, names_maccs = compute_maccs(smiles_neutral)
        all_X.append(X_maccs)
        all_names.extend(names_maccs)
        all_sources.extend(["MACCS"] * len(names_maccs))

        # ---- 2. Morgan fingerprints (1024) ----
        X_morgan, names_morgan = compute_morgan(
            smiles_neutral, radius=self.morgan_radius, n_bits=self.morgan_bits
        )
        all_X.append(X_morgan)
        all_names.extend(names_morgan)
        all_sources.extend(["Morgan"] * len(names_morgan))

        # ---- 3. RDKit descriptors (210 × 3 or 210 × 1) ----
        if smiles_protonated is not None:
            X_rdkit, names_rdkit = compute_rdkit_delta(smiles_neutral, smiles_protonated)
        else:
            from .descriptors import compute_rdkit_single_state
            X_rdkit, names_rdkit_raw = compute_rdkit_single_state(smiles_neutral)
            names_rdkit = [f"RDKit_{n}" for n in names_rdkit_raw]
        all_X.append(X_rdkit)
        all_names.extend(names_rdkit)
        all_sources.extend(["RDKit"] * len(names_rdkit))

        # ---- 4. PM7 quantum features (13 × 3 = 39) ----
        if pm7_df is not None:
            X_pm7, names_pm7 = compute_pm7_features(
                pm7_df,
                neutral_suffix=pm7_neutral_suffix,
                prot_suffix=pm7_prot_suffix,
            )
            all_X.append(X_pm7)
            all_names.extend(names_pm7)
            all_sources.extend(["PM7"] * len(names_pm7))

        # ---- 5. Mordred descriptors (pre-computed, optional) ----
        if mordred_df is not None:
            X_mordred, names_mordred = self._extract_mordred(mordred_df, mordred_prefix)
            all_X.append(X_mordred)
            all_names.extend(names_mordred)
            all_sources.extend(["Mordred"] * len(names_mordred))

        # ---- 6. 3D descriptors (optional, 10) ----
        if compute_3d:
            X_3d, names_3d = compute_3d_descriptors(smiles_neutral)
            all_X.append(X_3d)
            all_names.extend(names_3d)
            all_sources.extend(["3D"] * len(names_3d))

        # ---- 7. Site features (optional, ~6) ----
        if site_elements is not None and site_indices is not None and mol_ids is not None:
            X_site, names_site = compute_site_features(site_elements, site_indices, mol_ids)
            all_X.append(X_site)
            all_names.extend(names_site)
            all_sources.extend(["Site"] * len(names_site))

        # ---- Assemble ----
        X = np.hstack(all_X)
        X = np.nan_to_num(X)

        # Store metadata
        self.feature_names_ = all_names
        self.feature_sources_ = all_sources
        self.feature_counts_ = dict(Counter(all_sources))

        # Summary
        logger.info("\nFeature Summary:")
        for source, count in self.feature_counts_.items():
            logger.info(f"  {source}: {count} features")
        logger.info(f"  TOTAL: {len(all_names)} features")

        return X, all_names, all_sources

    def get_feature_summary(self) -> dict:
        """
        Return a summary of the last build_features() call.

        Returns
        -------
        dict
            Keys: total, per_source counts, feature_names.
        """
        return {
            "total": len(self.feature_names_),
            "per_source": self.feature_counts_.copy(),
            "feature_names": self.feature_names_.copy(),
        }

    @staticmethod
    def _extract_mordred(
        mordred_df: pd.DataFrame,
        prefix: str = "Mordred_",
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract Mordred descriptors from a pre-computed DataFrame.

        Parameters
        ----------
        mordred_df : pd.DataFrame
            DataFrame where each column is a Mordred descriptor.
            Non-numeric columns are automatically excluded.
        prefix : str, default 'Mordred_'
            Prefix for feature names.

        Returns
        -------
        X : np.ndarray
            Mordred feature matrix.
        names : list of str
            Feature names.
        """
        # Select only numeric columns
        numeric_cols = mordred_df.select_dtypes(include=[np.number]).columns.tolist()

        X = mordred_df[numeric_cols].values.astype(float)
        X = np.nan_to_num(X)

        names = [f"{prefix}{col}" for col in numeric_cols]
        logger.info(f"  Extracted {len(names)} Mordred features")
        return X, names
