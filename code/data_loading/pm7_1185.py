"""
Loader for PM7 semi-empirical data from the 1185-molecule NIST dataset.

Data files:
  - Dataset.csv: Mordred descriptors + PM7 electronic properties + fingerprints + EXP_PA
  - FINAL_PM7_ALL_proton_affinities.csv: Site-level PM7 PA with detailed properties
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .base import BaseDatasetLoader

logger = logging.getLogger(__name__)


class PM7_1185Loader(BaseDatasetLoader):
    """
    Load PM7 data for the 1185-molecule NIST dataset.

    Primary file (Dataset.csv) contains:
    - ~100 Mordred molecular descriptors
    - PM7 electronic properties (Ehomo, Elumo, chemical_potential, hardness, etc.)
    - ~80 fingerprint bits (fp_*)
    - Identifiers: smiles, reg_num
    - Target: EXP_PA (experimental proton affinity in kJ/mol)

    Secondary file (FINAL_PM7_ALL_proton_affinities.csv) contains:
    - Site-level PM7 proton affinities with detailed neutral/protonated properties

    Parameters
    ----------
    data_dir : str or Path
        Path to the 1185_molecules directory.
    """

    # Column groups for separating features
    _PM7_ELECTRONIC_COLS = [
        "Ehomo", "Elumo", "chemical potential", "hardness",
        "MK_charge", "Dipole moment", "CM5_charge",
    ]
    _IDENTIFIER_COLS = ["smiles", "reg_num"]
    _TARGET_COL = "EXP_PA"

    def __init__(self, data_dir: str | Path):
        super().__init__(data_dir)
        self.dataset_csv = self.data_dir / "pm7" / "Dataset.csv"
        self.pa_csv = self.data_dir / "pm7" / "FINAL_PM7_ALL_proton_affinities.csv"

    def _required_files(self) -> list[Path]:
        return [self.dataset_csv]

    def load(self) -> pd.DataFrame:
        """
        Load the main Dataset.csv with all features and the experimental PA target.

        Returns
        -------
        pd.DataFrame
            Full dataset with Mordred + PM7 + fingerprint features, smiles, and EXP_PA.
        """
        df = pd.read_csv(self.dataset_csv)
        logger.info(f"Loaded {len(df)} molecules from {self.dataset_csv.name} "
                     f"({len(df.columns)} columns)")
        return df

    def load_proton_affinities(self) -> pd.DataFrame:
        """
        Load site-level PM7 proton affinities with detailed properties.

        Returns
        -------
        pd.DataFrame
            Site-level data with neutral/protonated properties and PM7 PA.
        """
        if not self.pa_csv.exists():
            raise FileNotFoundError(f"PA file not found: {self.pa_csv}")

        df = pd.read_csv(self.pa_csv)
        logger.info(f"Loaded {len(df)} site-level records from {self.pa_csv.name}")
        return df

    def separate_features_and_targets(self, df: pd.DataFrame = None):
        """
        Split the dataset into feature matrix, target, and identifiers.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Pre-loaded DataFrame. If None, calls self.load().

        Returns
        -------
        tuple of (X, y, identifiers)
            X : pd.DataFrame — all feature columns
            y : pd.Series — EXP_PA values (kJ/mol)
            identifiers : pd.DataFrame — smiles and reg_num
        """
        if df is None:
            df = self.load()

        id_cols = [c for c in self._IDENTIFIER_COLS if c in df.columns]
        target_col = self._TARGET_COL

        feature_cols = [
            c for c in df.columns
            if c not in id_cols and c != target_col
        ]

        X = df[feature_cols].copy()
        y = df[target_col].copy()
        identifiers = df[id_cols].copy()

        logger.info(f"Features: {X.shape}, Target: {y.shape}")
        return X, y, identifiers

    def get_pm7_electronic_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract only PM7 electronic structure features.

        Returns
        -------
        pd.DataFrame
            Columns: Ehomo, Elumo, chemical_potential, hardness,
                     MK_charge, Dipole_moment, CM5_charge.
        """
        if df is None:
            df = self.load()
        cols = [c for c in self._PM7_ELECTRONIC_COLS if c in df.columns]
        return df[cols].copy()

    def get_mordred_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract only Mordred molecular descriptor features.

        Returns columns that are NOT fingerprints, NOT PM7 electronic,
        NOT identifiers, and NOT the target.
        """
        if df is None:
            df = self.load()

        exclude = set(self._PM7_ELECTRONIC_COLS + self._IDENTIFIER_COLS + [self._TARGET_COL])
        fp_cols = {c for c in df.columns if c.startswith("fp_")}
        exclude.update(fp_cols)

        mordred_cols = [c for c in df.columns if c not in exclude]
        return df[mordred_cols].copy()

    def get_fingerprints(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract only fingerprint columns (fp_*).

        Returns
        -------
        pd.DataFrame
            Binary fingerprint features.
        """
        if df is None:
            df = self.load()
        fp_cols = [c for c in df.columns if c.startswith("fp_")]
        return df[fp_cols].copy()

    def get_best_pm7_pa(self) -> pd.DataFrame:
        """
        Get molecule-level best (highest) PM7 PA for each molecule.

        Returns
        -------
        pd.DataFrame
            Columns: neutral_smiles, pm7_pa_kcal (best site PA).
        """
        df_pa = self.load_proton_affinities()

        # Get best PA per molecule (highest)
        idx_best = df_pa.groupby("neutral_smiles")["proton_affinity_kcal_mol"].idxmax()
        df_best = df_pa.loc[idx_best, ["neutral_smiles", "proton_affinity_kcal_mol"]].copy()
        df_best = df_best.rename(columns={"proton_affinity_kcal_mol": "pm7_pa_kcal"})

        return df_best.reset_index(drop=True)
