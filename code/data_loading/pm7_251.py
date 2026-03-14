"""
Loader for PM7 semi-empirical data from the 251-molecule custom dataset.

Data file:
  - FINAL_PM7_DFT_master.csv: Combined PM7 neutral/protonated properties + metadata
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .base import BaseDatasetLoader
from .constants import HOF_PROTON_KCALMOL

logger = logging.getLogger(__name__)


class PM7_251Loader(BaseDatasetLoader):
    """
    Load PM7 data for the 251-molecule custom dataset.

    The master CSV contains per-site records with:
    - Neutral PM7 properties (HOF, dipole, HOMO, LUMO, gap, etc.)
    - Protonated PM7 properties (HOF, dipole, orbital energies, etc.)
    - Protonation site info (element, index)
    - Metadata (latent features from k-means, HBD, HBA, heavy atoms, MW)

    PM7 PA is calculated from heats of formation:
        PA = HOF(neutral) + HOF(H+) - HOF(protonated)
        where HOF(H+) = 365.7 kcal/mol

    Parameters
    ----------
    data_dir : str or Path
        Path to the 251_molecules directory.
    """

    # Column groups
    _NEUTRAL_PROPERTY_COLS = [
        "heat_of_formation_neutral",
        "dipole_x_neutral", "dipole_y_neutral", "dipole_z_neutral",
        "dipole_moment_neutral",
        "homo_ev_neutral", "lumo_ev_neutral", "gap_ev_neutral",
        "ionization_potential_neutral",
        "cosmo_area_neutral", "cosmo_volume_neutral",
        "molecular_weight_neutral",
        "total_energy_ev_neutral",
        "num_atoms_neutral",
    ]

    _PROTONATED_PROPERTY_COLS = [
        "heat_of_formation_protonated",
        "dipole_x_protonated", "dipole_y_protonated",
        "dipole_z_protonated", "dipole_moment_protonated",
        "homo_ev_protonated", "lumo_ev_protonated", "gap_ev_protonated",
        "ionization_potential_protonated",
        "cosmo_area_protonated", "cosmo_volume_protonated",
        "molecular_weight_protonated",
        "total_energy_ev_protonated",
        "num_atoms_protonated",
    ]

    _METADATA_COLS = [
        "original_index_Metadata",
        "latent_1_Metadata", "latent_2_Metadata", "latent_3_Metadata",
        "HBD_Metadata", "HBA_Metadata",
        "heavy_atoms_Metadata", "MW_Metadata",
    ]

    def __init__(self, data_dir: str | Path):
        super().__init__(data_dir)
        self.master_csv = self.data_dir / "pm7" / "FINAL_PM7_DFT_master.csv"

    def _required_files(self) -> list[Path]:
        return [self.master_csv]

    def load(self) -> pd.DataFrame:
        """
        Load the master CSV with all PM7 properties and compute PM7 PA.

        Returns
        -------
        pd.DataFrame
            Full dataset with PM7 properties, metadata, and computed pm7_pa_kcal.
        """
        df = pd.read_csv(self.master_csv)
        logger.info(f"Loaded {len(df)} site records from {self.master_csv.name} "
                     f"({len(df.columns)} columns)")

        # Compute PM7 PA from heats of formation
        df = self._compute_pm7_pa(df)

        # Compute delta properties (protonated - neutral)
        df = self._compute_delta_properties(df)

        return df

    def _compute_pm7_pa(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate PM7 proton affinity from heats of formation.

        PA = HOF(neutral) + HOF(H+) - HOF(protonated)
        """
        hof_neutral_col = "heat_of_formation_neutral"
        hof_prot_col = "heat_of_formation_protonated"

        if hof_neutral_col in df.columns and hof_prot_col in df.columns:
            df["pm7_pa_kcal"] = (
                df[hof_neutral_col] + HOF_PROTON_KCALMOL - df[hof_prot_col]
            )
            logger.info(
                f"Computed PM7 PA: mean={df['pm7_pa_kcal'].mean():.1f} ± "
                f"{df['pm7_pa_kcal'].std():.1f} kcal/mol"
            )
        else:
            logger.warning(f"Cannot compute PM7 PA: missing HOF columns")

        return df

    def _compute_delta_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add delta columns (protonated - neutral) for shared properties."""
        delta_pairs = [
            ("homo_ev_protonated", "homo_ev_neutral", "delta_homo_ev"),
            ("lumo_ev_protonated", "lumo_ev_neutral", "delta_lumo_ev"),
            ("gap_ev_protonated", "gap_ev_neutral", "delta_gap_ev"),
            ("dipole_moment_protonated", "dipole_moment_neutral", "delta_dipole_moment"),
            ("ionization_potential_protonated", "ionization_potential_neutral", "delta_ip"),
            ("cosmo_area_protonated", "cosmo_area_neutral", "delta_cosmo_area"),
            ("cosmo_volume_protonated", "cosmo_volume_neutral", "delta_cosmo_volume"),
        ]
        for prot_col, neut_col, delta_col in delta_pairs:
            if prot_col in df.columns and neut_col in df.columns:
                df[delta_col] = df[prot_col] - df[neut_col]
        return df

    def get_neutral_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Extract only neutral PM7 property columns."""
        if df is None:
            df = self.load()
        cols = [c for c in self._NEUTRAL_PROPERTY_COLS if c in df.columns]
        return df[cols].copy()

    def get_protonated_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Extract only protonated PM7 property columns."""
        if df is None:
            df = self.load()
        cols = [c for c in self._PROTONATED_PROPERTY_COLS if c in df.columns]
        return df[cols].copy()

    def get_metadata(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Extract metadata columns (latent features, molecular descriptors)."""
        if df is None:
            df = self.load()
        cols = [c for c in self._METADATA_COLS if c in df.columns]
        return df[cols].copy()

    def separate_features_and_targets(self, df: pd.DataFrame = None):
        """
        Split into features, PM7 PA target, and identifiers.

        Returns
        -------
        tuple of (X, y, identifiers)
            X : pd.DataFrame — neutral + protonated + delta features
            y : pd.Series — pm7_pa_kcal
            identifiers : pd.DataFrame — smiles and site info
        """
        if df is None:
            df = self.load()

        id_cols = ["neutral_smiles", "protonated_smiles",
                   "protonation_site_index_protonated",
                   "protonation_site_element_protonated",
                   "molecule_type_neutral", "molecule_type_protonated",
                   "chunk_neutral"]

        target_col = "pm7_pa_kcal"
        exclude = set(id_cols + [target_col] +
                      ["success_neutral", "input_charge_neutral"])

        # Also exclude non-numeric metadata for ML
        exclude.update(["point_group_neutral", "point_group_protonated",
                        "spin_state_neutral", "spin_state_protonated"])

        feature_cols = [c for c in df.columns if c not in exclude]
        existing_id_cols = [c for c in id_cols if c in df.columns]

        X = df[feature_cols].copy()
        y = df[target_col].copy() if target_col in df.columns else None
        identifiers = df[existing_id_cols].copy()

        logger.info(f"Features: {X.shape}, Target: {y.shape if y is not None else 'N/A'}")
        return X, y, identifiers

    def load_molecules(self) -> pd.DataFrame:
        """
        Get molecule-level summary (best PM7 PA per molecule).

        Returns
        -------
        pd.DataFrame
            One row per molecule with best-site properties and n_sites.
        """
        df = self.load()
        if "pm7_pa_kcal" not in df.columns:
            logger.error("pm7_pa_kcal not computed")
            return df

        idx_best = df.groupby("neutral_smiles")["pm7_pa_kcal"].idxmax()
        df_mol = df.loc[idx_best].copy()

        site_counts = df.groupby("neutral_smiles").size().rename("n_sites")
        df_mol = df_mol.merge(site_counts, left_on="neutral_smiles", right_index=True)

        return df_mol.reset_index(drop=True)
