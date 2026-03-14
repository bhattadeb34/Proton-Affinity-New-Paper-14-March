"""
Loader for B3LYP DFT results from the 251-molecule custom dataset.

Data format: .log files organized as mol_XXXXX/neutral/ and mol_XXXXX/site_N/
with site-specific proton affinity calculations.
"""
from __future__ import annotations

import os
import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseDatasetLoader
from .constants import HARTREE_TO_KCALMOL, H_PROTON_HA

logger = logging.getLogger(__name__)


class B3LYP251Loader(BaseDatasetLoader):
    """
    Load B3LYP/def2-TZVP DFT results for the 251-molecule custom dataset.

    Each molecule has a directory structure:
        mol_XXXXX/
            neutral/neutral.log
            site_0/protonated_site0.log
            site_1/protonated_site1.log
            ...

    PA is calculated from H(total) values:
        PA = (H_neutral + H_proton - H_protonated) * 627.509 kcal/mol

    Parameters
    ----------
    data_dir : str or Path
        Path to the 251_molecules directory (e.g., 'data/251_molecules').
    """

    def __init__(self, data_dir: str | Path):
        super().__init__(data_dir)
        self.b3lyp_dir = self.data_dir / "b3lyp_dft" / "Deb_B3LYP_dataset"

    def _required_files(self) -> list[Path]:
        return [self.b3lyp_dir]

    def load(self) -> pd.DataFrame:
        """Alias for load_sites(). Returns site-level data."""
        return self.load_sites()

    def load_sites(self) -> pd.DataFrame:
        """
        Load site-level B3LYP data (one row per protonation site).

        Returns
        -------
        pd.DataFrame
            Columns include: mol_id, mol_index, site_index, neutral_smiles,
            protonated_smiles, dft_pa_kcal, neutral/prot electronic properties,
            and delta properties.
        """
        mol_dirs = sorted([d for d in self.b3lyp_dir.glob("mol_*") if d.is_dir()])
        logger.info(f"Found {len(mol_dirs)} molecule directories in {self.b3lyp_dir}")

        site_data = []

        for mol_idx, mol_dir in enumerate(mol_dirs):
            if mol_idx % 50 == 0:
                logger.info(f"Processing molecule {mol_idx}/{len(mol_dirs)}...")

            mol_id = mol_dir.name

            # Parse neutral
            neutral_log = mol_dir / "neutral" / "neutral.log"
            neutral_data = self._parse_log_file(neutral_log)

            if not neutral_data["success"]:
                logger.warning(f"Failed to parse neutral for {mol_id}")
                continue

            H_neutral = neutral_data["H_total_Ha"]
            neutral_smiles = neutral_data["smiles"]

            # Parse each protonation site
            site_dirs = sorted([d for d in mol_dir.glob("site_*") if d.is_dir()])

            for site_dir in site_dirs:
                site_num = int(site_dir.name.split("_")[1])

                # Try different log file name patterns
                prot_log = self._find_log_file(site_dir, site_num)
                if prot_log is None:
                    logger.warning(f"No log file found in {site_dir}")
                    continue

                prot_data = self._parse_log_file(prot_log)
                if not prot_data["success"]:
                    logger.warning(f"Failed to parse {prot_log}")
                    continue

                H_prot = prot_data["H_total_Ha"]
                pa_kcal = self._calculate_pa(H_neutral, H_prot)

                if pa_kcal is None:
                    continue

                site_data.append({
                    "mol_id": mol_id,
                    "mol_index": mol_idx,
                    "site_index": site_num,
                    "neutral_smiles": neutral_smiles,
                    "protonated_smiles": prot_data["smiles"],
                    "H_neutral_Ha": H_neutral,
                    "H_protonated_Ha": H_prot,
                    "dft_pa_kcal": pa_kcal,
                    # Neutral properties
                    "neutral_E_elec_Ha": neutral_data["E_elec_Ha"],
                    "neutral_ZPE_kjmol": neutral_data["ZPE_kjmol"],
                    "neutral_HOMO_eV": neutral_data["HOMO_eV"],
                    "neutral_LUMO_eV": neutral_data["LUMO_eV"],
                    "neutral_gap_eV": neutral_data["gap_eV"],
                    "neutral_dipole_Debye": neutral_data["dipole_Debye"],
                    "neutral_n_imag": neutral_data["n_imag_freq"],
                    # Protonated properties
                    "prot_E_elec_Ha": prot_data["E_elec_Ha"],
                    "prot_ZPE_kjmol": prot_data["ZPE_kjmol"],
                    "prot_HOMO_eV": prot_data["HOMO_eV"],
                    "prot_LUMO_eV": prot_data["LUMO_eV"],
                    "prot_gap_eV": prot_data["gap_eV"],
                    "prot_dipole_Debye": prot_data["dipole_Debye"],
                    "prot_n_imag": prot_data["n_imag_freq"],
                })

        df = pd.DataFrame(site_data)

        # Compute delta properties
        df = self._compute_delta_properties(df)

        logger.info(
            f"Loaded {len(df)} sites from {df['mol_id'].nunique()} molecules"
        )
        return df

    def load_molecules(self) -> pd.DataFrame:
        """
        Load molecule-level data (best/highest PA per molecule).

        Returns
        -------
        pd.DataFrame
            One row per molecule, with the properties of the best
            protonation site and an 'n_sites' column.
        """
        df_sites = self.load_sites()
        if len(df_sites) == 0:
            return df_sites

        # Best site = highest PA per molecule
        idx_best = df_sites.groupby("mol_id")["dft_pa_kcal"].idxmax()
        df_mol = df_sites.loc[idx_best].copy()

        # Add site count
        site_counts = df_sites.groupby("mol_id").size().rename("n_sites")
        df_mol = df_mol.merge(site_counts, left_on="mol_id", right_index=True)

        logger.info(
            f"Molecule-level: {len(df_mol)} molecules, "
            f"PA range: {df_mol['dft_pa_kcal'].min():.1f} - {df_mol['dft_pa_kcal'].max():.1f} kcal/mol"
        )
        return df_mol

    # ---- Private helpers ----

    def _find_log_file(self, site_dir: Path, site_num: int) -> Path | None:
        """Try multiple naming patterns for protonated log files."""
        candidates = [
            site_dir / f"protonated_site{site_num}.log",
            site_dir / "protonated.log",
            site_dir / "site.log",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _parse_log_file(self, log_path: Path) -> dict:
        """
        Parse a B3LYP .log file and extract key properties.

        Parameters
        ----------
        log_path : Path
            Path to the log file.

        Returns
        -------
        dict
            Keys: H_total_Ha, E_elec_Ha, ZPE_kjmol, HOMO_eV, LUMO_eV,
                  gap_eV, dipole_Debye, n_imag_freq, smiles, success.
        """
        result = {
            "H_total_Ha": None,
            "E_elec_Ha": None,
            "ZPE_kjmol": None,
            "HOMO_eV": None,
            "LUMO_eV": None,
            "gap_eV": None,
            "dipole_Debye": None,
            "n_imag_freq": None,
            "smiles": None,
            "success": False,
        }

        if not log_path.exists():
            return result

        try:
            content = log_path.read_text()

            if "Normal termination" not in content:
                return result

            # Regex patterns for property extraction
            patterns = {
                "smiles": (r"SMILES:\s+(\S+)", str),
                "H_total_Ha": (r"H\(total\)\s+=\s+([-\d.]+)\s+Ha", float),
                "E_elec_Ha": (r"E\(elec\)\s+=\s+([-\d.]+)\s+Ha", float),
                "ZPE_kjmol": (r"ZPE\s+=\s+([-\d.]+)\s+kJ/mol", float),
                "HOMO_eV": (r"HOMO\s+=\s+([-\d.]+)\s+eV", float),
                "LUMO_eV": (r"LUMO\s+=\s+([-\d.]+)\s+eV", float),
                "gap_eV": (r"HOMO-LUMO gap\s+=\s+([-\d.]+)\s+eV", float),
                "dipole_Debye": (r"\|mu\|\s+=\s+([-\d.]+)\s+Debye", float),
                "n_imag_freq": (r"Imaginary frequencies:\s+(\d+)", int),
            }

            for key, (pattern, cast) in patterns.items():
                match = re.search(pattern, content)
                if match:
                    result[key] = cast(match.group(1))

            # Mark success if H(total) was found
            result["success"] = result["H_total_Ha"] is not None

        except Exception as e:
            logger.error(f"Error parsing {log_path}: {e}")

        return result

    @staticmethod
    def _calculate_pa(H_neutral_Ha: float, H_protonated_Ha: float) -> float | None:
        """
        Calculate proton affinity from H(total) values.

        PA = (H_neutral + H_proton - H_protonated) * 627.509 kcal/mol

        Parameters
        ----------
        H_neutral_Ha : float
            H(total) for neutral species in Hartree.
        H_protonated_Ha : float
            H(total) for protonated species in Hartree.

        Returns
        -------
        float or None
            PA in kcal/mol.
        """
        if H_neutral_Ha is None or H_protonated_Ha is None:
            return None
        return (H_neutral_Ha + H_PROTON_HA - H_protonated_Ha) * HARTREE_TO_KCALMOL

    @staticmethod
    def _compute_delta_properties(df: pd.DataFrame) -> pd.DataFrame:
        """Add delta columns (protonated - neutral) for electronic properties."""
        delta_pairs = [
            ("prot_HOMO_eV", "neutral_HOMO_eV", "delta_HOMO_eV"),
            ("prot_LUMO_eV", "neutral_LUMO_eV", "delta_LUMO_eV"),
            ("prot_gap_eV", "neutral_gap_eV", "delta_gap_eV"),
            ("prot_dipole_Debye", "neutral_dipole_Debye", "delta_dipole_Debye"),
            ("prot_ZPE_kjmol", "neutral_ZPE_kjmol", "delta_ZPE_kjmol"),
        ]
        for prot_col, neut_col, delta_col in delta_pairs:
            if prot_col in df.columns and neut_col in df.columns:
                df[delta_col] = df[prot_col] - df[neut_col]
        return df
