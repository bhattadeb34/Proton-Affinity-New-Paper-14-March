"""
Loader for B3LYP DFT results from the 1185-molecule NIST dataset.

Data format: JSON files (one per molecule) containing DFT-calculated
proton affinities and molecular properties.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseDatasetLoader
from .constants import KJMOL_TO_KCALMOL

logger = logging.getLogger(__name__)


class B3LYP1185Loader(BaseDatasetLoader):
    """
    Load B3LYP/def2-TZVP DFT results for the 1185-molecule NIST dataset.

    Each molecule has a JSON file containing:
    - Experimental PA (from NIST, kJ/mol)
    - DFT PA (B3LYP/def2-TZVP with full thermochemistry, kJ/mol)
    - Neutral and protonated molecular properties
    - Site information

    Parameters
    ----------
    data_dir : str or Path
        Path to the 1185_molecules directory (e.g., 'data/1185_molecules').
    """

    def __init__(self, data_dir: str | Path):
        super().__init__(data_dir)
        self.json_dir = self.data_dir / "b3lyp_dft" / "results"

    def _required_files(self) -> list[Path]:
        return [self.json_dir]

    def load(self) -> pd.DataFrame:
        """
        Load all B3LYP JSON files and return a standardized DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per molecule with columns:
            - Identifiers: global_idx, smiles, json_file
            - PA values: exp_pa_kjmol, exp_pa_kcal, dft_pa_kjmol, dft_pa_kcal,
                         error_kjmol, error_kcal
            - Status: status, n_sites, best_site
            - Neutral properties: neutral_E_elec_Ha, neutral_H_total_Ha, etc.
            - Protonated properties: prot_E_elec_Ha, prot_H_total_Ha, etc.
            - Delta properties: delta_HOMO_eV, delta_LUMO_eV, etc.
        """
        json_files = sorted(self.json_dir.glob("mol_*.json"))
        if not json_files:
            # Try alternate pattern (flat directory)
            json_files = sorted(self.json_dir.glob("*.json"))

        logger.info(f"Found {len(json_files)} JSON files in {self.json_dir}")

        results = []
        failed = []

        for json_path in json_files:
            try:
                data = self._parse_json(json_path)
                mol_data = self._extract_molecule_data(data)
                mol_data["json_file"] = json_path.name
                results.append(mol_data)
            except Exception as e:
                failed.append({"file": json_path.name, "error": str(e)})
                logger.warning(f"Failed to parse {json_path.name}: {e}")

        logger.info(f"Successfully loaded: {len(results)}, Failed: {len(failed)}")

        df = pd.DataFrame(results)

        # Compute delta properties
        df = self._compute_delta_properties(df)

        return df

    def _parse_json(self, json_path: Path) -> dict:
        """Load and parse a single B3LYP JSON file."""
        with open(json_path, "r") as f:
            return json.load(f)

    def _extract_molecule_data(self, data: dict) -> dict:
        """
        Extract relevant fields from a parsed JSON dict.

        Parameters
        ----------
        data : dict
            Parsed JSON data for one molecule.

        Returns
        -------
        dict
            Flat dictionary of extracted properties.
        """
        mol = {
            "global_idx": data.get("global_idx"),
            "smiles": data.get("smiles"),
            "exp_pa_kjmol": data.get("exp_pa"),
            "dft_pa_kjmol": data.get("dft_pa"),
            "error_kjmol": data.get("error"),
            "status": data.get("status"),
            "n_sites": data.get("n_sites"),
            "best_site": data.get("best_site"),
        }

        # Convert kJ/mol → kcal/mol
        for src, dst in [
            ("exp_pa_kjmol", "exp_pa_kcal"),
            ("dft_pa_kjmol", "dft_pa_kcal"),
            ("error_kjmol", "error_kcal"),
        ]:
            mol[dst] = mol[src] * KJMOL_TO_KCALMOL if mol[src] is not None else None

        # Neutral properties
        neutral = data.get("neutral", {})
        for key, col in [
            ("E_elec", "neutral_E_elec_Ha"),
            ("H_total", "neutral_H_total_Ha"),
            ("ZPE_kjmol", "neutral_ZPE_kjmol"),
            ("HOMO_eV", "neutral_HOMO_eV"),
            ("LUMO_eV", "neutral_LUMO_eV"),
            ("HOMO_LUMO_gap_eV", "neutral_gap_eV"),
            ("dipole_debye", "neutral_dipole_debye"),
            ("n_atoms", "neutral_n_atoms"),
            ("n_imaginary", "neutral_n_imaginary"),
        ]:
            mol[col] = neutral.get(key)

        # Protonated properties (best site)
        prot = data.get("protonated_best", {})
        for key, col in [
            ("E_elec", "prot_E_elec_Ha"),
            ("H_total", "prot_H_total_Ha"),
            ("ZPE_kjmol", "prot_ZPE_kjmol"),
            ("HOMO_eV", "prot_HOMO_eV"),
            ("LUMO_eV", "prot_LUMO_eV"),
            ("HOMO_LUMO_gap_eV", "prot_gap_eV"),
            ("dipole_debye", "prot_dipole_debye"),
            ("n_atoms", "prot_n_atoms"),
            ("n_imaginary", "prot_n_imaginary"),
            ("smiles", "prot_smiles"),
        ]:
            mol[col] = prot.get(key)

        return mol

    def _compute_delta_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add delta columns (protonated - neutral) for key electronic properties.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with neutral_* and prot_* columns.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with additional delta_* columns.
        """
        delta_pairs = [
            ("prot_HOMO_eV", "neutral_HOMO_eV", "delta_HOMO_eV"),
            ("prot_LUMO_eV", "neutral_LUMO_eV", "delta_LUMO_eV"),
            ("prot_gap_eV", "neutral_gap_eV", "delta_gap_eV"),
            ("prot_dipole_debye", "neutral_dipole_debye", "delta_dipole_debye"),
            ("prot_ZPE_kjmol", "neutral_ZPE_kjmol", "delta_ZPE_kjmol"),
        ]
        for prot_col, neut_col, delta_col in delta_pairs:
            if prot_col in df.columns and neut_col in df.columns:
                df[delta_col] = df[prot_col] - df[neut_col]

        return df

    def get_successful(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Return only molecules with successful DFT calculations.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Pre-loaded DataFrame. If None, calls self.load().

        Returns
        -------
        pd.DataFrame
            Filtered to status == 'OK'.
        """
        if df is None:
            df = self.load()
        return df[df["status"] == "OK"].copy()
