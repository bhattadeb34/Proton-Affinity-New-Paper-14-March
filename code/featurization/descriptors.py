"""
Molecular descriptor computation: RDKit 2D descriptors and 3D shape descriptors.
"""
from __future__ import annotations

import math
import logging
from typing import List, Tuple

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

logger = logging.getLogger(__name__)


def _safe_float(x, default: float = 0.0) -> float:
    """Convert value to float, returning default on failure or NaN."""
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def _rdkit_descriptor_names() -> List[str]:
    """Get all RDKit 2D descriptor names (~210 descriptors)."""
    return [name for name, _ in Descriptors._descList]


def _rdkit_descriptor_values(mol) -> List[float]:
    """Calculate all RDKit 2D descriptors for a molecule."""
    vals = []
    for _, func in Descriptors._descList:
        try:
            val = func(mol)
            vals.append(_safe_float(val))
        except Exception:
            vals.append(0.0)
    return vals


def compute_rdkit_single_state(smiles_list: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute RDKit 2D descriptors for a single molecular state.

    Parameters
    ----------
    smiles_list : list of str
        SMILES strings.

    Returns
    -------
    X : np.ndarray of shape (n_molecules, ~210)
        Descriptor matrix.
    names : list of str
        Descriptor names.
    """
    desc_names = _rdkit_descriptor_names()
    logger.info(f"Computing {len(desc_names)} RDKit descriptors for {len(smiles_list)} molecules...")

    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            rows.append(_rdkit_descriptor_values(mol))
        else:
            rows.append([0.0] * len(desc_names))

    X = np.nan_to_num(np.array(rows, dtype=float))
    return X, desc_names


def compute_rdkit_delta(
    smiles_neutral: List[str],
    smiles_protonated: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute RDKit 2D descriptors for neutral, protonated, and delta states.

    Generates 3× the number of base descriptors (~210 × 3 = ~630 features).

    Parameters
    ----------
    smiles_neutral : list of str
        Neutral molecule SMILES.
    smiles_protonated : list of str
        Protonated molecule SMILES.

    Returns
    -------
    X : np.ndarray of shape (n_molecules, ~630)
        Concatenated [neutral, protonated, delta] descriptors.
    names : list of str
        Feature names with _Neu, _Prot, _Delta suffixes.
    """
    desc_names = _rdkit_descriptor_names()
    n_desc = len(desc_names)

    logger.info(
        f"Computing RDKit descriptors (neutral + protonated + delta) "
        f"for {len(smiles_neutral)} molecules..."
    )

    vals_neu, vals_prot = [], []

    for i, (s_n, s_p) in enumerate(zip(smiles_neutral, smiles_protonated)):
        if i > 0 and i % 200 == 0:
            logger.info(f"  Processing molecule {i}/{len(smiles_neutral)}...")

        mol_n = Chem.MolFromSmiles(str(s_n))
        mol_p = Chem.MolFromSmiles(str(s_p))

        vals_neu.append(_rdkit_descriptor_values(mol_n) if mol_n else [0.0] * n_desc)
        vals_prot.append(_rdkit_descriptor_values(mol_p) if mol_p else [0.0] * n_desc)

    neu_arr = np.array(vals_neu, dtype=float)
    prot_arr = np.array(vals_prot, dtype=float)
    delta_arr = prot_arr - neu_arr

    X = np.hstack([neu_arr, prot_arr, delta_arr])
    X = np.nan_to_num(X)

    names = (
        [f"RDKit_{n}_Neu" for n in desc_names]
        + [f"RDKit_{n}_Prot" for n in desc_names]
        + [f"RDKit_{n}_Delta" for n in desc_names]
    )

    logger.info(f"  Generated {len(names)} RDKit features ({n_desc} × 3 states)")
    return X, names


def compute_3d_descriptors(smiles_list: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute 3D shape descriptors from conformer generation.

    Uses ETKDG conformer generation + MMFF optimization for
    radius of gyration, principal moments, spherocity, etc.

    Parameters
    ----------
    smiles_list : list of str
        SMILES strings.

    Returns
    -------
    X : np.ndarray of shape (n_molecules, 10)
        3D descriptor matrix.
    names : list of str
        Feature names ['3D_RadGyration', '3D_PMI1', ...].
    """
    desc_3d_names = [
        "RadGyration", "PMI1", "PMI2", "PMI3", "Spherocity",
        "Asphericity", "Eccentricity", "InertialShapeFactor", "NPR1", "NPR2",
    ]
    n_3d = len(desc_3d_names)

    logger.info(f"Computing 3D descriptors for {len(smiles_list)} molecules...")

    rows = []
    n_failed = 0
    for i, smi in enumerate(smiles_list):
        if i > 0 and i % 200 == 0:
            logger.info(f"  Processing molecule {i}/{len(smiles_list)}...")

        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            try:
                m = Chem.AddHs(mol)
                AllChem.EmbedMolecule(m, AllChem.ETKDGv2())
                AllChem.MMFFOptimizeMolecule(m)
                vals = [
                    rdMolDescriptors.CalcRadiusOfGyration(m),
                    rdMolDescriptors.CalcPMI1(m),
                    rdMolDescriptors.CalcPMI2(m),
                    rdMolDescriptors.CalcPMI3(m),
                    rdMolDescriptors.CalcSpherocityIndex(m),
                    rdMolDescriptors.CalcAsphericity(m),
                    rdMolDescriptors.CalcEccentricity(m),
                    rdMolDescriptors.CalcInertialShapeFactor(m),
                    rdMolDescriptors.CalcNPR1(m),
                    rdMolDescriptors.CalcNPR2(m),
                ]
                rows.append([_safe_float(v) for v in vals])
            except Exception:
                rows.append([0.0] * n_3d)
                n_failed += 1
        else:
            rows.append([0.0] * n_3d)
            n_failed += 1

    if n_failed > 0:
        logger.warning(f"  3D: {n_failed} molecules failed")

    X = np.array(rows, dtype=float)
    names = [f"3D_{n}" for n in desc_3d_names]
    logger.info(f"  Generated {len(names)} 3D features")
    return X, names
