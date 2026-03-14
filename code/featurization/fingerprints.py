"""
Fingerprint computation: MACCS keys and Morgan circular fingerprints.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys

logger = logging.getLogger(__name__)


def compute_maccs(smiles_list: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute MACCS structural key fingerprints (167 bits).

    MACCS keys encode the presence of common chemical substructures,
    capturing functional groups and structural motifs.

    Parameters
    ----------
    smiles_list : list of str
        SMILES strings for each molecule.

    Returns
    -------
    X : np.ndarray of shape (n_molecules, 167)
        Binary fingerprint matrix.
    names : list of str
        Feature names ['MACCS_0', 'MACCS_1', ...].
    """
    logger.info(f"Computing MACCS fingerprints for {len(smiles_list)} molecules...")

    fps = []
    n_failed = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros(167, dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        else:
            fps.append(np.zeros(167, dtype=int))
            n_failed += 1

    if n_failed > 0:
        logger.warning(f"  MACCS: {n_failed} molecules failed to parse")

    X = np.array(fps)
    names = [f"MACCS_{i}" for i in range(167)]
    logger.info(f"  Generated {X.shape[1]} MACCS bits")
    return X, names


def compute_morgan(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 1024,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute Morgan circular fingerprints (ECFP-style).

    Morgan fingerprints with radius 2 encode the local chemical
    environment around each atom in the molecule.

    Parameters
    ----------
    smiles_list : list of str
        SMILES strings for each molecule.
    radius : int, default 2
        Fingerprint radius (2 → ECFP4 equivalent).
    n_bits : int, default 1024
        Number of fingerprint bits.

    Returns
    -------
    X : np.ndarray of shape (n_molecules, n_bits)
        Fingerprint matrix (bit vector).
    names : list of str
        Feature names ['Morgan_0', 'Morgan_1', ...].
    """
    logger.info(
        f"Computing Morgan fingerprints (r={radius}, {n_bits} bits) "
        f"for {len(smiles_list)} molecules..."
    )

    fps = []
    n_failed = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros(n_bits, dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        else:
            fps.append(np.zeros(n_bits, dtype=int))
            n_failed += 1

    if n_failed > 0:
        logger.warning(f"  Morgan: {n_failed} molecules failed to parse")

    X = np.array(fps)
    names = [f"Morgan_{i}" for i in range(n_bits)]
    logger.info(f"  Generated {X.shape[1]} Morgan bits")
    return X, names
