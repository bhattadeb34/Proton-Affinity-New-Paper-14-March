"""
Featurization module for proton affinity prediction.

Provides a unified FeatureEngine and standalone compute functions
for all feature categories described in the paper.

Usage
-----
>>> from code.featurization import FeatureEngine
>>> engine = FeatureEngine()
>>> X, names, sources = engine.build_features(
...     smiles_neutral=['CCO', 'CC(=O)O'],
...     smiles_protonated=['CC[OH2+]', 'CC(=O)[OH2+]'],
... )
"""

from .engine import FeatureEngine
from .fingerprints import compute_maccs, compute_morgan
from .descriptors import compute_rdkit_delta, compute_rdkit_single_state, compute_3d_descriptors
from .quantum_features import compute_pm7_features, PM7_PROPERTIES
from .site_features import compute_site_features, SITE_ELEMENTS

__all__ = [
    "FeatureEngine",
    "compute_maccs",
    "compute_morgan",
    "compute_rdkit_delta",
    "compute_rdkit_single_state",
    "compute_3d_descriptors",
    "compute_pm7_features",
    "compute_site_features",
    "PM7_PROPERTIES",
    "SITE_ELEMENTS",
]
