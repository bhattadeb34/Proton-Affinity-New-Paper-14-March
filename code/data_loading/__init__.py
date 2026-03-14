"""
Data loading module for proton affinity datasets.

Provides standardized loaders for:
- B3LYP DFT results (1185-molecule and 251-molecule datasets)
- PM7 semi-empirical results (1185-molecule and 251-molecule datasets)

Usage
-----
>>> from code.data_loading import B3LYP1185Loader, PM7_1185Loader
>>> b3lyp = B3LYP1185Loader("data/1185_molecules")
>>> df = b3lyp.load()
"""

from .b3lyp_1185 import B3LYP1185Loader
from .b3lyp_251 import B3LYP251Loader
from .pm7_1185 import PM7_1185Loader
from .pm7_251 import PM7_251Loader
from .constants import (
    HARTREE_TO_KCALMOL,
    KJMOL_TO_KCALMOL,
    KCALMOL_TO_KJMOL,
    H_PROTON_HA,
    HOF_PROTON_KCALMOL,
    ZPE_SCALE_FACTOR,
)

__all__ = [
    "B3LYP1185Loader",
    "B3LYP251Loader",
    "PM7_1185Loader",
    "PM7_251Loader",
    "HARTREE_TO_KCALMOL",
    "KJMOL_TO_KCALMOL",
    "KCALMOL_TO_KJMOL",
    "H_PROTON_HA",
    "HOF_PROTON_KCALMOL",
    "ZPE_SCALE_FACTOR",
]
