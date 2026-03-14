"""
Abstract base class for all dataset loaders.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseDatasetLoader(ABC):
    """
    Abstract base class for loading proton affinity datasets.

    All loaders take a data directory path and provide a standardized
    interface for loading, validating, and summarizing the data.
    """

    def __init__(self, data_dir: str | Path):
        """
        Parameters
        ----------
        data_dir : str or Path
            Root directory for this dataset (e.g., 'data/1185_molecules').
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load the dataset and return a standardized DataFrame.

        Returns
        -------
        pd.DataFrame
            Loaded dataset with standardized column names.
        """
        pass

    def validate(self) -> bool:
        """
        Check that all required files exist and have expected structure.

        Returns
        -------
        bool
            True if validation passes.
        """
        for path in self._required_files():
            if not path.exists():
                logger.error(f"Required file not found: {path}")
                return False
        return True

    def summary(self, df: pd.DataFrame = None) -> dict:
        """
        Return summary statistics for the loaded data.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Pre-loaded DataFrame. If None, calls self.load().

        Returns
        -------
        dict
            Dictionary with keys: n_rows, n_cols, pa_columns, pa_stats.
        """
        if df is None:
            df = self.load()

        info = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": list(df.columns),
        }

        # Auto-detect PA columns and compute stats
        pa_cols = [c for c in df.columns if "pa_kcal" in c.lower() or "exp_pa" in c.lower()]
        if pa_cols:
            info["pa_stats"] = {}
            for col in pa_cols:
                valid = df[col].dropna()
                info["pa_stats"][col] = {
                    "count": len(valid),
                    "mean": float(valid.mean()) if len(valid) > 0 else None,
                    "std": float(valid.std()) if len(valid) > 0 else None,
                    "min": float(valid.min()) if len(valid) > 0 else None,
                    "max": float(valid.max()) if len(valid) > 0 else None,
                }

        return info

    def _required_files(self) -> list[Path]:
        """
        Return list of required file paths for validation.
        Override in subclasses.
        """
        return []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data_dir='{self.data_dir}')"
