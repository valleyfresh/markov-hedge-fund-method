# markov/engine.py
from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd


class Regime(IntEnum):
    Bull = 0
    Bear = 1
    Sideways = 2


class RegimeEngine:
    """Markov regime model: fit on price history, read current regime and signal."""

    def __init__(self, lookback_days: int = 20, threshold_pct: float = 0.05) -> None:
        self.lookback_days = lookback_days
        self.threshold_pct = threshold_pct
        self._matrix: np.ndarray | None = None
        self._labels: pd.Series | None = None

    def _label(self, prices: pd.Series) -> pd.Series:
        ret = prices.pct_change(self.lookback_days)
        labels = pd.Series(int(Regime.Sideways), index=prices.index, dtype=int)
        labels[ret > self.threshold_pct] = int(Regime.Bull)
        labels[ret < -self.threshold_pct] = int(Regime.Bear)
        return labels.loc[ret.notna()]

    def fit(self, prices: pd.Series) -> "RegimeEngine":
        labels = self._label(prices)
        matrix = np.zeros((3, 3), dtype=float)
        arr = labels.to_numpy(dtype=int)
        for i in range(len(arr) - 1):
            matrix[arr[i], arr[i + 1]] += 1.0
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self._matrix = matrix / row_sums
        self._labels = labels
        return self

    def current_regime(self) -> Regime:
        if self._labels is None:
            raise RuntimeError("Call fit() before reading regime.")
        return Regime(int(self._labels.iloc[-1]))
