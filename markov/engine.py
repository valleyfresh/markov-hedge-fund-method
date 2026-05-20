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
        n = len(Regime)
        matrix = np.zeros((n, n), dtype=float)
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

    def signal(self) -> tuple[Regime, float]:
        if self._matrix is None:
            raise RuntimeError("Call fit() before reading signal.")
        current = int(self.current_regime())
        row = self._matrix[current]
        bull_prob = row[Regime.Bull]
        bear_prob = row[Regime.Bear]
        conviction = float(abs(bull_prob - bear_prob))
        direction = Regime.Bull if bull_prob >= bear_prob else Regime.Bear
        return direction, conviction

    def stationary_dist(self) -> dict[str, float]:
        if self._matrix is None:
            raise RuntimeError("Call fit() before reading stationary dist.")
        eigvals, eigvecs = np.linalg.eig(self._matrix.T)
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        pi = np.abs(np.real(eigvecs[:, idx]))
        pi = pi / pi.sum()
        return {
            "Bull":     float(pi[Regime.Bull]),
            "Bear":     float(pi[Regime.Bear]),
            "Sideways": float(pi[Regime.Sideways]),
        }

    def n_step_forecast(self, n: int) -> dict[str, float]:
        if self._matrix is None:
            raise RuntimeError("Call fit() before forecasting.")
        current = int(self.current_regime())
        m_n = np.linalg.matrix_power(self._matrix, n)
        row = m_n[current]
        return {
            "Bull":     float(row[Regime.Bull]),
            "Bear":     float(row[Regime.Bear]),
            "Sideways": float(row[Regime.Sideways]),
        }
