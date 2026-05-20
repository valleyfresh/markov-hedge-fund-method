# Phase 1 — Engine Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the existing Markov library into the `markov/` package, fix all 4 known bugs, and produce a tested `RegimeEngine` class that is the single source of truth for both the backtest (Phase 2) and live pipeline (Phase 3).

**Architecture:** Pure functions from `scripts/markov_regime.py` are wrapped into a `RegimeEngine` class with a stable public API. The original file is kept in `scripts/` as read-only reference. A single `config.yaml` replaces all hardcoded parameters. Pine Script receives three targeted line edits.

**Tech Stack:** Python 3.10+, numpy, pandas, pytest, PyYAML

**Dependencies:** None (this is Phase 1 — Phases 2 and 3 depend on it)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `markov/__init__.py` | Package entry point, re-exports `RegimeEngine`, `Regime` |
| Create | `markov/engine.py` | `Regime` enum + `RegimeEngine` class |
| Create | `tests/__init__.py` | Test package marker |
| Create | `tests/test_engine.py` | Full test suite for `RegimeEngine` |
| Create | `config.yaml` | Single source of truth for all parameters |
| Modify | `pine-script/markov-hedge-fund-method.pine` | 3 targeted bug fixes |
| Keep   | `scripts/markov_regime.py` | Reference only — do not import |

---

## Task 1: Package skeleton + config.yaml

**Files:**
- Create: `markov/__init__.py`
- Create: `tests/__init__.py`
- Create: `config.yaml`

- [ ] **Step 1: Create the package files**

```python
# markov/__init__.py
from markov.engine import Regime, RegimeEngine

__all__ = ["Regime", "RegimeEngine"]
```

```python
# tests/__init__.py
```

- [ ] **Step 2: Create config.yaml**

```yaml
# config.yaml
watchlist: [AAPL, NVDA, MSFT, AMZN, META]
spy_ticker: SPY
lookback_days: 20
threshold_pct: 0.05

risk:
  portfolio_value: 10000
  atr_period: 14
  atr_multiplier: 1.5
  conviction_tiers:
    low:    [0.00, 0.40, 0.01]
    medium: [0.40, 0.70, 0.02]
    high:   [0.70, 1.00, 0.03]

backtest:
  train_days: 252
  test_days: 21
  tx_cost_pct: 0.001
  slippage_pct: 0.0005
  short_borrow_daily_pct: 0.0001

telegram:
  token: "${TELEGRAM_TOKEN}"
  chat_id: "${TELEGRAM_CHAT_ID}"
```

- [ ] **Step 3: Verify Python can find the package**

```bash
cd /home/dlee1/repo/markov-hedge-fund-method
python -c "import markov; print('ok')"
```

Expected: `ok`

If you get `ModuleNotFoundError`, run from the repo root or add a `pyproject.toml`:

```toml
# pyproject.toml  (create only if the import fails)
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "markov"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy", "pandas", "yfinance", "scipy", "pyyaml", "requests"]

[tool.setuptools.packages.find]
where = ["."]
```

Then: `pip install -e .`

- [ ] **Step 4: Commit**

```bash
git add markov/__init__.py tests/__init__.py config.yaml
git commit -m "feat: add markov package skeleton and config.yaml"
```

---

## Task 2: Regime enum + label_regimes (TDD)

**Files:**
- Create: `markov/engine.py` (partial — Regime + _label only)
- Modify: `tests/test_engine.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_engine.py
import pandas as pd
import pytest
from markov.engine import Regime, RegimeEngine


def _prices(daily_pct: float, n: int = 60, start: float = 100.0) -> pd.Series:
    """Build a price series with a fixed daily return for n bars."""
    import numpy as np
    prices = [start * (1 + daily_pct) ** i for i in range(n)]
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx)


def _mixed_prices(n: int = 120) -> pd.Series:
    """Alternating up/down blocks to generate all three regimes."""
    import numpy as np
    block = n // 3
    up   = [100.0 * (1.01) ** i for i in range(block)]
    flat = [up[-1] * (1.001) ** i for i in range(block)]
    down = [flat[-1] * (0.99) ** i for i in range(block)]
    prices = up + flat + down
    idx = pd.date_range("2023-01-01", periods=len(prices), freq="B")
    return pd.Series(prices, index=idx)


def test_regime_enum_values():
    assert Regime.Bull == 0
    assert Regime.Bear == 1
    assert Regime.Sideways == 2


def test_current_regime_bull():
    # 1%/day for 60 bars → 20-day return ≈ 22% → above 5% threshold
    engine = RegimeEngine(lookback_days=20, threshold_pct=0.05)
    engine.fit(_prices(0.01))
    assert engine.current_regime() == Regime.Bull


def test_current_regime_bear():
    # -1%/day → 20-day return ≈ -18% → below -5% threshold
    engine = RegimeEngine(lookback_days=20, threshold_pct=0.05)
    engine.fit(_prices(-0.01))
    assert engine.current_regime() == Regime.Bear


def test_current_regime_sideways():
    # 0.1%/day → 20-day return ≈ 2% → within ±5% band
    engine = RegimeEngine(lookback_days=20, threshold_pct=0.05)
    engine.fit(_prices(0.001))
    assert engine.current_regime() == Regime.Sideways


def test_raises_before_fit():
    engine = RegimeEngine()
    with pytest.raises(RuntimeError, match="fit"):
        engine.current_regime()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_engine.py -v
```

Expected: `ERRORS` — `markov.engine` module not found (engine.py doesn't exist yet).

- [ ] **Step 3: Implement Regime enum and RegimeEngine._label + .fit + .current_regime**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_engine.py -v
```

Expected: 5 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add markov/engine.py tests/test_engine.py
git commit -m "feat: add RegimeEngine with label + fit + current_regime (TDD)"
```

---

## Task 3: Transition matrix, signal(), stationary_dist(), n_step_forecast() (TDD)

**Files:**
- Modify: `tests/test_engine.py` (add tests)
- Modify: `markov/engine.py` (add methods)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_engine.py`:

```python
def test_transition_matrix_rows_sum_to_one():
    engine = RegimeEngine(lookback_days=20, threshold_pct=0.05)
    engine.fit(_mixed_prices())
    for row in engine._matrix:
        assert abs(row.sum() - 1.0) < 1e-9


def test_signal_returns_regime_and_conviction_in_range():
    engine = RegimeEngine(lookback_days=20, threshold_pct=0.05)
    engine.fit(_prices(0.01))
    regime, conviction = engine.signal()
    assert isinstance(regime, Regime)
    assert 0.0 <= conviction <= 1.0


def test_signal_bull_when_bull_prob_exceeds_bear():
    # Strong uptrend → current state is Bull → P[Bull, Bull] > P[Bull, Bear]
    engine = RegimeEngine(lookback_days=20, threshold_pct=0.05)
    engine.fit(_prices(0.01))
    regime, _ = engine.signal()
    assert regime == Regime.Bull


def test_signal_bear_when_bear_prob_exceeds_bull():
    engine = RegimeEngine(lookback_days=20, threshold_pct=0.05)
    engine.fit(_prices(-0.01))
    regime, _ = engine.signal()
    assert regime == Regime.Bear


def test_stationary_dist_sums_to_one():
    engine = RegimeEngine(lookback_days=20, threshold_pct=0.05)
    engine.fit(_mixed_prices())
    dist = engine.stationary_dist()
    assert set(dist.keys()) == {"Bull", "Bear", "Sideways"}
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    for v in dist.values():
        assert v >= 0.0


def test_n_step_forecast_sums_to_one():
    engine = RegimeEngine(lookback_days=20, threshold_pct=0.05)
    engine.fit(_mixed_prices())
    forecast = engine.n_step_forecast(5)
    assert set(forecast.keys()) == {"Bull", "Bear", "Sideways"}
    assert abs(sum(forecast.values()) - 1.0) < 1e-6


def test_n_step_forecast_n1_matches_matrix_row():
    engine = RegimeEngine(lookback_days=20, threshold_pct=0.05)
    engine.fit(_mixed_prices())
    forecast = engine.n_step_forecast(1)
    current = int(engine.current_regime())
    assert abs(forecast["Bull"]     - engine._matrix[current, Regime.Bull])     < 1e-9
    assert abs(forecast["Bear"]     - engine._matrix[current, Regime.Bear])     < 1e-9
    assert abs(forecast["Sideways"] - engine._matrix[current, Regime.Sideways]) < 1e-9
```

- [ ] **Step 2: Run tests to verify new ones fail**

```bash
pytest tests/test_engine.py -v
```

Expected: 5 PASSED (from Task 2), 7 FAILED (new tests — methods not yet defined).

- [ ] **Step 3: Add the missing methods to RegimeEngine**

Add to the `RegimeEngine` class in `markov/engine.py` (after `current_regime`):

```python
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
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
pytest tests/test_engine.py -v
```

Expected: 12 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add markov/engine.py tests/test_engine.py
git commit -m "feat: add signal, stationary_dist, n_step_forecast to RegimeEngine (TDD)"
```

---

## Task 4: Fix Pine Script (3 targeted edits)

**Files:**
- Modify: `pine-script/markov-hedge-fund-method.pine`

No automated tests for Pine — verify visually in TradingView after each edit.

### Fix 1 — Return type: log → simple (Bug 2)

- [ ] **Step 1: Replace line 154**

Find:
```pine
log_ret = math.log(close / close[lookback_window])
regime  = na(log_ret) ? int(na) : log_ret > bull_threshold_pct / 100.0 ? 1 : log_ret < -bear_threshold_pct / 100.0 ? 2 : 0
```

Replace with:
```pine
simple_ret = (close - close[lookback_window]) / close[lookback_window]
regime     = na(simple_ret) ? int(na) : simple_ret > bull_threshold_pct / 100.0 ? 0 : simple_ret < -bear_threshold_pct / 100.0 ? 1 : 2
```

Note the encoding change: **0=Bull, 1=Bear, 2=Sideways** (now matches Python).

### Fix 2 — regime_name / regime_abbr helper functions (Bug 1)

- [ ] **Step 2: Update the colour/name helper functions**

Find (lines ~142–146):
```pine
regime_solid(r) => r == 1 ? c_bull_solid  : r == 2 ? c_bear_solid  : c_side_solid
regime_ribbon(r) => r == 1 ? c_bull_ribbon : r == 2 ? c_bear_ribbon : c_side_ribbon
regime_dim(r)    => r == 1 ? c_bull_dim    : r == 2 ? c_bear_dim    : c_side_dim
regime_name(r)   => r == 1 ? "Bull"        : r == 2 ? "Bear"        : "Sideways"
regime_abbr(r)   => r == 1 ? "BULL"        : r == 2 ? "BEAR"        : "SIDE"
```

Replace with (new encoding 0=Bull, 1=Bear, 2=Sideways):
```pine
regime_solid(r)  => r == 0 ? c_bull_solid  : r == 1 ? c_bear_solid  : c_side_solid
regime_ribbon(r) => r == 0 ? c_bull_ribbon : r == 1 ? c_bear_ribbon : c_side_ribbon
regime_dim(r)    => r == 0 ? c_bull_dim    : r == 1 ? c_bear_dim    : c_side_dim
regime_name(r)   => r == 0 ? "Bull"        : r == 1 ? "Bear"        : "Sideways"
regime_abbr(r)   => r == 0 ? "BULL"        : r == 1 ? "BEAR"        : "SIDE"
```

### Fix 3 — Transition matrix row labels (Bug 1, same root cause)

- [ ] **Step 3: Fix the row label in the transition matrix loop**

Find (line ~351):
```pine
            row_name = r == 0 ? "BULL" : r == 1 ? "BEAR" : "SIDE"
```

Replace with (use the now-correct `regime_abbr` helper):
```pine
            row_name = regime_abbr(r)
```

### Fix 4 — Stationary distribution variable assignments (Bug 1)

- [ ] **Step 4: Fix stat_bull / stat_bear / stat_side assignments**

Find (lines ~308–310):
```pine
    stat_bull = array.get(M, 0)
    stat_bear = array.get(M, 1)
    stat_side = array.get(M, 2)
```

Replace with (new encoding: index 0=Bull, 1=Bear, 2=Sideways):
```pine
    stat_bull = array.get(M, 0)
    stat_bear = array.get(M, 1)
    stat_side = array.get(M, 2)
```

These variable names are already correct for the *new* encoding — no change needed here. The bug was that the old encoding had 0=Sideways, which made these assignments wrong. The return-type fix (step 1) and helper fix (step 2) are sufficient once the encoding is consistent.

- [ ] **Step 5: Commit Pine fixes**

```bash
git add pine-script/markov-hedge-fund-method.pine
git commit -m "fix: align Pine encoding to 0=Bull/1=Bear/2=Sideways, use simple return"
```

- [ ] **Step 6: Manual TradingView verification**

Load the updated Pine Script in TradingView on SPY Daily. Verify:
- Regime banner shows correct label (not flipped)
- Transition matrix row headers match column headers semantically
- No compile errors in the Pine editor

---

## Task 5: Load config in engine (wire config.yaml)

**Files:**
- Create: `markov/config.py`
- Modify: `tests/test_engine.py` (add one config test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_engine.py`:

```python
def test_engine_loads_from_config():
    from markov.config import load_config
    cfg = load_config("config.yaml")
    engine = RegimeEngine(
        lookback_days=cfg["lookback_days"],
        threshold_pct=cfg["threshold_pct"],
    )
    assert engine.lookback_days == 20
    assert engine.threshold_pct == 0.05
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_engine.py::test_engine_loads_from_config -v
```

Expected: FAILED — `markov.config` not found.

- [ ] **Step 3: Create markov/config.py**

```python
# markov/config.py
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load config.yaml, expanding ${ENV_VAR} references."""
    text = Path(path).read_text()
    text = re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), text)
    return yaml.safe_load(text)
```

Update `markov/__init__.py`:

```python
# markov/__init__.py
from markov.config import load_config
from markov.engine import Regime, RegimeEngine

__all__ = ["Regime", "RegimeEngine", "load_config"]
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/ -v
```

Expected: 13 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add markov/config.py markov/__init__.py tests/test_engine.py
git commit -m "feat: add load_config, wire config.yaml to RegimeEngine"
```

---

## Phase 1 complete

Run the full test suite one final time:

```bash
pytest tests/ -v
```

Expected: all tests PASSED, no warnings about encoding mismatches.

The `markov/` package is now the canonical implementation. Phases 2 and 3 import from here — never from `scripts/`.
