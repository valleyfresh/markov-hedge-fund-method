# Phase 2 — Walk-Forward Backtest Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a realistic walk-forward backtest engine that applies the SPY gate, conviction-scaled position sizing, ATR stops, and full transaction costs to produce a trustworthy Sharpe / drawdown report for the watchlist.

**Architecture:** `WalkForwardBacktester` uses `RegimeEngine` from Phase 1. It slides a 252-day train / 21-day test window forward in 21-day steps. The SPY regime is computed on the same train window split so there is zero lookahead. Results are written to `results/backtest_YYYY-MM-DD.csv`.

**Tech Stack:** Python 3.10+, numpy, pandas, yfinance, pytest, PyYAML

**Prerequisite:** Phase 1 complete — `markov/engine.py` and `config.yaml` exist and all Phase 1 tests pass.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `markov/backtest.py` | `Trade`, `BacktestResult`, `WalkForwardBacktester` |
| Create | `tests/test_backtest.py` | Full test suite for backtester |
| Create | `markov/run_backtest.py` | CLI entry point — fetch, run, write CSV |
| Create | `results/.gitkeep` | Ensure results dir is tracked |

---

## Task 1: Trade and BacktestResult dataclasses (TDD)

**Files:**
- Create: `markov/backtest.py` (dataclasses only)
- Create: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_backtest.py
import pandas as pd
import pytest
from markov.backtest import BacktestResult, Trade


def _make_trade(side: str = "long", net_return: float = 0.05, risk_pct: float = 0.01) -> Trade:
    return Trade(
        ticker="SPY",
        side=side,
        entry_date=pd.Timestamp("2024-01-02"),
        exit_date=pd.Timestamp("2024-01-15"),
        entry_price=470.0,
        exit_price=494.0 if side == "long" else 447.0,
        stop_price=455.0 if side == "long" else 485.0,
        conviction=0.65,
        risk_pct=risk_pct,
        shares=12.5,
        gross_return=0.051,
        net_return=net_return,
    )


def test_trade_fields_accessible():
    t = _make_trade()
    assert t.ticker == "SPY"
    assert t.side == "long"
    assert t.net_return == 0.05


def test_backtest_result_defaults_to_empty():
    r = BacktestResult(ticker="SPY")
    assert r.trades == []
    assert r.sharpe == 0.0
    assert r.max_drawdown == 0.0
    assert r.win_rate == 0.0
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_backtest.py -v
```

Expected: FAILED — `markov.backtest` not found.

- [ ] **Step 3: Create the dataclasses**

```python
# markov/backtest.py
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Trade:
    ticker: str
    side: str          # "long" or "short"
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    stop_price: float
    conviction: float
    risk_pct: float
    shares: float
    gross_return: float
    net_return: float  # after all costs


@dataclass
class BacktestResult:
    ticker: str
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_backtest.py -v
```

Expected: 2 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add markov/backtest.py tests/test_backtest.py
git commit -m "feat: add Trade and BacktestResult dataclasses (TDD)"
```

---

## Task 2: Conviction tiers + ATR stop + position sizing (TDD)

**Files:**
- Modify: `markov/backtest.py` (add `WalkForwardBacktester` skeleton + sizing helpers)
- Modify: `tests/test_backtest.py` (add tests)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_backtest.py`:

```python
import yaml
from markov.backtest import WalkForwardBacktester


def _cfg() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def test_conviction_tier_low():
    bt = WalkForwardBacktester(_cfg())
    assert bt._risk_pct(0.20) == pytest.approx(0.01)


def test_conviction_tier_medium():
    bt = WalkForwardBacktester(_cfg())
    assert bt._risk_pct(0.55) == pytest.approx(0.02)


def test_conviction_tier_high():
    bt = WalkForwardBacktester(_cfg())
    assert bt._risk_pct(0.75) == pytest.approx(0.03)


def test_position_size_formula():
    bt = WalkForwardBacktester(_cfg())
    # risk_dollars = 10000 * 0.01 = 100
    # stop_distance = |470 - 455| = 15
    # shares = 100 / 15 ≈ 6.67
    shares = bt._position_size(
        portfolio_value=10000,
        risk_pct=0.01,
        entry_price=470.0,
        stop_price=455.0,
    )
    assert shares == pytest.approx(100.0 / 15.0)


def test_position_size_zero_when_stop_at_entry():
    bt = WalkForwardBacktester(_cfg())
    shares = bt._position_size(10000, 0.01, 470.0, 470.0)
    assert shares == 0.0


def test_atr_computed_correctly():
    bt = WalkForwardBacktester(_cfg())
    # Build OHLCV where H-L is always 2.0 (TR = H-L dominates)
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    ohlcv = pd.DataFrame({
        "Open":  [100.0] * n,
        "High":  [101.0] * n,
        "Low":   [99.0]  * n,
        "Close": [100.0] * n,
        "Volume":[1_000_000] * n,
    }, index=idx)
    atr = bt._atr(ohlcv, period=14)
    # After 14 bars, ATR should equal 2.0 (TR is constant)
    assert atr.iloc[-1] == pytest.approx(2.0, rel=1e-3)
```

- [ ] **Step 2: Run to verify new tests fail**

```bash
pytest tests/test_backtest.py -v
```

Expected: 2 PASSED (from Task 1), 6 FAILED (WalkForwardBacktester not defined).

- [ ] **Step 3: Add WalkForwardBacktester skeleton with helper methods**

Append to `markov/backtest.py`:

```python
class WalkForwardBacktester:
    def __init__(self, config: dict) -> None:
        self.lookback_days        = config["lookback_days"]
        self.threshold_pct        = config["threshold_pct"]
        self.train_days           = config["backtest"]["train_days"]
        self.test_days            = config["backtest"]["test_days"]
        self.tx_cost              = config["backtest"]["tx_cost_pct"]
        self.slippage             = config["backtest"]["slippage_pct"]
        self.short_borrow         = config["backtest"]["short_borrow_daily_pct"]
        self.portfolio_value      = config["risk"]["portfolio_value"]
        self.atr_period           = config["risk"]["atr_period"]
        self.atr_multiplier       = config["risk"]["atr_multiplier"]
        self.conviction_tiers     = config["risk"]["conviction_tiers"]

    def _risk_pct(self, conviction: float) -> float:
        for tier in self.conviction_tiers.values():
            low, high, pct = tier
            if low <= conviction < high:
                return float(pct)
        return float(self.conviction_tiers["high"][2])

    def _position_size(
        self,
        portfolio_value: float,
        risk_pct: float,
        entry_price: float,
        stop_price: float,
    ) -> float:
        stop_distance = abs(entry_price - stop_price)
        if stop_distance == 0:
            return 0.0
        return (portfolio_value * risk_pct) / stop_distance

    def _atr(self, ohlcv: pd.DataFrame, period: int) -> pd.Series:
        high  = ohlcv["High"]
        low   = ohlcv["Low"]
        close = ohlcv["Close"]
        prev  = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
        ).max(axis=1)
        return tr.rolling(period).mean()
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
pytest tests/test_backtest.py -v
```

Expected: 8 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add markov/backtest.py tests/test_backtest.py
git commit -m "feat: add WalkForwardBacktester sizing helpers (TDD)"
```

---

## Task 3: SPY gate logic (TDD)

**Files:**
- Modify: `tests/test_backtest.py` (add SPY gate tests)
- Modify: `markov/backtest.py` (add `_apply_gate`)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_backtest.py`:

```python
from markov.engine import Regime


def test_spy_bull_suppresses_short():
    bt = WalkForwardBacktester(_cfg())
    assert bt._apply_gate(ticker_regime=Regime.Bear, spy_regime=Regime.Bull, side="short") is False


def test_spy_bear_suppresses_long():
    bt = WalkForwardBacktester(_cfg())
    assert bt._apply_gate(ticker_regime=Regime.Bull, spy_regime=Regime.Bear, side="long") is False


def test_spy_sideways_allows_long():
    bt = WalkForwardBacktester(_cfg())
    assert bt._apply_gate(ticker_regime=Regime.Bull, spy_regime=Regime.Sideways, side="long") is True


def test_spy_sideways_allows_short():
    bt = WalkForwardBacktester(_cfg())
    assert bt._apply_gate(ticker_regime=Regime.Bear, spy_regime=Regime.Sideways, side="short") is True


def test_spy_bull_allows_long():
    bt = WalkForwardBacktester(_cfg())
    assert bt._apply_gate(ticker_regime=Regime.Bull, spy_regime=Regime.Bull, side="long") is True


def test_spy_bear_allows_short():
    bt = WalkForwardBacktester(_cfg())
    assert bt._apply_gate(ticker_regime=Regime.Bear, spy_regime=Regime.Bear, side="short") is True
```

- [ ] **Step 2: Run to verify new tests fail**

```bash
pytest tests/test_backtest.py -v
```

Expected: 8 PASSED, 6 FAILED.

- [ ] **Step 3: Add _apply_gate to WalkForwardBacktester**

Add this method to the `WalkForwardBacktester` class in `markov/backtest.py`:

```python
    def _apply_gate(self, ticker_regime: "Regime", spy_regime: "Regime", side: str) -> bool:
        from markov.engine import Regime
        if side == "long"  and spy_regime == Regime.Bear:
            return False
        if side == "short" and spy_regime == Regime.Bull:
            return False
        return True
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
pytest tests/test_backtest.py -v
```

Expected: 14 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add markov/backtest.py tests/test_backtest.py
git commit -m "feat: add SPY gate logic to WalkForwardBacktester (TDD)"
```

---

## Task 4: P&L calculation with costs (TDD)

**Files:**
- Modify: `tests/test_backtest.py` (add P&L tests)
- Modify: `markov/backtest.py` (add `_net_return`)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_backtest.py`:

```python
def test_long_net_return_deducts_costs():
    bt = WalkForwardBacktester(_cfg())
    # Long: buy at 100, sell at 110 → gross = 10/100 = 0.10
    # round-trip cost = 2 * (tx_cost + slippage) = 2 * (0.001 + 0.0005) = 0.003
    # net = 0.10 - 0.003 = 0.097
    net = bt._net_return(side="long", entry=100.0, exit_=110.0, days_held=5)
    assert net == pytest.approx(0.10 - 2 * (0.001 + 0.0005), rel=1e-6)


def test_short_net_return_deducts_costs_and_borrow():
    bt = WalkForwardBacktester(_cfg())
    # Short: sell at 100, buy back at 90 → gross = (100-90)/100 = 0.10
    # borrow cost = 0.0001 * 5 days = 0.0005
    # round-trip cost = 0.003
    # net = 0.10 - 0.003 - 0.0005 = 0.0965
    net = bt._net_return(side="short", entry=100.0, exit_=90.0, days_held=5)
    assert net == pytest.approx(0.10 - 2 * (0.001 + 0.0005) - 0.0001 * 5, rel=1e-6)


def test_long_losing_trade_net_return_is_negative():
    bt = WalkForwardBacktester(_cfg())
    net = bt._net_return(side="long", entry=100.0, exit_=95.0, days_held=3)
    assert net < 0


def test_short_losing_trade_net_return_is_negative():
    bt = WalkForwardBacktester(_cfg())
    net = bt._net_return(side="short", entry=100.0, exit_=105.0, days_held=3)
    assert net < 0
```

- [ ] **Step 2: Run to verify new tests fail**

```bash
pytest tests/test_backtest.py -v
```

Expected: 14 PASSED, 4 FAILED.

- [ ] **Step 3: Add _net_return to WalkForwardBacktester**

Add to the `WalkForwardBacktester` class in `markov/backtest.py`:

```python
    def _net_return(self, side: str, entry: float, exit_: float, days_held: int) -> float:
        round_trip_cost = 2.0 * (self.tx_cost + self.slippage)
        if side == "long":
            gross = (exit_ - entry) / entry
        else:
            gross = (entry - exit_) / entry - self.short_borrow * days_held
        return gross - round_trip_cost
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_backtest.py -v
```

Expected: 18 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add markov/backtest.py tests/test_backtest.py
git commit -m "feat: add P&L calculation with tx costs and borrow (TDD)"
```

---

## Task 5: Metrics — Sharpe, drawdown, win rate (TDD)

**Files:**
- Modify: `tests/test_backtest.py` (add metrics tests)
- Modify: `markov/backtest.py` (add `_compute_metrics`)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_backtest.py`:

```python
def test_metrics_win_rate():
    bt = WalkForwardBacktester(_cfg())
    trades = [
        _make_trade(net_return=0.05),
        _make_trade(net_return=-0.02),
        _make_trade(net_return=0.03),
    ]
    result = bt._compute_metrics("SPY", trades, 10150.0)
    assert result.win_rate == pytest.approx(2 / 3)


def test_metrics_avg_win_avg_loss():
    bt = WalkForwardBacktester(_cfg())
    trades = [
        _make_trade(net_return=0.06),
        _make_trade(net_return=0.04),
        _make_trade(net_return=-0.02),
    ]
    result = bt._compute_metrics("SPY", trades, 10200.0)
    assert result.avg_win  == pytest.approx(0.05)
    assert result.avg_loss == pytest.approx(-0.02)


def test_metrics_max_drawdown_negative():
    bt = WalkForwardBacktester(_cfg())
    trades = [
        _make_trade(net_return=0.10),
        _make_trade(net_return=-0.20),
        _make_trade(net_return=0.05),
    ]
    result = bt._compute_metrics("SPY", trades, 9000.0)
    assert result.max_drawdown < 0


def test_metrics_empty_trades_returns_zeros():
    bt = WalkForwardBacktester(_cfg())
    result = bt._compute_metrics("SPY", [], 10000.0)
    assert result.sharpe == 0.0
    assert result.win_rate == 0.0
    assert result.trades == []
```

- [ ] **Step 2: Run to verify new tests fail**

```bash
pytest tests/test_backtest.py -v
```

Expected: 18 PASSED, 4 FAILED.

- [ ] **Step 3: Add _compute_metrics to WalkForwardBacktester**

Add to the `WalkForwardBacktester` class in `markov/backtest.py`:

```python
    def _compute_metrics(
        self, ticker: str, trades: list[Trade], final_portfolio: float
    ) -> BacktestResult:
        if not trades:
            return BacktestResult(ticker=ticker)

        returns = [t.net_return for t in trades]
        wins   = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        pv = float(self.portfolio_value)
        equity = [pv]
        for t in trades:
            pv *= 1.0 + t.net_return * t.risk_pct
            equity.append(pv)
        equity_series = pd.Series(equity, dtype=float)

        peak     = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        max_dd   = float(drawdown.min())

        arr = np.array(returns, dtype=float)
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        sharpe = float(arr.mean() / std * np.sqrt(252)) if std > 0 else 0.0

        return BacktestResult(
            ticker=ticker,
            trades=trades,
            equity_curve=equity_series,
            sharpe=sharpe,
            max_drawdown=max_dd,
            win_rate=len(wins) / len(returns),
            avg_win=float(np.mean(wins))   if wins   else 0.0,
            avg_loss=float(np.mean(losses)) if losses else 0.0,
        )
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_backtest.py -v
```

Expected: 22 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add markov/backtest.py tests/test_backtest.py
git commit -m "feat: add backtest metrics — Sharpe, drawdown, win rate (TDD)"
```

---

## Task 6: Full walk-forward loop + run_backtest CLI

**Files:**
- Modify: `markov/backtest.py` (add `run` + `_run_ticker`)
- Create: `markov/run_backtest.py`
- Create: `results/.gitkeep`

This task wires everything together. No unit tests for the loop itself (it calls `RegimeEngine` which is already tested); we verify by running on SPY in Task 7.

- [ ] **Step 1: Add the full walk-forward loop to WalkForwardBacktester**

Append to the `WalkForwardBacktester` class in `markov/backtest.py`:

```python
    def run(
        self,
        ticker_prices: dict[str, pd.DataFrame],
        spy_prices: pd.DataFrame,
    ) -> dict[str, BacktestResult]:
        return {
            ticker: self._run_ticker(ticker, ohlcv, spy_prices)
            for ticker, ohlcv in ticker_prices.items()
        }

    def _run_ticker(
        self, ticker: str, ohlcv: pd.DataFrame, spy_ohlcv: pd.DataFrame
    ) -> BacktestResult:
        from markov.engine import Regime, RegimeEngine

        close     = ohlcv["Close"]
        spy_close = spy_ohlcv["Close"]
        atr_series = self._atr(ohlcv, self.atr_period)
        trades: list[Trade] = []
        portfolio_value = float(self.portfolio_value)
        total = len(close)

        # Sliding window: train=[start-train_days, start), test=[start, start+test_days)
        start = self.train_days
        while start + self.test_days <= total:
            train_close     = close.iloc[start - self.train_days : start]
            spy_train_close = spy_close.iloc[start - self.train_days : start]

            ticker_eng = RegimeEngine(self.lookback_days, self.threshold_pct).fit(train_close)
            spy_eng    = RegimeEngine(self.lookback_days, self.threshold_pct).fit(spy_train_close)

            position_side: str | None = None
            entry_price = stop_price = conviction = risk_pct = shares = 0.0
            entry_date: pd.Timestamp | None = None

            end = min(start + self.test_days, total - 1)
            for i in range(start, end):
                # Re-fit on expanding window within test period (no lookahead)
                win_start = max(0, i + 1 - self.train_days)
                ticker_eng.fit(close.iloc[win_start : i + 1])
                spy_eng.fit(spy_close.iloc[win_start : i + 1])

                ticker_regime = ticker_eng.current_regime()
                spy_regime    = spy_eng.current_regime()
                direction, conv = ticker_eng.signal()

                price = float(close.iloc[i])
                atr   = float(atr_series.iloc[i])
                if np.isnan(atr) or atr == 0:
                    atr = price * 0.02  # fallback: 2% of price

                if position_side is None:
                    # Entry check
                    if direction == Regime.Bull and self._apply_gate(ticker_regime, spy_regime, "long"):
                        rp    = self._risk_pct(conv)
                        stop  = price - self.atr_multiplier * atr
                        entry_price, stop_price = price * (1 + self.slippage), stop
                        entry_date, conviction, risk_pct = close.index[i], conv, rp
                        shares = self._position_size(portfolio_value, rp, entry_price, stop_price)
                        position_side = "long"
                    elif direction == Regime.Bear and self._apply_gate(ticker_regime, spy_regime, "short"):
                        rp    = self._risk_pct(conv)
                        stop  = price + self.atr_multiplier * atr
                        entry_price, stop_price = price * (1 - self.slippage), stop
                        entry_date, conviction, risk_pct = close.index[i], conv, rp
                        shares = self._position_size(portfolio_value, rp, entry_price, stop_price)
                        position_side = "short"
                else:
                    # Exit check
                    exit_triggered = (
                        (position_side == "long"  and (ticker_regime == Regime.Bear or spy_regime == Regime.Bear or price <= stop_price)) or
                        (position_side == "short" and (ticker_regime == Regime.Bull or spy_regime == Regime.Bull or price >= stop_price))
                    )
                    if exit_triggered:
                        exit_price = price * (1 - self.slippage if position_side == "long" else 1 + self.slippage)
                        days_held  = max(1, (close.index[i] - entry_date).days)
                        net        = self._net_return(position_side, entry_price, exit_price, days_held)
                        trades.append(Trade(
                            ticker=ticker, side=position_side,
                            entry_date=entry_date, exit_date=close.index[i],
                            entry_price=entry_price, exit_price=exit_price,
                            stop_price=stop_price, conviction=conviction,
                            risk_pct=risk_pct, shares=shares,
                            gross_return=(exit_price - entry_price) / entry_price if position_side == "long"
                                         else (entry_price - exit_price) / entry_price,
                            net_return=net,
                        ))
                        portfolio_value *= 1.0 + net * risk_pct
                        position_side = None

            start += self.test_days

        return self._compute_metrics(ticker, trades, portfolio_value)
```

- [ ] **Step 2: Create the CLI runner**

```python
# markov/run_backtest.py
"""Run walk-forward backtest: python -m markov.run_backtest"""
from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

import yfinance as yf

from markov.backtest import WalkForwardBacktester
from markov.config import load_config


def main() -> None:
    cfg = load_config("config.yaml")
    watchlist = cfg["watchlist"]
    spy_ticker = cfg["spy_ticker"]

    all_tickers = list({spy_ticker} | set(watchlist))
    print(f"Fetching {len(all_tickers)} tickers from yfinance...")
    raw = yf.download(all_tickers, period="5y", auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns when >1 ticker
    spy_ohlcv = raw.xs(spy_ticker, axis=1, level=1) if isinstance(raw.columns, __import__("pandas").MultiIndex) else raw
    ticker_ohlcv = {
        t: raw.xs(t, axis=1, level=1)
        for t in watchlist
        if t in raw.columns.get_level_values(1)
    }

    bt = WalkForwardBacktester(cfg)
    results = bt.run(ticker_ohlcv, spy_ohlcv)

    Path("results").mkdir(exist_ok=True)
    out_path = Path(f"results/backtest_{date.today()}.csv")

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "sharpe", "max_drawdown", "win_rate", "avg_win", "avg_loss", "n_trades"])
        for ticker, r in results.items():
            writer.writerow([
                ticker,
                f"{r.sharpe:.3f}",
                f"{r.max_drawdown:.3f}",
                f"{r.win_rate:.3f}",
                f"{r.avg_win:.3f}",
                f"{r.avg_loss:.3f}",
                len(r.trades),
            ])

    print(f"\nResults written to {out_path}\n")
    for ticker, r in results.items():
        print(f"  {ticker:6s}  Sharpe={r.sharpe:+.2f}  MaxDD={r.max_drawdown:.1%}  WinRate={r.win_rate:.0%}  Trades={len(r.trades)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create results placeholder**

```bash
mkdir -p results && touch results/.gitkeep
```

- [ ] **Step 4: Commit**

```bash
git add markov/backtest.py markov/run_backtest.py results/.gitkeep
git commit -m "feat: add walk-forward loop and run_backtest CLI"
```

---

## Task 7: Run backtest on SPY + watchlist

- [ ] **Step 1: Run the backtest**

```bash
python -m markov.run_backtest
```

Expected output (values will vary): 
```
Fetching 6 tickers from yfinance...

Results written to results/backtest_2026-05-20.csv

  SPY     Sharpe=+0.42  MaxDD=-8.3%  WinRate=54%  Trades=47
  AAPL    Sharpe=+0.61  MaxDD=-12.1% WinRate=58%  Trades=39
  ...
```

If any ticker shows `Trades=0`, reduce `train_days` in `config.yaml` to `180` and re-run. If `Sharpe=nan`, the ticker likely has insufficient history — remove it from the watchlist.

- [ ] **Step 2: Run full test suite to confirm nothing broke**

```bash
pytest tests/ -v
```

Expected: all tests PASSED.

- [ ] **Step 3: Commit results**

```bash
git add results/
git commit -m "chore: add initial backtest results"
```

---

## Phase 2 complete

The backtest engine is fully operational. Phase 3 (live pipeline) can now be implemented independently using the same `RegimeEngine` from Phase 1.
