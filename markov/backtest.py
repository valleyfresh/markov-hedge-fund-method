from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Trade:
    ticker: str
    side: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    stop_price: float
    conviction: float
    risk_pct: float
    shares: float
    gross_return: float
    net_return: float


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


class WalkForwardBacktester:
    def __init__(self, config: dict) -> None:
        self.lookback_days    = config["lookback_days"]
        self.threshold_pct    = config["threshold_pct"]
        self.train_days       = config["backtest"]["train_days"]
        self.test_days        = config["backtest"]["test_days"]
        self.tx_cost          = config["backtest"]["tx_cost_pct"]
        self.slippage         = config["backtest"]["slippage_pct"]
        self.short_borrow     = config["backtest"]["short_borrow_daily_pct"]
        self.portfolio_value  = config["risk"]["portfolio_value"]
        self.atr_period       = config["risk"]["atr_period"]
        self.atr_multiplier   = config["risk"]["atr_multiplier"]
        self.conviction_tiers = config["risk"]["conviction_tiers"]

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

    def _apply_gate(self, ticker_regime: "Regime", spy_regime: "Regime", side: str) -> bool:
        from markov.engine import Regime
        if side == "long"  and spy_regime == Regime.Bear:
            return False
        if side == "short" and spy_regime == Regime.Bull:
            return False
        return True

    def _net_return(self, side: str, entry: float, exit_: float, days_held: int) -> float:
        round_trip_cost = 2.0 * (self.tx_cost + self.slippage)
        if side == "long":
            gross = (exit_ - entry) / entry
        else:
            gross = (entry - exit_) / entry - self.short_borrow * days_held
        return gross - round_trip_cost

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
            avg_win=float(np.mean(wins))    if wins   else 0.0,
            avg_loss=float(np.mean(losses)) if losses else 0.0,
        )
